import os
import sys
import requests
import json
import gradio as gr
import logging
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import base64
import threading
import time
from logging.handlers import RotatingFileHandler
import functools
from PIL import Image
import io
from urllib.parse import urlparse
import tempfile

# Load environment variables
load_dotenv()

# Configure logging with rotation
def setup_logging():
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbot.log')
    handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Configuration
class Config:
    # Default values as class attributes
    API_BASE_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
    GRADIO_SERVER_PORT = int(os.getenv('GRADIO_SERVER_PORT', '7588'))
    OCR_TIMEOUT = int(os.getenv('OCR_TIMEOUT', '60'))
    CHAT_TIMEOUT = int(os.getenv('CHAT_TIMEOUT', '30'))
    MAX_HISTORY = int(os.getenv('MAX_HISTORY', '100'))
    RETRY_ATTEMPTS = int(os.getenv('RETRY_ATTEMPTS', '3'))
    REQUEST_RATE_LIMIT = int(os.getenv('REQUEST_RATE_LIMIT', '10'))
    CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', '3600'))
    MAX_INPUT_LENGTH = int(os.getenv('MAX_INPUT_LENGTH', '1000'))

    API_GENERATE_URL = f"{API_BASE_URL}/api/generate"
    API_CHAT_URL = f"{API_BASE_URL}/api/chat"

    @classmethod
    def validate(cls):
        if not (1024 <= cls.GRADIO_SERVER_PORT <= 65535):
            raise ValueError("Port must be between 1024 and 65535")
        if cls.OCR_TIMEOUT < 5:
            logger.warning(f"OCR_TIMEOUT is too small, using 5s")
            cls.OCR_TIMEOUT = 5
        if cls.CHAT_TIMEOUT < 5:
            logger.warning(f"CHAT_TIMEOUT is too small, using 5s")
            cls.CHAT_TIMEOUT = 5
        try:
            result = urlparse(cls.API_BASE_URL)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL format")
        except Exception as e:
            raise ValueError(f"Invalid API URL configuration: {str(e)}")

# Validate configuration on module load
Config.validate()

class RateLimit:
    def __init__(self, max_requests: int, per_seconds: int = 60):
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.requests = []
        self._lock = threading.Lock()

    def is_allowed(self) -> bool:
        now = time.time()
        with self._lock:
            # 清理过期的请求记录
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.per_seconds]
            
            if len(self.requests) >= self.max_requests:
                return False
                
            self.requests.append(now)
            return True

class Cache:
    def __init__(self, timeout: int = 3600):
        self.cache = {}
        self.timeout = timeout
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key in self.cache:
                timestamp, value = self.cache[key]
                if time.time() - timestamp < self.timeout:
                    return value
                del self.cache[key]
            return None

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self.cache[key] = (time.time(), value)

class Conversation:
    def __init__(self, history: List[str], max_length: int = Config.MAX_HISTORY):
        self.history = history
        self.max_length = max_length

    def add_message(self, message: str) -> None:
        self.history.append(message)
        if len(self.history) > self.max_length:
            self.history = self.history[-self.max_length:]

    def get_context(self) -> str:
        return "\n".join(self.history)

    def clear(self) -> None:
        self.history = []

class ResourceManager:
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
        self.base_dir = base_dir
        self.temp_files = set()
        self._lock = threading.Lock()
        self._setup_temp_dir()
        
    def _setup_temp_dir(self):
        """初始化临时目录"""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            # 启动时清理遗留的临时文件
            self.cleanup_all()
        except Exception as e:
            logger.error(f"Failed to setup temp directory: {e}")
            raise
            
    def add_temp_file(self, filepath: str):
        """记录临时文件"""
        with self._lock:
            self.temp_files.add(filepath)
            
    def remove_temp_file(self, filepath: str):
        """删除临时文件"""
        with self._lock:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                self.temp_files.discard(filepath)
            except Exception as e:
                logger.error(f"Failed to remove temp file {filepath}: {e}")
                
    def cleanup_all(self):
        """清理所有临时文件"""
        with self._lock:
            for filepath in list(self.temp_files):
                self.remove_temp_file(filepath)
            # 清理可能遗留的其他临时文件
            for filename in os.listdir(self.base_dir):
                try:
                    filepath = os.path.join(self.base_dir, filename)
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                except Exception as e:
                    logger.error(f"Failed to remove file {filepath}: {e}")
                    
    def create_temp_file(self, suffix: str = '') -> str:
        """创建临时文件"""
        with self._lock:
            try:
                fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=self.base_dir)
                os.close(fd)
                self.add_temp_file(temp_path)
                return temp_path
            except Exception as e:
                logger.error(f"Failed to create temp file: {e}")
                raise
                
    def __del__(self):
        """确保在对象销毁时清理临时文件"""
        self.cleanup_all()

class ProcessState:
    """处理状态管理"""
    def __init__(self):
        self._state = {}
        self._lock = threading.Lock()
        
    def set_state(self, key: str, value: Any) -> None:
        """设置状态"""
        with self._lock:
            self._state[key] = value
            
    def get_state(self, key: str, default: Any = None) -> Any:
        """获取状态"""
        with self._lock:
            return self._state.get(key, default)
            
    def clear_state(self, key: str) -> None:
        """清除状态"""
        with self._lock:
            self._state.pop(key, None)
            
    def is_processing(self, key: str) -> bool:
        """检查是否正在处理"""
        return self.get_state(key) == 'processing'

def retry_on_exception(retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (attempt + 1))  # 指数退避
            return None
        return wrapper
    return decorator

class ChatBot:
    def __init__(self):
        self.conversation = Conversation([])
        self.session = requests.Session()
        self.current_image: Optional[str] = None
        self.current_ocr_result: Optional[str] = None
        self.rate_limit = RateLimit(Config.REQUEST_RATE_LIMIT)
        self.cache = Cache(Config.CACHE_TIMEOUT)
        self.resource_manager = ResourceManager()
        self.process_state = ProcessState()
        
        # 设置会话超时
        self.session.timeout = (Config.CHAT_TIMEOUT, Config.CHAT_TIMEOUT)
        
        # 设置重试策略
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def check_ollama_service(self):
        return check_ollama_service()

    def _encode_image(self, image_path: str) -> str:
        """编码图片，支持压缩"""
        try:
            # 验证文件
            if not os.path.exists(image_path):
                raise ValueError("图片文件不存在")
                
            # 验证文件类型
            try:
                with Image.open(image_path) as img:
                    if img.format.upper() not in {'JPEG', 'PNG', 'GIF', 'WEBP'}:
                        raise ValueError("不支持的图片格式")
            except Exception as e:
                raise ValueError(f"无效的图片文件: {str(e)}")
                
            # 检查文件大小并在需要时压缩
            file_size = os.path.getsize(image_path)
            max_size = 10 * 1024 * 1024  # 10MB
            
            if file_size > max_size:
                compressed_path = self.resource_manager.create_temp_file(
                    suffix=os.path.splitext(image_path)[1]
                )
                image_path = self.compress_image(image_path, compressed_path, max_size)
            
            # 编码图片
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
                
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise
            
    def compress_image(self, input_path: str, output_path: str, max_size: int) -> str:
        """压缩图片到指定大小以下"""
        try:
            with Image.open(input_path) as img:
                format = img.format or 'JPEG'
                quality = 95
                
                while True:
                    # 保存压缩后的图片
                    img.save(output_path, format=format, quality=quality)
                    size = os.path.getsize(output_path)
                    
                    if size <= max_size or quality <= 5:
                        break
                        
                    quality -= 5
                    
                logger.info(f"Image compressed from {os.path.getsize(input_path)} to {size} bytes")
                return output_path
                
        except Exception as e:
            logger.error(f"Image compression failed: {str(e)}")
            if os.path.exists(output_path):
                self.resource_manager.remove_temp_file(output_path)
            raise

    @retry_on_exception(retries=Config.RETRY_ATTEMPTS)
    def _validate_input(self, text: str) -> str:
        """验证并清理输入文本"""
        if not text or not text.strip():
            raise ValueError("输入不能为空")
        
        text = text.strip()
        if len(text) > Config.MAX_INPUT_LENGTH:
            raise ValueError(f"输入长度不能超过{Config.MAX_INPUT_LENGTH}字符")
            
        # 移除潜在的危险字符
        text = text.replace('<', '&lt;').replace('>', '&gt;')
        return text

    def get_model_list(self) -> Tuple[List[str], str]:
        """Returns (model_list, status_message)"""
        try:
            response = self.session.get(
                f"{Config.API_BASE_URL}/api/tags",
                timeout=Config.CHAT_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            if not data or 'models' not in data or not data['models']:
                return [], "No models found. Please install at least one model using 'ollama pull model_name'"
            # Extract model names and check for vision capabilities
            models = []
            for model in data['models']:
                name = model['name']
                details = model.get('details', {})
                families = details.get('families', [])
                # Add a visual indicator for models with vision capabilities
                if 'clip' in families:
                    name = f"🖼️ {name}"
                models.append(name)
            return models, "Models refreshed successfully!"
        except requests.exceptions.ConnectionError:
            return [], "Error: Cannot connect to Ollama server. Is it running?"
        except requests.exceptions.Timeout:
            return [], "Error: Request timed out. Please try again."
        except Exception as e:
            logger.error(f"Error fetching model list: {e}")
            return [], f"Error: {str(e)}"

    def generate_response(self, prompt: str, model_name: str, image: Optional[str] = None, is_ocr: bool = False) -> str:
        """生成响应，支持OCR和问答两种模式"""
        # 生成唯一的处理ID
        process_id = f"{time.time()}_{model_name}"
        
        try:
            # 检查是否有其他处理正在进行
            if self.process_state.is_processing(model_name):
                return "Error: 模型正在处理其他请求，请稍后再试"
            
            # 设置处理状态
            self.process_state.set_state(model_name, 'processing')
            
            # 输入验证
            try:
                prompt = self._validate_input(prompt)
                model_name = self._validate_input(model_name)
            except ValueError as e:
                return f"Error: {str(e)}"

            # 请求频率限制
            if not self.rate_limit.is_allowed():
                return "Error: 请求过于频繁，请稍后再试"

            # 检查缓存
            cache_key = f"{prompt}_{model_name}_{image if image else ''}"
            cached_response = self.cache.get(cache_key)
            if cached_response:
                return cached_response

            # 原有的响应生成逻辑
            mode = "OCR" if is_ocr else "问答"
            for attempt in range(Config.RETRY_ATTEMPTS):
                try:
                    # 根据模式选择超时时间
                    timeout = Config.OCR_TIMEOUT if is_ocr else Config.CHAT_TIMEOUT
                    
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "model": model_name.replace("🖼️ ", ""),
                        "prompt": prompt,
                        "stream": True
                    }
                    
                    if image:
                        payload["images"] = [image]
                    
                    response = self.session.post(
                        Config.API_GENERATE_URL,
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=timeout
                    )
                    
                    if response.status_code != 200:
                        error_msg = f"服务器返回错误 {response.status_code}"
                        try:
                            error_data = response.json()
                            if 'error' in error_data:
                                error_msg += f": {error_data['error']}"
                        except:
                            if response.text:
                                error_msg += f": {response.text}"
                        logger.error(f"{mode}请求失败: {error_msg}")
                        return f"Error: {error_msg}"
                    
                    # Handle streaming response
                    full_response = []
                    try:
                        for line in response.iter_lines(decode_unicode=True):
                            if line:
                                try:
                                    data = json.loads(line)
                                    if 'response' in data:
                                        full_response.append(data['response'])
                                except json.JSONDecodeError as e:
                                    logger.warning(f"{mode}响应解析错误: {str(e)}, line: {line}")
                                    continue
                    except requests.exceptions.ChunkedEncodingError as e:
                        error_msg = f"{mode}流式响应中断: {str(e)}"
                        logger.error(error_msg)
                        if not full_response:
                            return f"Error: {error_msg}"
                        # 如果已经有部分响应，继续处理
                        logger.warning(f"{mode}使用已接收的部分响应")
                    
                    result = "".join(full_response).strip()
                    if not result:
                        error_msg = f"{mode}生成的响应为空"
                        logger.error(error_msg)
                        return f"Error: {error_msg}"
                    
                    # 缓存结果
                    self.cache.set(cache_key, result)
                    return result
                    
                except requests.exceptions.Timeout:
                    error_msg = f"{mode}请求超时(尝试 {attempt + 1}/{Config.RETRY_ATTEMPTS})"
                    logger.error(error_msg)
                    if attempt == Config.RETRY_ATTEMPTS - 1:
                        return f"Error: {error_msg}"
                    continue
                    
                except requests.exceptions.RequestException as e:
                    error_msg = f"{mode}请求异常: {str(e)}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
                    
                except Exception as e:
                    error_msg = f"{mode}处理过程出错: {str(e)}"
                    logger.error(error_msg)
                    return f"Error: {error_msg}"
                    
        finally:
            # 清理处理状态
            self.process_state.clear_state(model_name)

    def clear_conversation(self) -> str:
        self.conversation.clear()
        return "Conversation cleared."

    def __del__(self):
        """清理资源"""
        try:
            self.session.close()
            if hasattr(self, 'resource_manager'):
                self.resource_manager.cleanup_all()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

def check_ollama_service(max_attempts: int = 3, wait_time: int = 2) -> bool:
    """检查Ollama服务是否可用，支持多次重试"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(Config.API_BASE_URL, timeout=5)
            if response.status_code == 200:
                return True
            logger.warning(f"Ollama service check failed (attempt {attempt + 1}/{max_attempts}): Status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Ollama service connection failed (attempt {attempt + 1}/{max_attempts})")
        except requests.exceptions.Timeout:
            logger.warning(f"Ollama service timeout (attempt {attempt + 1}/{max_attempts})")
        except Exception as e:
            logger.error(f"Unexpected error checking Ollama service: {str(e)}")
            
        if attempt < max_attempts - 1:
            time.sleep(wait_time * (attempt + 1))  # 指数退避
            
    return False

def create_interface() -> None:
    try:
        chatbot = ChatBot()
        
        # 预先获取模型列表
        models, status = chatbot.get_model_list()
        if not models:
            logger.error(f"无法获取模型列表: {status}")
            raise ValueError(f"无法获取模型列表: {status}")
            
        # 分离视觉模型和普通模型
        vision_models = [m for m in models if "🖼️" in m]
        all_models = models
        
        if not vision_models:
            logger.error("未找到支持视觉功能的模型")
            raise ValueError("未找到支持视觉功能的模型，请确保安装了支持视觉功能的模型（如llava）")
        
        interface = gr.Blocks(
            title="智能文档分析系统",
            theme=gr.themes.Default(),
            css=".output-text {font-size: 16px;}"
        )

        with interface:
            # 标题
            with gr.Column(elem_classes="header"):
                gr.Markdown("# 智能文档分析系统")
            
            # OCR区域（可选）
            with gr.Column():
                gr.Markdown("### 图片识别（可选）")
                with gr.Row():
                    image_input = gr.Image(
                        type="filepath",
                        label="上传图片（可选）",
                        interactive=True
                    )
                    ocr_model_selector = gr.Dropdown(
                        choices=vision_models,
                        label="选择OCR模型（可选）",
                        value=None,
                        interactive=True
                    )
                ocr_output = gr.Textbox(
                    label="识别结果",
                    lines=4,
                    interactive=False,
                    elem_classes="output-text"
                )
            
            # 问答区域
            with gr.Column():
                gr.Markdown("### 问答")
                question_input = gr.Textbox(
                    label="输入问题",
                    lines=2,
                    interactive=True
                )
                with gr.Row():
                    chat_model_selector1 = gr.Dropdown(
                        choices=all_models,
                        label="模型1",
                        value=all_models[0] if all_models else None,
                        interactive=True
                    )
                    chat_model_selector2 = gr.Dropdown(
                        choices=all_models,
                        label="模型2（可选）",
                        value=None,
                        interactive=True
                    )
                    chat_model_selector3 = gr.Dropdown(
                        choices=all_models,
                        label="模型3（可选）",
                        value=None,
                        interactive=True
                    )
                with gr.Row():
                    chat_output1 = gr.Textbox(
                        label="模型1回答",
                        lines=4,
                        interactive=False,
                        elem_classes="output-text"
                    )
                    chat_output2 = gr.Textbox(
                        label="模型2回答",
                        lines=4,
                        interactive=False,
                        elem_classes="output-text"
                    )
                    chat_output3 = gr.Textbox(
                        label="模型3回答",
                        lines=4,
                        interactive=False,
                        elem_classes="output-text"
                    )
            
            # 状态显示
            status_text = gr.Textbox(
                label="状态",
                interactive=False,
                elem_classes="status-text"
            )
            
            # 处理按钮
            with gr.Row():
                process_button = gr.Button(
                    "开始处理",
                    variant="primary",
                    elem_classes="primary",
                    interactive=True
                )
                retry_button = gr.Button(
                    "重新问答",
                    variant="secondary",
                    elem_classes="secondary",
                    interactive=True
                )

            # 存储OCR结果用于重试
            ocr_result_store = gr.State(value=None)

            # 处理函数：一键处理所有流程
            def process_all(image_path, ocr_model, question, chat_model1, chat_model2, chat_model3, progress=gr.Progress()):
                """一键处理所有流程"""
                # 基本验证
                if not question.strip():
                    return (
                        "",
                        "",
                        "",
                        "",
                        "请输入问题",
                        None
                    )

                if not any([chat_model1, chat_model2, chat_model3]):
                    return (
                        "",
                        "",
                        "",
                        "",
                        "请至少选择一个问答模型",
                        None
                    )

                results = {
                    "ocr_output": "",
                    "chat_output1": "",
                    "chat_output2": "",
                    "chat_output3": "",
                    "status_text": "处理中...",
                    "ocr_result_store": None
                }

                try:
                    chatbot = ChatBot()
                    qa_prompt = question
                    
                    # OCR处理（完全可选）
                    if image_path and ocr_model:
                        try:
                            progress(0.2, desc="正在进行图片识别...")
                            base64_image = chatbot._encode_image(image_path)
                            ocr_prompt = "请识别这张图片中的文字内容，尽可能保持原有格式。"
                            ocr_result = chatbot.generate_response(ocr_prompt, ocr_model, base64_image, is_ocr=True)
                            
                            # 清理OCR结果格式
                            if ocr_result and not ocr_result.startswith("Error:"):
                                # 移除多余的空行,保留段落格式
                                ocr_result = "\n".join(line for line in ocr_result.splitlines() if line.strip())
                                results["ocr_output"] = ocr_result
                                results["ocr_result_store"] = ocr_result
                                # 将OCR结果作为补充信息添加到问题中
                                qa_prompt = f"{question}\n\n补充信息：\n{ocr_result}"
                            else:
                                # OCR失败,记录错误但继续问答
                                logger.error(f"OCR识别失败: {ocr_result}")
                                results["ocr_output"] = ocr_result
                                results["ocr_result_store"] = None
                                qa_prompt = question
                        except Exception as e:
                            error_msg = f"OCR处理出错: {str(e)}"
                            logger.error(error_msg)
                            results["ocr_output"] = f"错误: {error_msg}"
                            results["ocr_result_store"] = None
                            # OCR错误不影响问答继续进行
                            qa_prompt = question
                    
                    # 问答处理
                    progress(0.4, desc="准备进行问答...")
                    total_models = sum(1 for m in [chat_model1, chat_model2, chat_model3] if m)
                    current_model = 0
                    
                    if chat_model1:
                        current_model += 1
                        progress(0.4 + 0.6 * (current_model / total_models), desc=f"获取模型{current_model}回答...")
                        try:
                            results["chat_output1"] = chatbot.generate_response(qa_prompt, chat_model1)
                        except Exception as e:
                            logger.error(f"模型1回答出错: {str(e)}")
                            results["chat_output1"] = f"错误: {str(e)}"
                    
                    if chat_model2:
                        current_model += 1
                        progress(0.4 + 0.6 * (current_model / total_models), desc=f"获取模型{current_model}回答...")
                        try:
                            results["chat_output2"] = chatbot.generate_response(qa_prompt, chat_model2)
                        except Exception as e:
                            logger.error(f"模型2回答出错: {str(e)}")
                            results["chat_output2"] = f"错误: {str(e)}"
                    
                    if chat_model3:
                        current_model += 1
                        progress(0.4 + 0.6 * (current_model / total_models), desc=f"获取模型{current_model}回答...")
                        try:
                            results["chat_output3"] = chatbot.generate_response(qa_prompt, chat_model3)
                        except Exception as e:
                            logger.error(f"模型3回答出错: {str(e)}")
                            results["chat_output3"] = f"错误: {str(e)}"
                    
                    progress(1.0, desc="处理完成")
                    results["status_text"] = "处理完成"
                    return (
                        results["ocr_output"],
                        results["chat_output1"],
                        results["chat_output2"],
                        results["chat_output3"],
                        results["status_text"],
                        results["ocr_result_store"]
                    )
                    
                except Exception as e:
                    logger.error(f"处理过程出错: {str(e)}")
                    error_msg = f"错误: {str(e)}"
                    return (
                        "",
                        "",
                        "",
                        "",
                        error_msg,
                        None
                    )

            # 重试问答函数
            def retry_qa(question, chat_model1, chat_model2, chat_model3, ocr_result):
                """重新进行问答"""
                if not question.strip():
                    return ["请输入问题"] * 3

                if not any([chat_model1, chat_model2, chat_model3]):
                    return ["请至少选择一个问答模型"] * 3

                try:
                    chatbot = ChatBot()
                    # 构建提示词：如果有OCR结果就作为补充信息
                    qa_prompt = question
                    if ocr_result and ocr_result.strip():
                        qa_prompt = f"{question}\n\n补充信息：\n{ocr_result}"
                    
                    results = [""] * 3
                    if chat_model1:
                        try:
                            results[0] = chatbot.generate_response(qa_prompt, chat_model1)
                        except Exception as e:
                            logger.error(f"模型1重试出错: {str(e)}")
                            results[0] = f"错误: {str(e)}"
                    
                    if chat_model2:
                        try:
                            results[1] = chatbot.generate_response(qa_prompt, chat_model2)
                        except Exception as e:
                            logger.error(f"模型2重试出错: {str(e)}")
                            results[1] = f"错误: {str(e)}"
                    
                    if chat_model3:
                        try:
                            results[2] = chatbot.generate_response(qa_prompt, chat_model3)
                        except Exception as e:
                            logger.error(f"模型3重试出错: {str(e)}")
                            results[2] = f"错误: {str(e)}"
                    
                    return results
                except Exception as e:
                    logger.error(f"重试过程出错: {str(e)}")
                    return [f"错误: {str(e)}"] * 3

            # 处理按钮点击事件
            process_button.click(
                fn=process_all,
                inputs=[
                    image_input,
                    ocr_model_selector,
                    question_input,
                    chat_model_selector1,
                    chat_model_selector2,
                    chat_model_selector3
                ],
                outputs=[
                    ocr_output,
                    chat_output1,
                    chat_output2,
                    chat_output3,
                    status_text,
                    ocr_result_store
                ],
                api_name="process"
            )

            # 重试按钮点击事件
            retry_button.click(
                fn=retry_qa,
                inputs=[
                    question_input,
                    chat_model_selector1,
                    chat_model_selector2,
                    chat_model_selector3,
                    ocr_result_store
                ],
                outputs=[
                    chat_output1,
                    chat_output2,
                    chat_output3
                ],
                api_name="retry"
            )

            # 图片和模型选择变更事件
            def clear_outputs():
                return [""] * 4  # 清空4个输出框
                
            image_input.change(
                fn=clear_outputs,
                inputs=None,
                outputs=[ocr_output, chat_output1, chat_output2, chat_output3]
            )
            
            ocr_model_selector.change(
                fn=clear_outputs,
                inputs=None,
                outputs=[ocr_output, chat_output1, chat_output2, chat_output3]
            )

        interface.launch(
            server_name="0.0.0.0",
            server_port=7588,
            share=False,
            inbrowser=True
        )
        
    except Exception as e:
        logger.error(f"Failed to create interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 设置日志输出到文件
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbot.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        logger.info("Starting chatbot interface...")
        print("正在启动聊天机器人界面...")
        if not ChatBot().check_ollama_service():
            error_msg = "Error: Ollama service is not running. Please start Ollama first."
            logger.error(error_msg)
            print(error_msg)
            input("按回车键退出...")
            sys.exit(1)
            
        create_interface()  # 直接调用create_interface，它已经包含了launch逻辑
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)
