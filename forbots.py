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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    API_BASE_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
    API_GENERATE_URL = f"{API_BASE_URL}/api/generate"
    API_CHAT_URL = f"{API_BASE_URL}/api/chat"  # Added chat endpoint
    GRADIO_SERVER_PORT = int(os.getenv('GRADIO_SERVER_PORT', '7888'))
    OCR_TIMEOUT = int(os.getenv('OCR_TIMEOUT', '60'))  # 图片处理超时时间
    CHAT_TIMEOUT = int(os.getenv('CHAT_TIMEOUT', '30'))  # 问答超时时间
    MAX_HISTORY = int(os.getenv('MAX_HISTORY', '100'))
    RETRY_ATTEMPTS = int(os.getenv('RETRY_ATTEMPTS', '3'))

def check_ollama_service():
    try:
        requests.get(Config.API_BASE_URL, timeout=5)
        return True
    except:
        return False

@dataclass
class Conversation:
    history: List[str]
    max_length: int = Config.MAX_HISTORY

    def add_message(self, message: str) -> None:
        self.history.append(message)
        if len(self.history) > self.max_length:
            self.history = self.history[-self.max_length:]

    def get_context(self) -> str:
        return "\n".join(self.history)

    def clear(self) -> None:
        self.history = []

class ChatBot:
    """
    智能文档分析系统的核心类，提供OCR和多轮对话功能。
    
    主要功能：
    1. 灵活的图像识别：支持自定义OCR识别提示，可以针对不同类型的文档优化识别效果
    2. 多轮对话：支持在多个独立的对话上下文中进行交互
    3. 错误处理：提供详细的错误信息和故障排除建议
    
    属性：
        conversation: 主对话历史记录
        qa_conversations: 多个独立的问答对话历史记录
        session: HTTP会话对象
        current_image: 当前处理的图片
        current_ocr_result: 当前的OCR识别结果
    """
    
    def __init__(self):
        """初始化ChatBot实例，设置对话历史和会话对象"""
        self.conversation = Conversation([])
        self.session = requests.Session()
        self.current_image: Optional[str] = None
        self.current_ocr_result: Optional[str] = None
        # Add separate conversation histories for each QA section
        self.qa_conversations = {
            "qa1": Conversation([]),
            "qa2": Conversation([]),
            "qa3": Conversation([])
        }

    def _encode_image(self, image_path: str) -> str:
        # Check file size before processing
        file_size = os.path.getsize(image_path)
        max_size = 10 * 1024 * 1024  # 10MB limit
        if file_size > max_size:
            raise ValueError(f"Image file too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is 10MB.")
        
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        if not os.path.splitext(image_path)[1].lower() in valid_extensions:
            raise ValueError("Invalid image format. Supported formats: JPG, PNG, GIF, WEBP")
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_response(self, prompt: str, model_name: str, image: Optional[str] = None, is_ocr: bool = False, conversation_id: Optional[str] = None) -> str:
        """
        生成响应，支持OCR和问答两种模式，并支持多轮对话
        :param prompt: 提示词
        :param model_name: 模型名称
        :param image: 可选的图片数据
        :param is_ocr: 是否是OCR模式
        :param conversation_id: 对话ID，用于多轮对话
        """
        for attempt in range(Config.RETRY_ATTEMPTS):
            try:
                timeout = Config.OCR_TIMEOUT if is_ocr else Config.CHAT_TIMEOUT
                
                # 获取对话历史
                conversation = self.qa_conversations.get(conversation_id, self.conversation) if conversation_id else self.conversation
                context = conversation.get_context()
                
                # 构建完整的提示词，包含历史对话
                full_prompt = f"{context}\n\n{prompt}" if context else prompt
                
                # 添加OCR特定的错误处理
                if is_ocr and image:
                    try:
                        # 验证图片数据
                        if not image or len(image) < 100:  # 基本验证
                            return "错误: 图片数据无效或损坏"
                    except Exception as e:
                        return f"错误: 图片处理失败 - {str(e)}"
                
                headers = {"Content-Type": "application/json"}
                payload = {
                    "model": model_name.replace("🖼️ ", ""),
                    "prompt": full_prompt,
                    "stream": True
                }
                
                if image:
                    payload["images"] = [image]
                
                try:
                    response = self.session.post(
                        Config.API_GENERATE_URL,
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=timeout
                    )
                except requests.exceptions.Timeout:
                    return "错误: 服务器响应超时，请检查网络连接或稍后重试"
                except requests.exceptions.ConnectionError:
                    return "错误: 无法连接到服务器，请检查网络连接"
                
                if response.status_code != 200:
                    error_msg = "服务器错误"
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            error_msg = error_data['error']
                    except:
                        if response.text:
                            error_msg = response.text
                    # 提供更友好的错误消息
                    if "model not found" in error_msg.lower():
                        return f"错误: 模型 {model_name} 未找到，请确保已安装该模型"
                    elif "out of memory" in error_msg.lower():
                        return "错误: 服务器内存不足，请稍后重试或尝试处理较小的图片"
                    else:
                        return f"错误: {error_msg}"
                
                full_response = []
                try:
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    full_response.append(data["response"])
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    return f"错误: 处理服务器响应时出错 - {str(e)}"
                
                if full_response:
                    final_response = "".join(full_response)
                    # 将对话记录添加到相应的历史记录中
                    if conversation_id:
                        self.qa_conversations[conversation_id].add_message(f"User: {prompt}")
                        self.qa_conversations[conversation_id].add_message(f"Assistant: {final_response}")
                    else:
                        self.conversation.add_message(final_response)
                    return final_response
                else:
                    return "错误: 服务器返回了空响应"
                    
            except Exception as e:
                logger.error(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt == Config.RETRY_ATTEMPTS - 1:
                    return f"错误: 多次尝试后仍然失败 - {str(e)}"
                continue
            
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return f"Error: An unexpected error occurred: {str(e)}"

    def get_model_list(self) -> Tuple[List[str], str]:
        """获取可用的模型列表
        
        Returns:
            Tuple[List[str], str]: 返回一个元组，包含模型列表和状态信息
        """
        try:
            response = self.session.get(f"{Config.API_BASE_URL}/api/tags", timeout=Config.CHAT_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            if 'models' in data:
                # Extract model names 
                models = [model.get('name', '') for model in data['models']]
                return models, "成功获取模型列表"
            else:
                return [], "API返回数据格式不正确"
                
        except requests.exceptions.RequestException as e:
            logger.error(f"获取模型列表失败: {str(e)}")
            return [], f"获取模型列表失败: {str(e)}"
        except Exception as e:
            logger.error(f"获取模型列表时发生未知错误: {str(e)}")
            return [], f"获取模型列表时发生未知错误: {str(e)}"

    def clear_conversation(self, conversation_id: Optional[str] = None) -> str:
        """清空指定对话历史或所有对话历史"""
        if conversation_id:
            if conversation_id in self.qa_conversations:
                self.qa_conversations[conversation_id].clear()
                return f"Conversation {conversation_id} cleared."
        else:
            self.conversation.clear()
            for conv in self.qa_conversations.values():
                conv.clear()
            return "All conversations cleared."

def create_interface() -> None:
    try:
        # 首先检查Ollama服务是否可用
        if not check_ollama_service():
            logger.error("无法连接到Ollama服务")
            with gr.Blocks() as interface:
                gr.Markdown("# 智能文档分析系统 - 服务未连接")
                gr.Markdown("""
                ### ⚠️ 无法连接到Ollama服务
                
                请检查以下几点：
                1. 确保Ollama服务已经启动
                2. 检查环境变量OLLAMA_API_URL是否正确设置
                3. 默认地址是http://localhost:11434，确保此端口可访问
                
                系统会在服务可用时自动重新连接。
                """)
                retry_button = gr.Button("重试连接")
                status_text = gr.Textbox(
                    label="状态",
                    value="等待服务连接...",
                    interactive=False
                )
                
                retry_button.click(
                    lambda: "服务已连接，请刷新页面" if check_ollama_service() else "服务仍然无法连接，请检查服务状态",
                    None,
                    status_text
                )
            return
        
        chatbot = ChatBot()
        
        # 预先获取模型列表
        models, status = chatbot.get_model_list()
        if not models:
            logger.warning(f"无法获取模型列表: {status}")
            with gr.Blocks() as interface:
                gr.Markdown("# 智能文档分析系统 - 无可用模型")
                gr.Markdown(f"""
                ### ⚠️ 未检测到可用模型
                
                请确保安装了所需的模型：
                1. 运行 `ollama pull llava` 安装llava模型
                2. 或运行 `ollama pull bakllava` 安装bakllava模型
                3. 或安装其他支持图像分析的模型
                
                错误信息：{status}
                """)
                retry_button = gr.Button("重新检查模型")
                status_text = gr.Textbox(
                    label="状态",
                    value="等待安装模型...",
                    interactive=False
                )
                
                retry_button.click(
                    lambda: "检测到模型，请刷新页面" if chatbot.get_model_list()[0] else f"仍未检测到模型: {status}",
                    None,
                    status_text
                )
            return
            
        # 
        all_models = models
        
        with gr.Blocks(css="""
            :root {
                --primary-color: #2563eb;
                --primary-light: #3b82f6;
                --success-color: #059669;
                --warning-color: #d97706;
                --error-color: #dc2626;
                --background-color: #f3f4f6;
                --card-background: #ffffff;
                --text-primary: #1f2937;
                --text-secondary: #4b5563;
                --border-color: #e5e7eb;
                --chat-user-bg: #e5e7eb;
                --chat-assistant-bg: #dbeafe;
            }
            
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 0.5rem;
                background-color: var(--background-color);
                min-height: auto;
                height: auto;
                display: flex;
                flex-direction: column;
            }
            
            .header {
                text-align: center;
                margin-bottom: 0.75rem;
                color: var(--text-primary);
                flex: none;
            }
            
            .section {
                background-color: var(--card-background);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                height: auto;
                min-height: 0;
                display: flex;
                flex-direction: column;
            }
            
            .gr-image {
                border-radius: 6px !important;
                overflow: hidden !important;
                transition: all 0.2s !important;
                border: 2px solid transparent !important;
                height: auto !important;
                max-height: 400px !important;
                object-fit: contain !important;
            }
            
            .gr-form {
                gap: 0.75rem !important;
                height: auto !important;
                min-height: 0 !important;
            }
            
            .image-container {
                display: flex;
                height: auto;
                flex-direction: column;
                justify-content: flex-start;
                align-items: center;
                max-height: none;
                min-height: 0;
            }
            
            .step-label {
                margin: 0.5rem 0;
                color: var(--text-secondary);
                font-size: 0.9rem;
            }
            
            .result-label {
                margin: 0.5rem 0;
                color: var(--text-secondary);
                font-size: 0.9rem;
            }
            
            .gr-form {
                gap: 0.5rem !important;
            }
            
            .gr-input, .gr-dropdown {
                margin-bottom: 0.5rem !important;
            }
            
            .chat-history-container {
                max-height: 300px;
                overflow-y: auto;
                border: 1px solid var(--border-color);
                border-radius: 8px;
                margin-bottom: 1rem;
                padding: 0.5rem;
                background-color: var(--card-background);
            }
            
            .user-message {
                background-color: var(--chat-user-bg);
                padding: 0.5rem;
                margin: 0.25rem 0;
                border-radius: 8px;
                max-width: 85%;
                margin-left: auto;
                word-wrap: break-word;
            }
            
            .assistant-message {
                background-color: var(--chat-assistant-bg);
                padding: 0.5rem;
                margin: 0.25rem 0;
                border-radius: 8px;
                max-width: 85%;
                margin-right: auto;
                word-wrap: break-word;
            }
        """) as interface:
            with gr.Column(elem_classes="container"):
                with gr.Column(elem_classes="header"):
                    gr.Markdown("# 智能文档分析系统")
                
                # 图像分析区域
                with gr.Column():
                    gr.Markdown("### 图像分析")
                    image_input = gr.Image(
                        type="filepath",
                        label="上传图片"
                    )
                    
                    ocr_question_input = gr.Textbox(
                        label="图像分析问题",
                        placeholder="请输入关于图片的具体问题，例如：'请分析图片中的内容' 或 '请识别图片中的文字'",
                        lines=2
                    )
                    
                    with gr.Row():
                        ocr_model_selector = gr.Dropdown(
                            choices=all_models,
                            label="选择图像分析模型",
                            value=all_models[0] if all_models else None
                        )
                        ocr_retry_button = gr.Button(
                            "重新分析",
                            variant="secondary",
                            elem_classes="secondary"
                        )
                    ocr_output = gr.Textbox(
                        label="分析结果",
                        lines=4,
                        interactive=False,
                        elem_classes="output-text"
                    )
                
                # 问答区域1
                with gr.Column():
                    gr.Markdown("### 问答1")
                    with gr.Column(elem_classes="chat-history"):
                        chat_history1 = gr.HTML(label="对话历史")
                    question_input1 = gr.Textbox(
                        label="输入问题",
                        lines=2,
                        interactive=True
                    )
                    with gr.Row():
                        chat_model_selector1 = gr.Dropdown(
                            choices=all_models,
                            label="选择问答模型",
                            value=all_models[0] if all_models else None
                        )
                        qa_retry_button1 = gr.Button(
                            "重新回答",
                            variant="secondary",
                            elem_classes="secondary"
                        )
                    chat_output1 = gr.Textbox(
                        label="回答",
                        lines=4,
                        interactive=False,
                        elem_classes="output-text"
                    )

                # 问答区域2
                with gr.Column():
                    gr.Markdown("### 问答2")
                    with gr.Column(elem_classes="chat-history"):
                        chat_history2 = gr.HTML(label="对话历史")
                    question_input2 = gr.Textbox(
                        label="输入问题",
                        lines=2,
                        interactive=True
                    )
                    with gr.Row():
                        chat_model_selector2 = gr.Dropdown(
                            choices=all_models,
                            label="选择问答模型",
                            value=all_models[0] if all_models else None
                        )
                        qa_retry_button2 = gr.Button(
                            "重新回答",
                            variant="secondary",
                            elem_classes="secondary"
                        )
                    chat_output2 = gr.Textbox(
                        label="回答",
                        lines=4,
                        interactive=False,
                        elem_classes="output-text"
                    )

                # 问答区域3
                with gr.Column():
                    gr.Markdown("### 问答3")
                    with gr.Column(elem_classes="chat-history"):
                        chat_history3 = gr.HTML(label="对话历史")
                    question_input3 = gr.Textbox(
                        label="输入问题",
                        lines=2,
                        interactive=True
                    )
                    with gr.Row():
                        chat_model_selector3 = gr.Dropdown(
                            choices=all_models,
                            label="选择问答模型",
                            value=all_models[0] if all_models else None
                        )
                        qa_retry_button3 = gr.Button(
                            "重新回答",
                            variant="secondary",
                            elem_classes="secondary"
                        )
                    chat_output3 = gr.Textbox(
                        label="回答",
                        lines=4,
                        interactive=False,
                        elem_classes="output-text"
                    )
                
                # 主控制区
                with gr.Row():
                    process_button = gr.Button(
                        "开始处理",
                        variant="primary",
                        elem_classes="primary"
                    )
                    clear_button = gr.Button(
                        "清空",
                        variant="secondary",
                        elem_classes="secondary"
                    )
                
                # 状态显示
                with gr.Row():
                    status_text = gr.Textbox(
                        label="状态",
                        interactive=False,
                        value="请上传图片并输入问题",
                        elem_classes="status"
                    )
                
                # 隐藏的OCR结果存储
                ocr_result_store = gr.State(value=None)

            # 处理函数：一键处理所有流程
            def process_all(image_path, ocr_question, ocr_model, question1, chat_model1, 
                          question2, chat_model2, question3, chat_model3):
                """一键处理所有流程"""
                if not image_path or not ocr_model:
                    return {
                        ocr_output: "请上传图片并选择图像分析模型",
                        chat_output1: "",
                        chat_output2: "",
                        chat_output3: "",
                        status_text: "等待输入",
                        ocr_result_store: None,
                        chat_history1: "",
                        chat_history2: "",
                        chat_history3: ""
                    }
                    
                try:
                    chatbot = ChatBot()
                    
                    # 第一步：图片分析
                    base64_image = chatbot._encode_image(image_path)
                    analysis_prompt = ocr_question.strip() if ocr_question.strip() else "请分析并描述这张图片的内容。"
                    analysis_result = chatbot.generate_response(analysis_prompt, ocr_model, base64_image, is_ocr=True)
                    
                    # 第二步：问答处理
                    # 检查是否所有问答框都有内容
                    has_question1 = bool(question1.strip())
                    has_question2 = bool(question2.strip())
                    has_question3 = bool(question3.strip())
                    
                    # 如果第一个问答框为空，直接返回
                    if not has_question1:
                        return {
                            ocr_output: analysis_result,
                            chat_output1: "请在问答框1中输入问题",
                            chat_output2: "",
                            chat_output3: "",
                            status_text: "等待问题输入",
                            ocr_result_store: analysis_result,
                            chat_history1: "",
                            chat_history2: "",
                            chat_history3: ""
                        }
                    
                    # 处理问答1
                    chat_output1_text = ""
                    chat_history1_html = ""
                    if chat_model1:
                        qa_prompt1 = f"基于以下分析结果回答问题。\n分析结果：\n{analysis_result}\n\n问题：{question1}"
                        chat_output1_text = chatbot.generate_response(qa_prompt1, chat_model1, conversation_id="qa1")
                        chat_history1_html = format_chat_history(chatbot.qa_conversations["qa1"].history)
                    
                    # 处理问答2
                    chat_output2_text = ""
                    chat_history2_html = ""
                    if chat_model2:
                        # 如果问答框2有内容，使用问答框2的问题，否则使用问答框1的问题
                        qa_question2 = question2.strip() if has_question2 else question1.strip()
                        qa_prompt2 = f"基于以下分析结果回答问题。\n分析结果：\n{analysis_result}\n\n问题：{qa_question2}"
                        chat_output2_text = chatbot.generate_response(qa_prompt2, chat_model2, conversation_id="qa2")
                        chat_history2_html = format_chat_history(chatbot.qa_conversations["qa2"].history)
                    
                    # 处理问答3
                    chat_output3_text = ""
                    chat_history3_html = ""
                    if chat_model3:
                        # 如果问答框3有内容，使用问答框3的问题，否则使用问答框1的问题
                        qa_question3 = question3.strip() if has_question3 else question1.strip()
                        qa_prompt3 = f"基于以下分析结果回答问题。\n分析结果：\n{analysis_result}\n\n问题：{qa_question3}"
                        chat_output3_text = chatbot.generate_response(qa_prompt3, chat_model3, conversation_id="qa3")
                        chat_history3_html = format_chat_history(chatbot.qa_conversations["qa3"].history)
                    
                    return {
                        ocr_output: analysis_result,
                        chat_output1: chat_output1_text,
                        chat_output2: chat_output2_text,
                        chat_output3: chat_output3_text,
                        status_text: "处理完成",
                        ocr_result_store: analysis_result,
                        chat_history1: chat_history1_html,
                        chat_history2: chat_history2_html,
                        chat_history3: chat_history3_html
                    }
                    
                except Exception as e:
                    logger.error(f"处理过程中出错: {str(e)}")
                    return {
                        ocr_output: f"错误: {str(e)}",
                        chat_output1: "",
                        chat_output2: "",
                        chat_output3: "",
                        status_text: "处理出错",
                        ocr_result_store: None,
                        chat_history1: "",
                        chat_history2: "",
                        chat_history3: ""
                    }

            # 格式化对话历史为HTML
            def format_chat_history(history: List[str]) -> str:
                if not history:
                    return ""
                html = "<div class='chat-history-container'>"
                for msg in history:
                    if msg.startswith("User: "):
                        html += f"<div class='user-message'>{msg[6:]}</div>"
                    elif msg.startswith("Assistant: "):
                        html += f"<div class='assistant-message'>{msg[11:]}</div>"
                html += "</div>"
                return html

            # 绑定事件处理函数
            process_button.click(
                process_all,
                [image_input, ocr_question_input, ocr_model_selector,
                 question_input1, chat_model_selector1,
                 question_input2, chat_model_selector2,
                 question_input3, chat_model_selector3],
                [ocr_output, chat_output1, chat_output2, chat_output3,
                 status_text, ocr_result_store,
                 chat_history1, chat_history2, chat_history3]
            )

            # 清空按钮事件
            def clear_all():
                return [""] * 9  # 返回9个空字符串

            clear_button.click(
                clear_all,
                None,
                [ocr_output, chat_output1, chat_output2, chat_output3,
                 status_text, ocr_result_store,
                 chat_history1, chat_history2, chat_history3]
            )

            # 重试按钮事件
            ocr_retry_button.click(
                process_all,
                [image_input, ocr_question_input, ocr_model_selector,
                 question_input1, chat_model_selector1,
                 question_input2, chat_model_selector2,
                 question_input3, chat_model_selector3],
                [ocr_output, chat_output1, chat_output2, chat_output3,
                 status_text, ocr_result_store,
                 chat_history1, chat_history2, chat_history3]
            )
        
        interface.launch(
            server_port=Config.GRADIO_SERVER_PORT,
            share=True,
            inbrowser=True  # 自动打开浏览器
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
        if not check_ollama_service():
            error_msg = "Error: Ollama service is not running. Please start Ollama first."
            logger.error(error_msg)
            print(error_msg)
            input("按回车键退出...")
            sys.exit(1)
            
        demo = create_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=Config.GRADIO_SERVER_PORT,
            share=False,
            inbrowser=True
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)
