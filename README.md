# 智能文档分析系统

这是一个基于Ollama的智能文档分析系统，提供OCR和多轮对话功能。

## 主要功能

1. 图像识别：支持多种格式的图片文件识别（JPG, PNG, GIF），基于Ollama里面提供的图像识别的LLM,建议使用面壁智能的MiniCPM-v，实测比Llava的好用。
2. 多轮对话：支持在多个独立的LLM对话，可以选择同时三个,这边边建议选DeepSeek，Qwen，以及Ph4同时进行上下文中进行交互，一次提问，三个回答PK，供你选择。
3. 用户界面：使用Gradio提供友好的Web界面

## 系统要求

- Python 3.7+
- Ollama服务运行在本地或远程服务器上

## 安装步骤

1. 安装依赖包：
   ```
   pip install -r requirements.txt
   ```

2. 配置环境变量（可选）：
   创建.env文件并设置以下变量：
   - OLLAMA_API_URL：Ollama服务地址（默认：http://localhost:11434）
   - GRADIO_SERVER_PORT：Gradio服务端口（默认：7888）
   - OCR_TIMEOUT：图片处理超时时间（默认：60秒）
   - CHAT_TIMEOUT：问答超时时间（默认：30秒）
   - MAX_HISTORY：历史记录最大长度（默认：100）
   - RETRY_ATTEMPTS：重试次数（默认：3）

## 运行方式

直接运行forbot.py文件：
```
python forbot.py
```

## 注意事项

- 图片大小限制为10MB
- 支持的图片格式：JPG, PNG, GIF, WEBP
- 确保Ollama服务正常运行
# forbots
Bots can gather for one purpose.
