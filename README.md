# 智能文档分析系统

这是一个基于Ollama的智能文档分析系统，提供OCR和多轮对话功能。

## 主要功能

1. 图像识别：支持多种格式的图片文件识别（JPG, PNG, GIF, WEBP）
2. 多轮对话：支持在多个独立的对话上下文中进行交互
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
