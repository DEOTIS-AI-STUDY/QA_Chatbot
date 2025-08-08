# Simple PDF QA Chatbot

A simple PDF-based Question-Answering chatbot using local LLM (Ollama).

## Quick Start

### Prerequisites

1. **Install Ollama**: [Download from official site](https://ollama.ai/)
2. **Pull LLM model**:
   ```bash
   ollama pull llama3:8b
   ```

### Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
simple_pdf_chatbot/
├── app.py               # Main Streamlit application
├── requirements.txt     # Dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── config.yaml         # Configuration (optional)
```

## Usage

1. **Upload PDF**: Use the sidebar to upload a PDF document
2. **Process**: Click "PDF 처리" to extract and vectorize content
3. **Ask Questions**: Use the chat interface to ask questions about the document
4. **View Sources**: Expand "참고 문서" to see source passages

## Features

- 🔒 **Local Processing**: Uses Ollama for completely local LLM inference
- 📄 **PDF Support**: Extract and process PDF documents for Q&A
- 🌍 **Korean Support**: Optimized for Korean language processing
- 🎨 **Simple UI**: Clean Streamlit interface

## Configuration

You can modify the model settings directly in `app.py` or create a `config.yaml` file.

## License

MIT License
