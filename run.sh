#!/bin/bash

# Simple PDF QA Chatbot Run Script

echo "🚀 Starting Simple PDF QA Chatbot..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is required but not installed."
    echo "Please install Ollama from: https://ollama.ai/"
    exit 1
fi

# Check if llama3:8b model is available
if ! ollama list | grep -q "llama3:8b"; then
    echo "🤖 Pulling llama3:8b model..."
    ollama pull llama3:8b
fi

# Check if we're in the parent directory and can use the existing venv
if [ -f "../.venv/bin/python" ]; then
    echo "📦 Using existing virtual environment..."
    PYTHON_CMD="../.venv/bin/python"
    PIP_CMD="../.venv/bin/pip"
    STREAMLIT_CMD="../.venv/bin/streamlit"
elif [ -f ".venv/bin/python" ]; then
    echo "📦 Using local virtual environment..."
    PYTHON_CMD=".venv/bin/python"
    PIP_CMD=".venv/bin/pip"
    STREAMLIT_CMD=".venv/bin/streamlit"
else
    echo "📦 Creating new virtual environment..."
    python3 -m venv .venv
    PYTHON_CMD=".venv/bin/python"
    PIP_CMD=".venv/bin/pip"
    STREAMLIT_CMD=".venv/bin/streamlit"
fi

# Install requirements
echo "📥 Installing requirements..."
$PIP_CMD install -r requirements.txt

# Run the application
echo "🎯 Launching demo application..."
$STREAMLIT_CMD run src/demo.py
