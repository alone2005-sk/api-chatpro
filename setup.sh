#!/bin/bash

echo "🚀 Setting up Local AI Backend..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p workspace uploads outputs logs

# Set up environment variables
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOL
# API Keys (optional - for fallback to remote APIs)
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Local LLM Settings
OLLAMA_HOST=localhost:11434
LLAMACPP_HOST=localhost:8080
GPT4ALL_HOST=localhost:4891
EOL
fi

echo "✅ Setup complete!"
echo ""
echo "To start the server:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start Ollama (optional): ollama serve"
echo "3. Run the server: python main.py"
echo ""
echo "API will be available at: http://localhost:8000"
echo "Documentation at: http://localhost:8000/docs"
