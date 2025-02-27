# DocuRAG - Intelligent Document Analysis System

![DocuRAG Interface Screenshot](./docs/screenshot.png)

A powerful document analysis tool combining RAG (Retrieval-Augmented Generation) with multi-modal processing for PDF/text analysis.

## Features

- **Document Processing**
  - PDF/TXT file ingestion
  - Language detection
  - Domain classification (Education, Health, Technology, etc.)
  - Key term extraction & visualization
  
- **AI Analysis**
  - Context-aware summarization
  - Interactive Q&A interface
  - Semantic search capabilities
  - Notes preservation system

- **Technical Highlights**
  - Hybrid retrieval (TF-IDF + Sentence Transformers)
  - Quantized LLMs for efficient inference
  - GPU/CPU compatible architecture
  - Secure API key management

## Installation

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/docurag.git
cd docurag


python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows


pip install -r requirements.txt
python -m nltk.downloader punkt


cp config/api_keys.env.example config/api_keys.env
# Add your API keys to config/api_keys.env

# Start the application
python app.py



docurag/
├── src/
│   ├── document_processing.py
│   ├── generation.py
│   ├── hybrid_retrieval.py
│   └── rag_pipeline.py
├── config/
│   └── api_keys.env
├── outputs/
├── docs/
├── app.py
└── requirements.txt