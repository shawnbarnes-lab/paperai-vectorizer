#!/usr/bin/env python3
"""
PaperAI Vectorizer Server
=========================
REST API server that receives documents, vectorizes them on GPU,
and returns the .paperai vector file.

Run this on your RTX 3090 Ubuntu machine.

Requirements:
    pip install flask sentence-transformers PyPDF2 torch

Usage:
    python server.py
    python server.py --host 0.0.0.0 --port 5000

The server will listen for document uploads and return vectorized files.
"""

import argparse
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, request, jsonify
import torch
from sentence_transformers import SentenceTransformer

# Optional PDF support
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("Warning: PyPDF2 not installed. PDF support disabled.")
    print("Install with: pip install PyPDF2")

app = Flask(__name__)

# Global model (loaded once at startup)
model = None
device = None

# Must match the model used in the Android app!
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_model():
    """Load the embedding model at startup."""
    global model, device
    
    print(f"Loading embedding model: {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Model loaded! Embedding dimension: {EMBEDDING_DIM}")


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF, returning list of {page, text} dicts."""
    if not HAS_PDF:
        raise ImportError("PyPDF2 is required for PDF processing")
    
    pages = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    'page': i + 1,
                    'text': text.strip()
                })
    return pages


def extract_text_from_txt(txt_path: str) -> List[Dict[str, Any]]:
    """Extract text from TXT/MD file."""
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Split by double newlines to create logical sections
    sections = re.split(r'\n\s*\n', text)
    pages = []
    for i, section in enumerate(sections):
        if section.strip():
            pages.append({
                'page': i + 1,
                'text': section.strip()
            })
    return pages


def chunk_text(text: str, page_number: int, source_name: str,
               chunk_size: int = 400, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """Split text into chunks with overlap."""
    chunks = []
    
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) <= chunk_size:
        if text:
            chunks.append({
                'text': text,
                'pageNumber': page_number,
                'sourceName': source_name
            })
    else:
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                for punct in ['. ', '! ', '? ', '\n']:
                    last_punct = text.rfind(punct, start + chunk_size // 2, end + 50)
                    if last_punct != -1:
                        end = last_punct + 1
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'pageNumber': page_number,
                    'sourceName': source_name
                })
            
            start = end - chunk_overlap
            if start >= len(text) - chunk_overlap:
                break
    
    return chunks


def process_document(file_path: str, source_name: str) -> List[Dict[str, Any]]:
    """Process a document into chunks."""
    ext = Path(file_path).suffix.lower()
    
    print(f"Processing: {source_name} ({ext})")
    
    # Extract text based on file type
    if ext == '.pdf':
        if not HAS_PDF:
            raise ValueError("PDF support not available. Install PyPDF2.")
        pages = extract_text_from_pdf(file_path)
    elif ext in ['.txt', '.md', '.text']:
        pages = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    print(f"  Extracted {len(pages)} pages/sections")
    
    # Chunk all pages
    all_chunks = []
    for page_data in pages:
        chunks = chunk_text(
            page_data['text'],
            page_data['page'],
            source_name
        )
        all_chunks.extend(chunks)
    
    print(f"  Created {len(all_chunks)} chunks")
    return all_chunks


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate embeddings for all chunks."""
    if not chunks or model is None:
        return chunks
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    
    texts = [c['text'] for c in chunks]
    
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding.tolist()
    
    print("Embeddings generated!")
    return chunks


def build_output(chunks: List[Dict[str, Any]], source_name: str) -> Dict[str, Any]:
    """Build the .paperai output structure."""
    sources = list(set(c['sourceName'] for c in chunks))
    
    return {
        'version': 1,
        'model': MODEL_NAME,
        'embeddingDimension': EMBEDDING_DIM,
        'createdAt': datetime.now().isoformat(),
        'chunkCount': len(chunks),
        'sourceCount': len(sources),
        'sources': sources,
        'chunks': chunks
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model': MODEL_NAME,
        'device': device,
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None
    })


@app.route('/vectorize', methods=['POST'])
def vectorize():
    """
    Vectorize an uploaded document.
    
    Expects multipart/form-data with:
    - file: The document file (.txt, .pdf)
    
    Returns:
    - JSON .paperai content
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get source name from filename
    source_name = Path(file.filename).stem
    ext = Path(file.filename).suffix.lower()
    
    if ext not in ['.txt', '.pdf', '.md', '.text']:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Process document
        chunks = process_document(tmp_path, source_name)
        
        if not chunks:
            return jsonify({'error': 'No text extracted from document'}), 400
        
        # Generate embeddings
        chunks = generate_embeddings(chunks)
        
        # Build output
        output = build_output(chunks, source_name)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        print(f"Successfully vectorized: {source_name}")
        print(f"  Chunks: {len(chunks)}")
        
        return jsonify(output)
    
    except Exception as e:
        print(f"Error processing document: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Simple index page."""
    return """
    <html>
    <head><title>PaperAI Vectorizer Server</title></head>
    <body>
        <h1>PaperAI Vectorizer Server</h1>
        <p>Status: Running</p>
        <p>Device: {device}</p>
        <h2>Endpoints:</h2>
        <ul>
            <li>GET /health - Health check</li>
            <li>POST /vectorize - Upload document, receive vectors</li>
        </ul>
        <h2>Test Upload:</h2>
        <form action="/vectorize" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".txt,.pdf,.md">
            <button type="submit">Vectorize</button>
        </form>
    </body>
    </html>
    """.format(device=device)


def main():
    parser = argparse.ArgumentParser(description='PaperAI Vectorizer Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load model at startup
    load_model()
    
    print(f"\n{'='*50}")
    print(f"PaperAI Vectorizer Server")
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"{'='*50}\n")
    
    # Get local IP for easy reference
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"Local IP: {local_ip}")
        print(f"Connect from Android app using: http://{local_ip}:{args.port}")
    except:
        pass
    
    print()
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
