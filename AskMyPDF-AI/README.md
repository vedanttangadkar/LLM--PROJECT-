# AskMyPDF-AI

AskMyPDF-AI is a beginner-friendly PDF Question Answering project built with Python, Streamlit, LangChain, ChromaDB, and PyPDF. It now supports a **free local mode by default**, so you can run and deploy the project even without OpenAI billing.

You can still optionally use OpenAI mode later, but the project is fully usable in local mode with:

```bash
streamlit run app.py
```

## Project Overview

This application uses a Retrieval-Augmented Generation style workflow for PDF question answering:

1. Upload one or more PDF files
2. Extract text from each PDF
3. Split the text into smaller chunks
4. Create embeddings for each chunk
5. Store the embeddings in ChromaDB
6. Retrieve the most relevant chunks for a user question
7. Generate an answer and show source citations

By default, the app uses a **local free engine**:

- Local hash-based embeddings
- ChromaDB vector search
- Local extractive answer generation

Optional OpenAI mode is still available if you want GPT-based answers and have a funded API key.

## Features

- Upload one or multiple PDF files
- Extract text from PDFs page by page
- Split text using `RecursiveCharacterTextSplitter`
- Store vector embeddings in ChromaDB
- Ask questions in natural language
- Retrieve relevant chunks with semantic search
- Show source citations below every answer
- Maintain chat history using Streamlit session state
- Loading spinners during processing and answering
- Duplicate PDF prevention using SHA256 file hashes
- Clean modular Python structure
- Dark UI with better answer readability
- Free local mode for deployment without OpenAI billing
- Optional OpenAI mode for GPT-powered answers

## Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- PyPDF
- python-dotenv
- OpenAI API (optional mode only)

## Architecture Explanation

AskMyPDF-AI supports two answer engines:

### 1. Local Free Mode

- PDFs are parsed with `pypdf`
- Text is split into chunks with `RecursiveCharacterTextSplitter`
- Chunks are embedded locally using a lightweight hash-based embedding class
- Embeddings are stored in ChromaDB
- Retrieved chunks are turned into answers using a local extractive answer builder

### 2. OpenAI Mode

- PDFs are parsed and split in the same way
- Embeddings are generated with `OpenAIEmbeddings`
- ChromaDB stores and retrieves the chunk vectors
- `RetrievalQA` with `gpt-3.5-turbo` generates final answers

## Folder Structure

```text
AskMyPDF-AI/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .env.example
|-- .gitignore
|
|-- uploads/
|
|-- database/
|
|-- utils/
|   |-- pdf_loader.py
|   |-- embeddings.py
|   |-- retriever.py
|   `-- chat_handler.py
|
`-- assets/
    `-- screenshot.png
```

## Installation Steps

### 1. Open the Project in VS Code

Open the `AskMyPDF-AI` folder in VS Code.

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Optional Environment File

If you only want the free local mode, you do **not** need an API key.

If you want OpenAI mode, copy `.env.example` to `.env` and add:

```env
OPENAI_API_KEY=your_api_key_here
```

## How to Get an OpenAI API Key

Only needed for OpenAI mode.

1. Go to the OpenAI platform dashboard
2. Create an API key
3. Add it to your `.env`
4. Make sure the API project has billing or credits

Important:
ChatGPT Plus does **not** automatically include OpenAI API billing.

## How to Run the Project

From the project root:

```bash
streamlit run app.py
```

After the app opens:

1. Leave the engine on `Local (Free, no API billing)` if you want the easiest setup
2. Upload one or more PDF files
3. Click **Process PDFs**
4. Ask questions in the chat box
5. Review the cited source chunks under each answer

## Deployment Notes

This project is now deployment-friendly because:

- It runs in free local mode without external API billing
- Sensitive `.env` files are ignored
- Generated uploads, vector databases, logs, and caches should not be committed
- The app recreates `uploads/` and `database/` automatically if they are missing

Recommended before uploading:

- Keep the code files
- Remove temporary logs and caches
- Do not upload real `.env` secrets
- Start with empty `uploads/` and `database/` folders for a clean deployment

## Example Screenshots

The project includes a placeholder screenshot file:

```text
assets/screenshot.png
```

You can replace it with a real screenshot later.

## Engineering Challenge

**Challenge:**  
The system initially retrieved irrelevant chunks from PDFs which caused inaccurate answers and hallucinations.

**Solution:**  
Implemented better chunking strategy, semantic retrieval, reduced model temperature, and optimized chunk overlap.

## Future Improvements

- Add OCR support for scanned PDFs
- Add per-document delete or reset options
- Add support for DOCX and TXT files
- Add streaming answers
- Add a stronger local summarization model
- Add authentication and cloud storage
- Add document filters and per-file Q&A

## Final Notes

This project is simple enough for a student project, but now it is also much easier to run and deploy because it no longer depends on paid API billing by default. You can use local mode immediately, and enable OpenAI mode only when you actually want it.
