# DocuAsk

DocuAsk is a lightweight PDF question answering web application built with Flask, PyMuPDF, Anthropic Claude, and a single-file HTML/CSS/JavaScript frontend. Upload a PDF, extract its text, and ask questions using only the document content.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set your Anthropic API key:

   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

   PowerShell alternative:

   ```powershell
   $env:ANTHROPIC_API_KEY="your_key_here"
   ```

3. Start the backend:

   ```bash
   python app.py
   ```

   Windows-friendly option for this project:

   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```

4. Open `http://localhost:5000` in your browser.

   You can also open `index.html` directly, but serving it from Flask is simpler on Windows.

## Anthropic API Key

You can create an Anthropic API key at: https://console.anthropic.com
