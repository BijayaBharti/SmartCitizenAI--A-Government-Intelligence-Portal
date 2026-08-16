# AI Citizen Assistance Portal

Built on top of your original `Chat-With-PDF` RAG app. Same base structure (Streamlit sidebar API config,
PDF upload, FAISS + LangChain RAG chain), extended into a full citizen-services portal.

## Files

- `app.py` — the complete application (single file, no extra folders)
- `schemes.json` — small bundled dataset of 12 Indian government schemes used by the recommendation engine
- `requirements.txt` — all Python dependencies
- `.env.example` — template for your API keys (optional — you can also paste keys into the sidebar)

## Features

1. **AI Assistant** — general chat about government services, with conversation memory (session-based)
2. **OCR Document Reader** — Aadhaar, PAN, Passport, Driving License, Income Certificate images (EasyOCR)
   or PDFs (PyPDFLoader); auto-classifies the document type and pulls out key fields (Aadhaar/PAN numbers, dates)
3. **Government Scheme Recommendation** — enter age, income and category, get matching schemes from
   `schemes.json` plus an AI-generated priority explanation
4. **RAG Chat with Documents** — upload any government PDF, it's chunked, embedded with
   `GoogleGenerativeAIEmbeddings`, indexed in FAISS, and you can chat with it (streaming answers)
5. **Live Search** — Tavily-powered live web search with an AI summary of the results
6. **Complaint Generator** — fills a form and drafts a formal complaint letter, downloadable as PDF/TXT
7. **Document Checklist Generator** — pick a service (passport, PAN, ration card, etc.) and get the
   exact document checklist, downloadable as PDF/TXT
8. **Translator** — English ⇄ Hindi ⇄ Hinglish
9. **History** — unified session log of every action across all tabs, downloadable

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your keys, **or** just paste them into the sidebar at runtime:

```bash
cp .env.example .env
```

- `GOOGLE_API_KEY` (required) — free from https://aistudio.google.com/app/apikey — powers the LLM (Gemini
  2.5 Flash), embeddings, and every AI feature.
- `TAVILY_API_KEY` (optional) — free from https://tavily.com — only needed for the Live Search tab.

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Notes

- Uploaded PDFs are saved locally to a `pdf_files/` folder (created automatically) — same pattern as your
  original app.
- If `easyocr` fails to install on your machine, every other tab still works; only image OCR is disabled
  (PDF text extraction in the OCR tab keeps working via PyPDFLoader).
- PDF export uses `fpdf2` with a core Latin font, so Hindi/Devanagari text in a PDF export will show as
  best-effort transliteration; use the TXT download (always available) for exact Hindi/Hinglish text.
- All chat history, RAG index, and activity log are session-scoped (reset on page refresh), matching the
  original app's caching style — no external database required, keeping this simple for a final-year project.
