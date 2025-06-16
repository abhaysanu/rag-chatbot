# rag-chatbot

Here's a clean and professional `README.md` for your **LangGraph ChatBot with OCR + FAISS + FastAPI + Streamlit** project:

---

```markdown
# 🧠 LangGraph ChatBot (OCR + FAISS + FastAPI + Streamlit)

An AI-powered chatbot that reads text from images using **EasyOCR**, creates semantic embeddings using **SentenceTransformers**, stores them in a **FAISS vector store**, and provides conversational answers via **LangGraph** and **Open Source LLMs**.

It features:
- 🔍 OCR from images (PNG, JPG, JPEG)
- 🧠 Embedding + FAISS-based similarity search
- 💬 ChatBot interface via LangGraph & Transformers
- ⚡ REST API using FastAPI
- 🌐 Streamlit Frontend
- 🐳 Dockerized for deployment

---

## 📁 Project Structure

```

DB\_FETCH\_APP/
│
├── images/                  # Folder containing input images
├── chunks.json              # JSON file of text chunks from OCR
├── vector.index             # FAISS vector store
│
├── main.py                  # FastAPI app with LangGraph logic
├── frontend.py              # Streamlit frontend app
├── requirements.txt         # Python dependencies
├── Dockerfile               # Dockerfile to build the container
└── README.md                # You're here :)

````

---

## 🚀 Features

- **Image-to-Text**: Uses `EasyOCR` for extracting text from scanned image documents.
- **Semantic Chunking**: Uses LangChain's `RecursiveCharacterTextSplitter` to create manageable chunks.
- **Embedding**: `sentence-transformers` to generate dense vectors.
- **Similarity Search**: `FAISS` to retrieve relevant chunks given a question.
- **Chat Reasoning**: Open-source LLM answers based on document context.
- **API Interface**: Built using FastAPI and deployable on any server or cloud.
- **UI**: Streamlit interface for user-friendly chatbot interactions.

---

## 🔧 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/langgraph-chatbot.git
cd langgraph-chatbot
````

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add images

Place your input scanned images inside the `images/` folder.

### 5. Run FastAPI server

```bash
uvicorn main:app --reload
```

### 6. Run Streamlit UI

In a separate terminal:

```bash
streamlit run frontend.py
```

---

## 🐳 Docker Instructions

### Build Docker image

```bash
docker build -t langgraph-chatbot .
```

### Run container

```bash
docker run -p 8000:8000 langgraph-chatbot
```

Then access the Streamlit frontend separately or run it locally pointing to `http://127.0.0.1:8000`.

---

## 🧪 Sample API Usage

**POST** `/chat`

```json
{
  "query": "What is this document about?",
  "chat_history": []
}
```

**Response:**

```json
{
  "response": "This document explains..."
}
```

---

