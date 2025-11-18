# RAG-Based Chat Assistan

A Retrieval-Augmented Generation (RAG) chatbot that answers **science questions with explanations**, built using LangChain, modern LLMs, and a vector database (Chroma / Pinecone).

The Streamlit app in `app.py` provides an interactive **Science QA assistant** that retrieves relevant context and walks through its reasoning before giving the final answer.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Build the Chroma Vector Store](#build-the-chroma-vector-store)
  - [Run the App](#run-the-app)
- [Evaluation & Results](#evaluation--results)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Project Overview
This project builds a RAG-based question–answering system for science questions. It combines retrieval, reasoning, and LLM-generated explanations to produce accurate, interpretable answers.

---

## Features
- 🔎 **Retrieval-Augmented Generation (RAG)** using Chroma or Pinecone  
- 🧠 **Explain-then-answer prompting** for clearer reasoning  
- 🧪 **Multiple model experiments** (GPT-4o mini, Gemini, Claude, LLaMA, BART)  
- 📊 **Evaluation with BLEU + ROUGE**  
- 🖥️ **Streamlit UI for querying the model interactively**

---

## Architecture
1. **Embed science QA text** using OpenAI or HuggingFace embeddings  
2. **Store embeddings** in ChromaDB or Pinecone  
3. **Retrieve context** at query time  
4. **LLM generates reasoning + final answer**  
5. Optional **evaluation** in notebooks  

---

## Repository Structure
```text
.
├─ app.py
├─ notebooks/
│  ├─ Bart+Anthropic.ipynb
│  ├─ Gemini+OpenAI+Pinecone.ipynb
│  └─ Openai+Chroma +LLama3.2.ipynb
├─ chromadb/              # local Chroma vector store (ignored by git)
├─ generate_chromadb.py   # script to build the Chroma DB
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## Tech Stack
- **Python**, **Streamlit**
- **LangChain** (`ChatOpenAI`, `OpenAIEmbeddings`, `RetrievalQA`, `Chroma`)
- **ChromaDB / Pinecone**
- **OpenAI, Gemini, Claude, LLaMA models**
- **NLTK**, **Evaluate** for BLEU/ROUGE

---

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Environment Variables

Set your OpenAI API key (and any others you need) in the environment:

```bash
export OPENAI_API_KEY="your_key_here"
```

On Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

Make sure `app.py` and `generate_chromadb.py` read the key from the environment (not hard-coded).

---

### Build the Chroma Vector Store

Before running the app, you need to create the local Chroma database in the `chromadb/` folder.

Run:

```bash
python generate_chromadb.py
```

This script will:

1. Load your science QA data (you plug in your own data-loading logic).
2. Chunk the text into passages.
3. Create embeddings with `OpenAIEmbeddings`.
4. Store everything in a local Chroma database under `chromadb/`.

> Note: The `chromadb/` folder is **ignored by git** and is generated locally on each machine.

---

### Run the App

Once the Chroma database has been created:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`) and start asking science questions.

---

## Evaluation & Results
BLEU and ROUGE metrics are used to benchmark model performance.  
Experiment notebooks compare different LLM + vector DB combinations.

---

## Limitations & Future Work
- Larger models can be added for higher accuracy  
- UI can be extended for model selection  
- More evaluation metrics can be integrated  

---

## References
- ScienceQA dataset  
- BLIP image captioning  
- Retrieval-Augmented Generation (RAG) research papers  
