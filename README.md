# RAG-Based Chat Assistant

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
- 🧪 **Multiple model experiments** (GPT‑4o mini, Gemini, Claude, LLaMA, BART)  
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
```
.
├─ app.py
├─ notebooks/
│  ├─ Bart+Anthropic.ipynb
│  ├─ Gemini+OpenAI+Pinecone.ipynb
│  └─ Openai+Chroma+LLama3.2.ipynb
├─ report/
│  └─ ReLu Rebels Report.pdf
├─ chromadb/
├─ requirements.txt
└─ README.md
```

---

## Tech Stack
- **Python**, **Streamlit**
- **LangChain** (ChatOpenAI, OpenAIEmbeddings, RetrievalQA, Chroma)
- **ChromaDB / Pinecone**
- **OpenAI, Gemini, Claude, LLaMA models**
- **NLTK**, **Evaluate** for BLEU/ROUGE

---

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Set Environment Variable
```bash
export OPENAI_API_KEY="your_key_here"
```

### Run the App
```bash
streamlit run app.py
```

---

## Evaluation & Results
BLEU and ROUGE metrics are used to benchmark model performance.  
Experiment notebooks compare different LLMs + vector DBs.

---

## Limitations & Future Work
- Larger models can be added for higher accuracy  
- UI can be extended for model selection  
- More evaluation metrics can be integrated  

---

## References
- ScienceQA dataset  
- BLIP image captioning  
- RAG research papers  
