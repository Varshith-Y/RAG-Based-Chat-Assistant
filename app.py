import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import nltk
import evaluate
from nltk.translate.bleu_score import sentence_bleu
import os

nltk.download('punkt')
rouge = evaluate.load('rouge')


OPENAI_API_KEY = 'sk-' #Enter api key

os.environ["OPENAI_API_KEY"] ='sk-'#Enter api key

llm = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, show_progress_bar=True)
persist_directory = 'chromadb/'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

def ask_question(question, choices, llm, vectordb):
    template = f"""Use the following context retrieved from similar training data to answer the question. 
      Provide your answer in two parts:
      1. Explain how you arrived at the answer based on the context.
      **Information Provided:**
      Context: {{context}}
      Question: {question}
      Choices: {choices}
      Helpful Answer: [Your answer here]
      """
    input_vars = ["context", "question", "choices"]
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=input_vars, template=template)

    retrieved_docs = vectordb.as_retriever().get_relevant_documents(question)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    result = qa_chain({"query": question, "context": retrieved_docs, "question": question, "choices": choices})
    

    answer = result["result"]
    retrieved_content = [doc.page_content for doc in retrieved_docs]
    
    return answer, retrieved_content
# Streamlit UI
st.title("RAG Science QA Chatbot")
st.write("Ask a science-related question and provide optional answer choices.")


question = st.text_input("Enter your question:")
choices = st.text_input("Provide options (e.g., Option 1, Option 2):")


if st.button("Get Answer"):
    if question:
        answer, retrieved_docs = ask_question(question, choices, llm, vectordb)


        st.write("### Answer:")
        st.write(answer)


        st.write("### Retrieved Documents:")
        for i, doc in enumerate(retrieved_docs, start=1):
            st.write(f"**Document {i}:**")
            st.write(doc)
    else:
        st.write("Please enter a question.")


if st.button("Restart"):
    st.session_state["question"] = ""
    st.session_state["choices"] = ""
