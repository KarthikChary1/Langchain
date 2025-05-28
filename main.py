import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.hub import pull
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def document_vector():
    if os.path.exists("faiss_index/index.faiss"):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        file_path = "C:/Users/Karth/Langchain/Langchain/Research paper(transformer).pdf"
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)


        print("Creating new FAISS vector store...")
        vector_store = FAISS.from_documents(split_docs, embeddings)
        vector_store.save_local("faiss_index")
    return vector_store

def ask_llm(query,chat_history):
    vector_store = document_vector()
    retriever = vector_store.as_retriever()

    prompt = pull("langchain-ai/retrieval-qa-chat")
    llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0,verbose=True)

    stuff_doc_chain = create_stuff_documents_chain(llm,prompt)
    rephrase_prompt = pull("langchain-ai/chat-langchain-rephrase")
    history_retriever = create_history_aware_retriever(llm,retriever,rephrase_prompt)

    chain = create_retrieval_chain(history_retriever,stuff_doc_chain)

    return chain.invoke({"input":query,"chat_history":chat_history})

if __name__ == "__main__":
    text_input = st.chat_input(placeholder="Ask any query about transformers")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if text_input:
        with st.spinner("Generating response.."):
            generated_response = ask_llm(query=text_input, chat_history=st.session_state["chat_history"])
            response =  f"{generated_response['answer']}"
            st.session_state["chat_history"].append({"role": "user", "content": text_input})
            st.session_state["chat_history"].append({"role": "assistant", "content": response})

    if st.session_state["chat_history"]:
        for chat in st.session_state["chat_history"]:
            if chat["role"] == "user":
                st.chat_message("user").write(chat["content"])
            elif chat["role"] == "assistant":
                st.chat_message("assistant").write(chat["content"])
