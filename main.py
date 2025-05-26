from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.hub import pull
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
import streamlit as st
load_dotenv()

quest = st.text_input("Question: ")

if quest:
    file_path = "C:/Users/Karth/Langchain/Langchain/Research paper(transformer).pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    new_vector_store = FAISS.from_documents(split_docs, embeddings)

    retriever = new_vector_store.as_retriever()

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    prompt = pull("langchain-ai/retrieval-qa-chat")
    stuffed_doc_chain = create_stuff_documents_chain(llm, prompt)
    model = create_retrieval_chain(retriever, stuffed_doc_chain)
    result = model.invoke({"input": quest})
    st.write(result["answer"])
