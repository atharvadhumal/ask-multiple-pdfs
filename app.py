import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import google.generativeai as genai

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # Check if text_chunks is empty
    if not text_chunks:
        st.error("No text was extracted from the documents. Please check if the PDFs contain readable text.")
        return None
    
    # Add a sample chunk if debugging is needed
    if len(text_chunks) == 0:
        text_chunks = ["This is a sample text to prevent index error during development"]
        
    api_key = os.getenv("GEMINI_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_key
    )
    
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def get_conversation_chain(vectorstore):
    if vectorstore is None:
        return None
        
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.7,
        google_api_key=api_key
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please process some documents first before asking questions.")
        return
        
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    
    # Configure Google Generative AI with API key
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.error("Gemini API Key not found. Please add GEMINI_API_KEY to your .env file.")
        return
    
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF document.")
                return
                
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                if not raw_text.strip():
                    st.error("Could not extract text from the uploaded PDFs. Please make sure they contain readable text.")
                    return

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Show number of chunks extracted (helpful for debugging)
                st.info(f"Extracted {len(text_chunks)} text chunks from your documents.")

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                if vectorstore is None:
                    st.error("Failed to create vector store. Please try with different documents.")
                    return

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                if st.session_state.conversation is None:
                    st.error("Failed to initialize the conversation chain.")
                    return
                    
                st.success("Processing complete! You can now ask questions about your documents.")


if __name__ == '__main__':
    main()