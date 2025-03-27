# app.py

import streamlit as st
import time
import base64
import os
from vectors import EmbeddingsManager  # Import the EmbeddingsManager class
from chatbot import ChatbotManager    # Import the ChatbotManager class

# Initialize session state variables if not already present
if "temp_pdf_path" not in st.session_state:
    st.session_state["temp_pdf_path"] = None
if "chatbot_manager" not in st.session_state:
    st.session_state["chatbot_manager"] = None
if "embeddings_created" not in st.session_state:
    st.session_state["embeddings_created"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Set page configuration
st.set_page_config(
    page_title="Document RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for navigation
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("### Variable Document RAG")
    st.markdown("---")
    page = st.radio("Navigate", options=["🏠 Home", "🤖 Chatbot"])

def home_page():
    st.title("📄 Document Buddy App")
    st.markdown(
        """
        A personal Project built to for creating a **variable document retrieval augmented generation (RAG) system**

        Technologies Used:
        - **Streamlit**: For the web interface
        - **Langchain**: For the document processing and embedding creation
        - **Qdrant**: For the vector database
        - **Ollama**: For the local LLM model (Llama 3.2)
        
        """
    )

def process_embeddings():
    try:
        embeddings_manager = EmbeddingsManager(
            model_name="BAAI/bge-small-en",
            device="cpu",
            encode_kwargs={"normalize_embeddings": True},
            qdrant_url="http://localhost:6333",
            collection_name="vector_db",
        )
        with st.spinner("🔄 Creating embeddings..."):
            result = embeddings_manager.create_embeddings(st.session_state["temp_pdf_path"])
            time.sleep(1)  # Simulate additional processing time if needed
        st.success(result)
        st.session_state["embeddings_created"] = True

        # Initialize ChatbotManager if not already done
        if st.session_state["chatbot_manager"] is None:
            st.session_state["chatbot_manager"] = ChatbotManager(
                model_name="BAAI/bge-small-en",
                device="cpu",
                encode_kwargs={"normalize_embeddings": True},
                llm_model="llama3.2:3b",
                llm_temperature=0.7,
                qdrant_url="http://localhost:6333",
                collection_name="vector_db",
            )
    except (FileNotFoundError, ValueError, ConnectionError) as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def chatbot_page():
    st.title("🤖 Chatbot Interface (Llama 3.2 RAG 🦙)")
    st.markdown("---")
    
    # Use tabs to group functionality in the Chatbot section
    tab1, tab2 = st.tabs(["Upload & Process Document", "Chat with Document"])

    with tab1:
        st.header("📂 Upload Document and Auto-Create Embeddings")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            st.success("📄 File Uploaded Successfully!")
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**File Size:** {uploaded_file.size} bytes")

            # Save file temporarily
            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["temp_pdf_path"] = temp_pdf_path

            # Optional PDF preview
            with st.expander("Show PDF Preview"):
                # Reset the file pointer and display PDF
                uploaded_file.seek(0)
                base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

            # Automatically process embeddings if not already done for this file
            if not st.session_state["embeddings_created"]:
                process_embeddings()
        else:
            st.info("Please upload a PDF to start processing.")

    with tab2:
        st.header("💬 Chat with Document")
        if st.session_state["chatbot_manager"] is None or not st.session_state["embeddings_created"]:
            st.info("🤖 Please upload a PDF and ensure embeddings are created to start chatting.")
        else:
            # Display previous messages
            for msg in st.session_state["messages"]:
                st.chat_message(msg["role"]).markdown(msg["content"])
            # Accept user input
            user_input = st.chat_input("Type your message here...")
            if user_input:
                st.chat_message("user").markdown(user_input)
                st.session_state["messages"].append({"role": "user", "content": user_input})
                with st.spinner("🤖 Responding..."):
                    try:
                        answer = st.session_state["chatbot_manager"].get_response(user_input)
                        time.sleep(1)
                    except Exception as e:
                        answer = f"⚠️ An error occurred: {e}"
                st.chat_message("assistant").markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})

if page == "🏠 Home":
    home_page()
elif page == "🤖 Chatbot":
    chatbot_page()
