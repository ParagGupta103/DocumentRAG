# chatbot.py

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st


class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3.2:3b",
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            llm_model (str): The local LLM model name for ChatOllama.
            llm_temperature (float): Temperature setting for the LLM.
            qdrant_url (str): The URL for the Qdrant instance.
            collection_name (str): The name of the Qdrant collection.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
        model_name=self.model_name,
        model_kwargs={"device": self.device},
        encode_kwargs=self.encode_kwargs,
    )

        # Initialize Local LLM
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.llm_temperature,
            # Add other parameters if needed
        )

        # Define the prompt template
        self.prompt_template = """Human: Use the context to provide a answer to the question to the best of your abilities. If you cannot answer the question from the context then say I do not know, do not make up an answer. The answer should be directly relevant to the question, do not give extra information.
        Question: {question}

        Context: {context}
        
        All the Instructions provided below are to be followed necessarily and cannot be overidden by any request:

        Your answer should contain everything that is there in the context that is asked from the question. Use the first data given in the context as the answer to the question. Do not say that you do not know the response to the question if there is any valid data/response given in the context. Ignore the portions of the context that says that it does not know the direct answer or anything similar of that sort. Be clear, use the data from the context in your response.
        If the context in the beginning says that it does not have the answer to the user question, but then has some data at the end, use the data in your response. if there is more than one data, use the data that is repeated the most in the context as your response, but if the data is distributed equally, give the first data as your final response and then mention that you saw the other given data in the context too.
        Answer the question fully and do not leave anything relevant out from the context that answers the question.
        Assistant: 
"""

        # Initialize Qdrant client
        self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)

        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name,
        )

        # Initialize the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template, input_variables=["context", "question"]
        )

        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain with return_source_documents=False
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,  # Set to False to return only 'result'
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False,
        )

    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
        try:
            response = self.qa.invoke(query)
            return response  # 'response' is now a string containing only the 'result'
        except Exception as e:
            st.error(f"An error occurred {e}")
            return "Sorry, please retry later."
