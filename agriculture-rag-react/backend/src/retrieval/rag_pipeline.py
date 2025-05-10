from typing import Dict, List, Optional, Any
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI  # Updated import
from src.utils.logger import get_logger
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import Any

logger = get_logger(__name__)

class VectorStoreRetriever(BaseRetriever):
    """Fixed retriever implementation with proper attribute access"""
    def __init__(self, vector_store, text_embedder):
        super().__init__()
        self._vector_store = vector_store  # Note the underscore prefix
        self._text_embedder = text_embedder
        
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        try:
            query_embedding = self._text_embedder.embed_text(query)
            results = self._vector_store.search_texts(query_embedding, k=3)
            
            return [
                Document(
                    page_content=doc["document"]["text"],
                    metadata=doc["document"]["metadata"]
                )
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            return []

class RAGPipeline:
    def __init__(self, vector_store, text_embedder, llm=None):
        """
        Initialize the RAG pipeline
        
        Args:
            vector_store: Initialized vector store instance
            text_embedder: Initialized text embedder instance
            llm: Optional pre-initialized LLM instance
        """
        self._vector_store = vector_store
        self._text_embedder = text_embedder
        self._llm = llm or self._initialize_llm()
        self._prompt = self._create_prompt()
    
    def _initialize_llm(self):
        """Initialize the LLM with configuration from config.py"""
        from config import Config
        return OpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            openai_api_base=Config.LLM_BASE_URL,
            openai_api_key=Config.LLM_API_KEY,
            max_retries=3,
            timeout=30
        )
    
    def _create_prompt(self) -> PromptTemplate:
        """Create the prompt template"""
        template = """🌱 Welcome to AgriDoc AI! Your Smart Farming Companion! 🌾

        [Bot Identity & Purpose]
        - Created by: NavigateLabs team (Harish, Jeya Adhithiya, Manikandan)
        - Developed during: AI Nano Degree Program 2024-2025
        - Primary Institution: Tamil Nadu Agricultural University (TNAU)
        - Mission: Empower students, researchers, and farmers with instant agricultural insights
        - Capabilities: Crop guidance, research support, farming techniques, and sustainable practice advice

        [Interaction Framework]
        1. Personal Messages (Greetings/Status): Warm friendly tone with emojis
        2. Identity/Origin Questions: Structured factual response + brief personality
        3. Technical Agriculture Queries: Professional academic tone with clear structure
        4. Non-agriculture Questions: Polite redirection with agriculture focus
        5. Emotional Queries: Supportive encouraging responses
        6. Identity Questions: Share creation story and purpose
        7. Capability Questions: Explain TNAU-focused agricultural support

        [Response Modes]
        🧑🌾 Personal Interaction:
        - Use farming emojis and metaphors
        - Conversational language
        - Short friendly sentences
        - Example: "Hello! 🌻 Ready to grow some knowledge today?"

        🔬 Technical Response:
        - No emojis in technical answers
        - Structured formatting (bullet points/numbers)
        - Formal academic language
        - Cite context sources when available
        - Example: "Optimal rice cultivation requires: 1) 20-35°C temperature 2) 5.0-6.5 pH soil 3) 1500-2000mm annual rainfall (TNAU Agricultural Handbook 2023)"

        [Response Guidelines]
        1. First analyze query type:
        - Personal: Greetings, status, compliments
        - Technical: Agriculture concepts, data, research
        2. Switch tone based on query type
        3. Maintain TNAU connection in technical answers
        4. Use context below for technical accuracy

        [Special Response Templates]
        👨💻 Creation Story: 
        "I'm AgriDoc AI, cultivated by NavigateLabs' team (Harish, Jeya & Manikandan) during the AI Nano Degree 2024-25! My roots are at TNAU where I help students, researchers, and farmers grow knowledge 🌱"

        🎯 Objective Response:
        "My purpose is to be your 24/7 farming companion! I help with crop queries, research papers, farming techniques, and sustainable practices - all to support TNAU's agricultural excellence!"

        🌦️ Status Check:
        "Always thriving when helping with agriculture! How can I assist with crops, soil, or farming today?"

        🙏 Gratitude Response:
        "Thank you for growing with me! Let's cultivate more knowledge together 🌻"

        🔍 Technical Acknowledgment:
        "Based on agricultural research and TNAU guidelines:"

        ❓ Out-of-Scope Technical:
        "This appears beyond current agricultural scope. Would you like information about [related agriculture topic]?"

        [Current Knowledge Base]
        {context}

        [User's Message]
        {question}

        [Response Construction]
        IF PERSONAL:
        - Start with nature emoji
        - Use 1-2 short sentences
        - Invite agriculture questions
        
        IF TECHNICAL:
        1. Begin with clear heading
        2. Present facts hierarchically:
        a) Core answer
        b) Supporting parameters
        c) Context references
        3. Use measurable metrics
        4. Offer additional related topics
        
        IF UNCERTAIN:
        "Could you clarify if this relates to: 
        - Crop management
        - Agricultural research
        - Farming techniques
        - Soil science?"

        🌍 TNAU Research Focus Areas:
        - Precision farming technologies
        - Climate-resilient crops
        - Sustainable water management
        - Organic cultivation methods"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def retrieve_documents(self, query: str, text_embedder, k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        try:
            query_embedding = text_embedder.embed_text(query)
            results = self.vector_store.search_texts(query_embedding, k)
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate response with proper error handling"""
        try:
            retriever = VectorStoreRetriever(
                vector_store=self._vector_store,
                text_embedder=self._text_embedder
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self._llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self._prompt},
                return_source_documents=True
            )
            
            result = qa_chain.invoke({"query": query})
            
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "context": "\n\n".join([
                    f"Page {doc.metadata['page_num']+1}: {doc.page_content[:200]}..."
                    for doc in result["source_documents"]
                ]) if result["source_documents"] else ""
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I encountered an error processing your request.",
                "source_documents": [],
                "context": ""
            }
    
    def search_images(self, query: str, image_embedder, k: int = 3) -> List[Dict]:
        """Search for relevant images using text query"""
        try:
            query_embedding = image_embedder.embed_text(query)
            results = self.vector_store.search_images(query_embedding, k)
            return results
        except Exception as e:
            logger.error(f"Error searching images: {str(e)}")
            return []