import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    # Application Configuration
    APP_NAME = "Agriculture Document Analysis"
    APP_DESCRIPTION = "AI-powered analysis of agricultural documents using RAG"
    VERSION = "1.0.0"

    # Flask Configuration
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-123")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB upload limit

    # LLM Configuration
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 1.0
    LLM_BASE_URL = os.getenv("LLM_BASE_URL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_TIMEOUT = 30  # seconds

    # Embedding Models
    TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    IMAGE_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
    
    # Vector Store
    VECTOR_STORE_PATH = os.path.join("data", "vector_store")
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_PAGES = 50 
    MAX_PAGE_LENGTH = 1000  # Characters per chunk
    OVERLAP = 200          # Overlap between chunks
    MAX_IMAGE_SIZE = 512   # Max dimension for image processing
    
    # File Storage
    UPLOAD_DIR = os.path.join("data", "uploads")
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
    
    # Cache and Storage
    CACHE_DIR = ".cache"
    LOG_DIR = "logs"

    @property
    def VECTOR_STORE_PATH(self):
        """Get the full path to the vector store"""
        path = os.path.join("data", "vector_store")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    @staticmethod
    def setup():
        """Initialize required directories with error handling"""
        try:
            # Create necessary directories
            os.makedirs("data/uploads", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            os.makedirs(".cache", exist_ok=True)
            
            # Ensure vector store directory exists
            _ = Config().VECTOR_STORE_PATH
            
            # Create initial log files
            open(os.path.join("logs", "app.log"), 'a').close()
            open(os.path.join("logs", "document_processor.log"), 'a').close()
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    @property
    def SQLALCHEMY_DATABASE_URI(self):
        """Database URI (if using SQL database in future)"""
        return f"sqlite:///{os.path.join('data', 'agriculture_rag.db')}"