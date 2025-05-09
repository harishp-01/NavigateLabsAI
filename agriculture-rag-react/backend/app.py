from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from config import Config
from src.document_processor.pdf_processor import PDFProcessor
from src.embeddings.text_embeddings import TextEmbedder
from src.embeddings.image_embeddings import ImageEmbedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.helpers import save_uploaded_file, is_pdf, extract_first_page_as_image
from src.utils.logger import get_logger
import warnings
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/static/*": {"origins": "*"}
}, supports_credentials=True)

# Initialize configuration and logging
config = Config()
config.setup()
logger = get_logger(__name__)

# Initialize components
text_embedder = TextEmbedder()
image_embedder = ImageEmbedder()
vector_store = VectorStore()

# Handle vector store loading
if not vector_store.load(config.VECTOR_STORE_PATH):
    logger.warning("Could not load vector store, initializing empty")
    vector_store.initialize_indexes()

pdf_processor = PDFProcessor()
rag_pipeline = RAGPipeline(
    vector_store=vector_store,
    text_embedder=text_embedder
)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = Config().UPLOAD_DIR
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

frontend_build = os.path.join('..', 'frontend', 'build')
app.static_folder = frontend_build

# Serve React frontend in production
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# API Routes
@app.route('/api/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        if is_pdf(file_path):
            # Process PDF
            text_chunks, images = pdf_processor.process_pdf(file_path)
            text_chunks = text_embedder.embed_documents(text_chunks)
            images = image_embedder.embed_images(images)
            
            vector_store.add_texts(text_chunks)
            vector_store.add_images(images)
            vector_store.save(Config().VECTOR_STORE_PATH)
            
            # Extract preview image
            preview_image = extract_first_page_as_image(file_path)
            preview_path = None
            if preview_image:
                preview_path = os.path.join(app.config['UPLOAD_FOLDER'], f"preview_{filename}.jpg")
                preview_image.save(preview_path)
            
            return jsonify({
                'success': True,
                'message': 'PDF processed successfully',
                'preview': preview_path.split('/')[-1] if preview_path else None
            })
        else:
            # Process image
            image = Image.open(file_path)
            image_data = {
                "image": image,
                "caption": "Uploaded image",
                "metadata": {
                    "page_num": 0,
                    "img_index": 0,
                    "source": "upload",
                    "type": "image"
                }
            }
            
            image_data = image_embedder.embed_images([image_data])[0]
            vector_store.add_images([image_data])
            vector_store.save(Config().VECTOR_STORE_PATH)
            
            return jsonify({
                'success': True,
                'message': 'Image processed successfully',
                'preview': filename
            })
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({'error': str(e)}), 500

# API Routes
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    response = rag_pipeline.generate_response(data['message'])
    return _corsify_response(jsonify({
        'answer': response['answer'],
        'sources': [
            {
                'page_num': doc.metadata['page_num'] + 1,
                'content': doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                'type': doc.metadata.get('type', 'text')
            }
            for doc in response['source_documents']
        ]
    }))

@app.route('/api/preview/<filename>', methods=['GET'])
def get_preview(filename):
    try:
        safe_filename = secure_filename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(file_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _build_cors_preflight_response():
    response = jsonify({'success': True})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _corsify_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

if __name__ == '__main__':
    # Verify build directory
    if not os.path.exists(app.static_folder):
        logger.error(f"React build not found at {app.static_folder}")
        logger.info("Run: cd frontend && npm run build")
    
    app.run(host='0.0.0.0', port=5000, debug=True)