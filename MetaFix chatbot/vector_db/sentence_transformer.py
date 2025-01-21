from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from logging_utils import setup_logger 
# Set up logger
logger = setup_logger(__name__)

def generate_embeddings(text):
    logger.info("Generating embeddings for text")
    try:
        model = init_embedding_model()
        embeddings = model.encode(text).tolist()
        logger.info("Embeddings generated successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        raise

def init_embedding_model(embd_model_type='Sentence_Transformer', model_name="BAAI/bge-large-en"):
    logger.info(f"Initializing embedding model: {model_name}")
    try:
        if embd_model_type == 'Sentence_Transformer':
            embd_model = SentenceTransformer(model_name)
            logger.info("Embedding model initialized successfully")
        else:
            embd_model = None
            logger.warning(f"Unsupported embedding model type: {embd_model_type}")
        return embd_model
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}", exc_info=True)
        raise