from pinecone import Pinecone
from typing import List, Dict, Any
from sentence_transformer import generate_embeddings
from pathlib import Path
import sys
from dotenv import load_dotenv
import os
import config_api

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from logging_utils import setup_logger

# Load .env file
load_dotenv(dotenv_path='../.env')

# Set up logger
logger = setup_logger(__name__)
api_key = config_api.PINECONE_API_KEY
#api_key=os.getenv("PINECONE_API_KEY")

def query_pinecone(
        query,
        api_key=api_key,
        index_name="incident-solutions",
        top_k: int = 20,
        namespace: str = "ns1",
        include_metadata: bool = True,
        include_values: bool = False
    ):
    
    logger.info(f"Processing query: {query[:100]}...")  # Log first 100 chars of query
    
    try:
        query_vector = generate_embeddings(query)
        logger.info("Query vector generated successfully")
        
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        logger.info("Pinecone initialized successfully")
        
        stats = index.describe_index_stats()
        
        if namespace not in stats.namespaces:
            logger.error(f"Namespace '{namespace}' not found. Available namespaces: {list(stats.namespaces.keys())}")
            raise ValueError(f"Namespace '{namespace}' not found. Available namespaces: {list(stats.namespaces.keys())}")
        
        query_response = index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=include_metadata,
            include_values=include_values
        )
        logger.info(f"Query completed successfully. Found {len(query_response['matches'])} matches")
        
        results = []
        for match in query_response['matches']:
            result = {
                'id': match['id'],
                'score': match['score']
            }
            
            if include_metadata and 'metadata' in match:
                result['metadata'] = match['metadata']
                
            if include_values and 'values' in match:
                result['values'] = match['values']
                
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in query_pinecone: {str(e)}", exc_info=True)
        raise

