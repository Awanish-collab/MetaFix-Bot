import streamlit as st
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from logging_utils import setup_logger

# Set up logger
logger = setup_logger(__name__)

# Add vector_db directory to Python path
vector_db_path = Path(__file__).parent.parent / 'vector_db'
sys.path.append(str(vector_db_path))

from query import query_pinecone
from nlp.text_generation import text_generator

def log_conversation(query, response, category):
    """Log conversation details including query, response, and metadata"""
    conversation_log = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'response': response,
        'category': category,
    }
    logger.info(f"Conversation Log: {json.dumps(conversation_log, ensure_ascii=False, indent=2)}")

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    logger.info("Initializing session state")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = "All Categories"

def display_chat_message(role, content, avatar=None):
    """Display a chat message with the specified role and content"""
    logger.debug(f"Displaying chat message - Role: {role}")
    with st.chat_message(role, avatar=avatar):
        st.write(content)

def get_unique_categories(results):
    """Extract unique categories from results"""
    logger.debug("Extracting unique categories from results")
    categories = set()
    for result in results:
        if 'metadata' in result and 'category' in result['metadata']:
            categories.add(result['metadata']['category'])
    return sorted(list(categories))

def format_pinecone_response(query, results, selected_category):
    logger.info(f"Formatting response for query: {query[:100]}...")
    logger.info(f"Selected category: {selected_category}")
    
    formatted_response = "Here are the relevant solutions"
    if selected_category != "All Categories":
        formatted_response += f" for category: {selected_category}"
    formatted_response += ":\n"
    
    # Log matched results details
    for result in results:
        if 'metadata' in result:
            logger.debug(f"Match found - Score: {result['score']}, Category: {result['metadata'].get('category', 'N/A')}")
    
    count = 1
    for result in results:
        if 'metadata' in result and 'solution' in result['metadata']:
            category = result['metadata'].get('category', 'N/A')
            if selected_category == "All Categories" or category == selected_category:
                severity = result['metadata'].get('severity', 'N/A')
                formatted_response += f"\n{count}. **Solution:** {result['metadata']['solution']}\n"
                formatted_response += f"   - Severity: {severity}\n"
                formatted_response += f"   - Category: {category}\n"
                formatted_response += f"   - Score: {result['score']}\n"
                count += 1
    
    logger.info(f"Found {count-1} matching solutions")
    final_result = text_generator(query, formatted_response)
    
    if count == 1:
        logger.warning("No solutions found for the selected category")
        return "No solutions found for the selected category."
    
    return final_result

def main():
    logger.info("Starting application")
    st.title("📚 Knowledge Base Q&A Bot")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("Filters & Controls")
        st.divider()
        
        try:
            all_results = query_pinecone("")
            categories = ["All Categories"] + get_unique_categories(all_results)
            
            st.session_state.selected_category = st.selectbox(
                "Select Category",
                categories,
                index=categories.index(st.session_state.selected_category),
                help="Filter solutions by category"
            )
        except Exception as e:
            logger.error(f"Error loading categories: {str(e)}", exc_info=True)
            st.error("Unable to load categories")
        
        st.divider()
        if st.button("Clear Chat History", type="primary"):
            logger.info("Clearing chat history")
            st.session_state.chat_history = []
            st.rerun()
    
    for message in st.session_state.chat_history:
        display_chat_message(
            role=message["role"],
            content=message["content"],
            avatar="🧑‍💻" if message["role"] == "user" else "🤖"
        )
    
    if query := st.chat_input("Ask your question here..."):
        logger.info(f"New user query received at {datetime.now().isoformat()}")
        logger.info(f"Query content: {query}")
        
        display_chat_message("user", query, avatar="🧑‍💻")
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })
        
        with st.spinner("Searching for solutions..."):
            try:
                start_time = time.time()
                results = query_pinecone(query)
                query_time = time.time() - start_time
                logger.info(f"Query processing time: {query_time:.2f} seconds")
                
                if results:
                    response = format_pinecone_response(query, results, st.session_state.selected_category)
                    # Log the complete conversation
                    log_conversation(query, response, st.session_state.selected_category)
                else:
                    logger.warning("No results found for query")
                    response = "I couldn't find any relevant information for your query. Please try rephrasing your question."
                    log_conversation(query, response, st.session_state.selected_category)
                
                time.sleep(0.5)
                
                display_chat_message("assistant", response, avatar="🤖")
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                st.error(error_message)
                
                # Log the error conversation
                log_conversation(query, error_message, st.session_state.selected_category)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_message
                })

if __name__ == "__main__":
    main()