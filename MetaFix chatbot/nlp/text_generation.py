from groq import Groq
from pathlib import Path
import sys
import os

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from logging_utils import setup_logger

# Set up logger
logger = setup_logger(__name__)

GROQ_API_KEY = os.getenv("groq_api_key")
client = Groq(api_key=GROQ_API_KEY)
#client = Groq(api_key=os.getenv("groq_api_key"))

def text_generator(query, content):
    logger.info(f"Generating text for query: {query[:100]}...")  # Log first 100 chars of query
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that filters and displays solutions based on their relevance.\n"
                        "Follow these criteria:\n"
                        "- If similarity score >= 0.8, include it under 'Highly Suitable Solutions'.\n"
                        "- If 0.5 < similarity score < 0.8, include it under 'Potential Solutions'.\n"
                        "- If similarity score ≤ 0.5, exclude the solution.\n"
                        "Display solutions in the following format:\n"
                        "1. Start with the message: 'Below are the steps to address the issue:'\n"
                        "2. Show 'Highly Suitable Solutions' only if solutions meet this criterion.\n"
                        "3. Show 'Potential Solutions' only if solutions meet this criterion.\n"
                        "4. Do not display severity, category, or similarity scores.\n"
                        "5. If no solutions exist for a category, do not display that category.\n"
                        "Display only the solutions."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n"
                        "Provided Solutions:\n"
                        f"{content}\n"
                        "Filter and display the solutions based on the criteria."
                    )
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_completion_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        
        final_result = chat_completion.choices[0].message.content
        logger.info("Text generation completed successfully")
        return final_result
        
    except Exception as e:
        logger.error(f"Error in text generation: {str(e)}", exc_info=True)
        raise