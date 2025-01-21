from data_extraction import prepare_ticket_data
from pinecone import Pinecone
from sentence_transformer import generate_embeddings
import os

def initialize_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    return Pinecone(api_key=api_key)


def load_and_generate_embeddings(file_path):
    data = prepare_ticket_data(file_path)
    print("Data Extraction Done, Now Embedding Started ...")
    embeddings_data = []
    
    for d in data[0]:
        embeddings = generate_embeddings(d)
        embeddings_data.append(embeddings)
    
    return data, embeddings_data


def prepare_vectors(data, embeddings_data):
    """
    Prepare vectors with metadata for Pinecone indexing
    """
    vectors = []
    for i in range(len(embeddings_data)):
        vectors.append({
            "id": data[1][i],
            "values": embeddings_data[i],
            "metadata": data[2][i]
        })
    return vectors


def upsert_to_pinecone(index, vectors, namespace, batch_size=100):
    # Process vectors in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(
                vectors=batch,
                namespace=namespace
            )
            print(f"Uploaded batch {i//batch_size + 1} of {len(vectors)//batch_size + 1}")
        except Exception as e:
            print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
            raise


def main():
    # Initialize Pinecone
    pc = initialize_pinecone()
    
    # Load data and generate embeddings
    data, embeddings_data = load_and_generate_embeddings(
        "../assets/support_issues_enhanced_data.csv"
    )
    print("Embedding Generation Completed, Now Creating Vectors and Upserting the data into Vector DB ...")
    # Prepare vectors for indexing
    vectors = prepare_vectors(data, embeddings_data)
    
    # Initialize index and upsert vectors
    index = pc.Index("incident-solutions")
    upsert_to_pinecone(index, vectors, "ns1", batch_size=100)  # Added batch_size parameter
    print("Data Inserted Successfully ...")
    


if __name__ == "__main__":
    main()