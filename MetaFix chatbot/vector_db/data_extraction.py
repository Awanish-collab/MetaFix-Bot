from typing import Dict, List
import uuid
import pandas as pd

def prepare_ticket_data(csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)

    documents = [str(row['Short Description/Issue']) for _, row in df.iterrows()]
    
    # Generate IDs if not present
    ids = [str(uuid.uuid4()) if pd.isna(id_) else str(id_) 
           for id_ in df['Id']]
    
    # Create metadata dictionary for each document
    metadata = []
    for _, row in df.iterrows():
        meta = {
            'solution': str(row['Solution']),
            'issue_description': str(row['Short Description/Issue']),
            'severity': str(row['Severity']),
            'category': str(row['Issue Category']),
            'status': str(row['Status']),
            'assigned_to': str(row['Assigned To']),
            'reported_by': str(row['Reported By']),
            'resolution_time': str(row['Resolution Time (hours)']),
            'timestamp': str(row['Timestamp'])
        }
        metadata.append(meta)
    
    return documents, ids, metadata

