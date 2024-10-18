import os
import docx
import math
import numpy as np
from sklearn.preprocessing import normalize
from k_means_constrained import KMeansConstrained
from sentence_transformers import SentenceTransformer
import json

def read_docx(file_path):
    """
    Read text from a .docx file.
    
    Parameters:
    - file_path: Path to the .docx file.
    
    Returns:
    - A string containing the extracted text.
    """
    doc = docx.Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def load_sentence_bert_model(model_name='all-MiniLM-L6-v2'):
    """
    Load a pre-trained Sentence-BERT model.
    
    Parameters:
    - model_name: Name of the pre-trained model to load.
                  Default is 'all-MiniLM-L6-v2'.
    
    Returns:
    - An instance of SentenceTransformer.
    """
    print(f"Loading Sentence-BERT model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("Model loaded successfully.")
    return model

def main(docx_directory, K, model_name='all-MiniLM-L6-v2'):
    """
    Cluster student bios into K equal-sized clusters based on Sentence-BERT embeddings.
    
    Parameters:
    - docx_directory: Path to the directory containing .docx files.
    - K: Number of clusters.
    - model_name: Name of the pre-trained Sentence-BERT model to use.
    
    Returns:
    - A list of clusters, each containing student names.
    """
    # Load Sentence-BERT model
    model = load_sentence_bert_model(model_name)
    
    # Read and encode all bios
    student_names = []
    bio_texts = []
    
    print("Reading and encoding student bios...")
    for filename in os.listdir(docx_directory):
        if filename.endswith('.docx'):
            student_name = os.path.splitext(filename)[0]
            file_path = os.path.join(docx_directory, filename)
            text = read_docx(file_path)
            if not text.strip():
                print(f"Warning: {student_name}'s bio is empty.")
            student_names.append(student_name)
            bio_texts.append(text)
            print(f"Processed {student_name}")
    
    # Generate embeddings for all bios
    bio_embeddings = model.encode(bio_texts, show_progress_bar=True)
    
    # Convert embeddings to a NumPy array
    bio_embeddings = np.array(bio_embeddings)
    
    # Normalize embeddings to unit length
    bio_embeddings = normalize(bio_embeddings)
    
    # Check if K divides the number of students
    num_students = len(student_names)
    
    #minimum cluster size; this allows the number of students to be not exactly divisible by K. Max cluster size = min+1
    cluster_size = math.floor(num_students // K)
    
    # Perform balanced K-Means clustering
    print("Clustering...")
    kmeans = KMeansConstrained(
        n_clusters=K,
        size_min=cluster_size,
        size_max=cluster_size+1, 
        random_state=42
    )
    kmeans.fit(bio_embeddings)
    labels = kmeans.labels_
    print("Clustering completed.")
    
    # Organize students into clusters
    clusters = [[] for _ in range(K)]
    for student, label in zip(student_names, labels):
        clusters[label].append(student)
    
    # Verify cluster sizes
    for idx, cluster in enumerate(clusters):
        if (len(cluster) > cluster_size+1) | (len(cluster) < cluster_size) :
            print(f"Warning: Cluster {idx + 1} has size {len(cluster)} outside of [ {cluster_size}, {cluster_size+1} ]")
    
    # Output the clusters
    for idx, cluster in enumerate(clusters):
        print(f"\nCluster {idx + 1}:")
        for student in cluster:
            print(f"- {student}")
    
    # Save the clusters to a JSON file
    with open('student_clusters.json', 'w') as f:
        json.dump(clusters, f, indent=4)
    
    return clusters

if __name__ == "__main__":
    # Example usage
    # Define the directory containing the .docx files
    docx_dir = 'C:/Python/Clustering_algo/TC.py/Data/Student_Profiles/'  # Replace with your directory path
    
    # Define the number of clusters
    K = 5  # Example: 5 clusters
    
    # Define the pre-trained Sentence-BERT model to use
    model_name = 'all-MiniLM-L6-v2'  # You can change this to another model if desired
    
    # Perform clustering
    clusters = main(docx_dir, K, model_name)
    
    # Optional: Save the clusters to a file (already done in the main function)
    # The clusters are saved as 'student_clusters.json'
