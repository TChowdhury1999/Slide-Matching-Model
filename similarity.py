from create_embeddings import preprocess_image, extract_embedding
import numpy as np
import json

# Calculate similarity
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# Retrieve most similar image
def find_most_similar(query_embedding, database):
    max_similarity = -1
    most_similar_image = None
    for filename, embedding in database.items():
        embedding = np.array(embedding)
        similarity = cosine_similarity(query_embedding, embedding)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_image = filename
    return most_similar_image, max_similarity

# Read the embeddings dictionary from the JSON file
with open("embeddings_dict.json", 'r') as f:
    embeddings_dict = json.load(f)

# Example usage
query_image_path = "data/slide_query_images/Slide5.jpg"
query_embedding = extract_embedding(query_image_path)
most_similar_image, similarity_score = find_most_similar(query_embedding, embeddings_dict)
print("Most similar image:", most_similar_image)
print("Similarity score:", similarity_score)