import torch

# Input token IDs for a sequence of 4 tokens
input_ids = torch.tensor([2, 3, 5, 1])
print("Input token IDs:", input_ids)

# Define the vocabulary size and output dimension for the embedding
vocab_size = 6
output_dim = 3
print("\nVocabulary size:", vocab_size)
print("Output dimension:", output_dim)
print("\n")


# Create an embedding layer
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# Get the embeddings for the input token IDs
print("Emedding layer weights:\n", embedding_layer.weight)

# To convert a token with id 3 into a 3-dimensional vector
print("\nEmbedding for token ID 3:\n", embedding_layer(torch.tensor([3])))

# Get the embeddings for the entire input sequence
print("\nEmbeddings for the entire input sequence:\n", embedding_layer(input_ids))

# convert embeddings to token IDs
# This is not a straightforward process, as multiple token IDs can have similar embeddings.
# However, we can find the closest token ID for a given embedding by calculating the cosine similarity between the embedding and the weights of the embedding layer.
from torch.nn.functional import cosine_similarity
def embedding_to_token_id(embedding, embedding_layer):
    # Calculate cosine similarity between the embedding and the weights of the embedding layer
    similarities = cosine_similarity(embedding.unsqueeze(0), embedding_layer.weight)
    # Get the index of the most similar token ID
    token_id = torch.argmax(similarities).item()
    return token_id

# Example: Convert the embedding of token ID 3 back to a token ID
embedding = embedding_layer(torch.tensor([3]))
token_id = embedding_to_token_id(embedding, embedding_layer)
print("\nOriginal token ID:", 3)
print("Recovered token ID from embedding:", token_id)

# Token id to sentence
