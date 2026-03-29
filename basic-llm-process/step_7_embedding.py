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