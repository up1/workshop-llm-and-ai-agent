import tiktoken

# Example text to encode and decode
text = "Hello, world. Is this-- a test?"
print("Original text:", text)

# Get the encoding for the GPT-4 model
enc = tiktoken.encoding_for_model("gpt-4o")
# Encode the text into tokens
tokens = enc.encode(text)
print("Encoded tokens:", tokens)

# Decode the tokens back into text
decoded_text = enc.decode(tokens)
print("Decoded text:", decoded_text)

# Get the number of tokens in the text
num_tokens = len(tokens)
print("Number of tokens:", num_tokens)
