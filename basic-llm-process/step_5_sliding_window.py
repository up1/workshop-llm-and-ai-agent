import tiktoken

# Example text to encode and decode
text = "Hello, world. Is this a test?"
print("Original text:", text)

# Get the encoding for the GPT-4 model
enc = tiktoken.encoding_for_model("gpt-4o")
# Encode the text into tokens
tokens = enc.encode(text)
print("Encoded tokens:", tokens)

# Decode the tokens back into text
decoded_text = enc.decode(tokens)
print("Decoded text:", decoded_text)

# Sliding window parameters
context_size = 4

x = tokens[:context_size]
y = tokens[1:context_size+1]

# Print the contexts and desired tokens
print("\nContexts and desired tokens:")
for i in range(1, context_size+1):
    context = tokens[:i]
    desired = tokens[i]

    print(context, "-->", desired)

# Decode the contexts and desired tokens back into text for better readability
print("\nDecoded contexts and desired tokens:")
for i in range(1, context_size+1):
    context = tokens[:i]
    desired = tokens[i]

    print(enc.decode(context), "-->", enc.decode([desired]))
