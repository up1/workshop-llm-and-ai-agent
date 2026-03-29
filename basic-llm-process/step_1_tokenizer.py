import re

# Input text
text = "Hello, world. This, is a test."

# Split by whitespace
print("Split by whitespace:")
result = re.split(r'(\s)', text)
print(result)

# Split by comma and period
print("\nSplit by comma and period:")
result = re.split(r'([,.]|\s)', text)
print(result)

# Strip whitespace from each item and then filter out any empty strings.
print("\nStrip whitespace and filter out empty strings:")
result = [item for item in result if item.strip()]
print(result)


# Split by multiple delimiters: comma, period, and whitespace
print("\nSplit by multiple delimiters: comma, period, and whitespace")
text = "Hello, world. Is this-- a test?"

result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)