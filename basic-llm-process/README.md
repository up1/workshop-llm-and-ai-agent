# Workshop :: Basic LLM from Scratch


## 1. Setup project and install dependencies

```
$python -m venv pytorch_env
$source pytorch_env/bin/activate

$export PATH=.:$(pwd)/pytorch_env/bin/:$PATH
$alias pip=pip3

$pip install -r requirements.txt
```

## 2. Tokenizer
```
$python step_1_tokenizer.py 

Split by whitespace:
['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']

Split by comma and period:
['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is', ' ', 'a', ' ', 'test', '.', '']

Strip whitespace and filter out empty strings:
['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
```

## 3. Converting tokens into token IDs
```
$python step_2_token_id.py

(',', 0)
('--', 1)
('.', 2)
('?', 3)
('Hello', 4)
('Is', 5)
('a', 6)
('test', 7)
('this', 8)
('world', 9)
```

## 4. Building a simple language model
* Encoding the input text into token IDs
* Decoding the token IDs back into text
* Add some extensions to the tokenizer to handle more complex tokenization rules
```
$python step_3_simple_language_model.py

Original text: Hello, world. Is this-- a test?
Vocabulary: {',': 0, '--': 1, '.': 2, '?': 3, 'Hello': 4, 'Is': 5, 'a': 6, 'test': 7, 'this': 8, 'world': 9}
Encoded token IDs: [4, 0, 9, 2, 5, 8, 1, 6, 7, 3]
Decoded text: Hello, world. Is this -- a test?
```

## 5. Working with [TikToken](https://github.com/openai/tiktoken)
```
$python step_4_tiktoken.py  

Original text: Hello, world. Is this-- a test?
Encoded tokens: [13225, 11, 2375, 13, 2763, 495, 375, 261, 1746, 30]
Decoded text: Hello, world. Is this-- a test?
Number of tokens: 10
```

## 6. Data sampling with a sliding window
* Sliding window approach, changing the position by +1
* Training data for next token prediction
```
python step_5_sliding_window.py
Original text: Hello, world. Is this a test?
Encoded tokens: [13225, 11, 2375, 13, 2763, 495, 261, 1746, 30]
Decoded text: Hello, world. Is this a test?

Contexts and desired tokens:
[13225] --> 11
[13225, 11] --> 2375
[13225, 11, 2375] --> 13
[13225, 11, 2375, 13] --> 2763

Decoded contexts and desired tokens:
Hello --> ,
Hello, -->  world
Hello, world --> .
Hello, world. -->  Is
```

## 7. Working with PyTorch
* Create a PyTorch Dataset and DataLoader for training a language model
* Use the sliding window approach to create input-target pairs for training

```
$python step_6_working_with_pytorch.py

First batch: [tensor([[13225,    11,  2375,    13]]), tensor([[  11, 2375,   13, 2763]])]
Second batch: [tensor([[  11, 2375,   13, 2763]]), tensor([[2375,   13, 2763,  495]])]
```
