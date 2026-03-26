from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")

sentence = "Hello, y'all! How are you 😁 ?"
output = tokenizer.encode(sentence)
print(output.tokens)
print(output.ids)
print(output.offsets[9])
print(tokenizer.decode(output.ids))
print(sentence[26:27])