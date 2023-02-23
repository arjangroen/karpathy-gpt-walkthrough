import os
import torch

if "Book 3 - The Prisoner of Azkaban.txt" not in os.listdir('.'):
    os.system("wget https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%203%20-%20The%20Prisoner%20of%20Azkaban.txt")

with open("Book 3 - The Prisoner of Azkaban.txt", 'r') as file:
    textlines = file.readlines()

text = ""
for line in textlines:
    if "Harry Potter and the Prisoner of Azkaban - J.K. R" in line:
        continue
    else:
        text = text + line

text = text.replace("]","")

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print(encode("Harry Potter"))
print(decode(encode("Harry Potter")))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data.dtype)