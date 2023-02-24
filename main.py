import os
import torch
from gpt_utils import get_batch
from models import BigramLanguageModel

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

split = int(len(data) * 0.9)

train = data[:split]
test = data[split:]

block_size = 8

model = BigramLanguageModel(vocab_size=vocab_size)

xb, yb = get_batch(train)

logits, loss = model(xb, yb)


