import os
import torch
from models import BigramLanguageModel, AttentionBigramLanguageModel, get_batch
import logging
logging.basicConfig(level=logging.INFO)

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

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

split = int(len(data) * 0.9)

train = data[:split]
test = data[split:]

model = AttentionBigramLanguageModel(vocab_size=vocab_size)

xb, yb = get_batch(train)

logits, loss = model(xb, yb)

@torch.no_grad()
def check_validation_loss():
    model.eval()
    xb, yb = get_batch(data=test)
    _, loss = model(xb, yb)
    model.train()
    return loss


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
early_stopping_rounds = 8
min_loss = 5.
rem = early_stopping_rounds

for steps in range(20000):
    xb, yb = get_batch(data=train)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % 50 == 0:
        val_loss = check_validation_loss()
        info = "train loss: " + str(loss.detach().numpy()) + " val loss: " + str(val_loss.numpy())
        logging.info(info)
        val_loss_np = val_loss.numpy()
        if val_loss_np < min_loss:
            min_loss = val_loss_np
            rem  = early_stopping_rounds
        else:
            rem -= 1
        
    if rem == 0:
        logging.info(f"early stopping triggered at step {steps}")
        break

torch.save(model.state_dict(), "magicGPT")

        

        

print(loss.item())
generated = model.generate(torch.zeros((1,1), dtype=torch.long),max_new_tokens=1000)


with open('magicGPT.txt', 'w') as file:
    file.write(decode(generated[0,:].tolist()))


