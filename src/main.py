import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tqdm

class ChatDataset(Dataset):
    def __init__(self, file_path: str, tokenizer):
        self.conversations = self.load_conversations(file_path)
        self.encoded_data = self.encode_conversations(tokenizer)

    def load_conversations(self, file_path: str):
        with open(file_path, "r") as file:
            data = json.load(file)
        
        conversations = []
        for item in data:
            for dialog in item['dialog']:
                conversations.append(dialog['text'])
        
        formatted_conversations = [
            f"<startofstring> {conversations[i]} <bot>: {conversations[i + 1]} <endofstring>"
            for i in range(len(conversations) - 1)
        ][:5000]
        
        return formatted_conversations

    def encode_conversations(self, tokenizer):
        encoded_data = tokenizer(
            self.conversations,
            max_length=40,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'input_ids': encoded_data['input_ids'],
            'attention_mask': encoded_data['attention_mask']
        }

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index):
        return self.encoded_data['input_ids'][index], self.encoded_data['attention_mask'][index]

def train_model(data_loader, model, optimizer, num_epochs=12, start_epoch=0):
    model.train()
    best_loss = float('inf')
    
    with open("training.log", "a") as log_file:
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            for input_ids, attention_mask in tqdm.tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                optimizer.zero_grad()
                loss = model(input_ids, attention_mask=attention_mask, labels=input_ids).loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(data_loader)
            log_file.write(f"Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}\n")
            print(f"Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "model_best.pt")
                print(f"Model improved and saved as best_model.pt")

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"model_state_epoch_{epoch + 1}.pt")

def generate_response(user_input):
    formatted_input = f"<startofstring> {user_input} <bot>: "
    tokenized_input = tokenizer(formatted_input, return_tensors="pt")
    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)
    generated_output = model.generate(input_ids, attention_mask=attention_mask, max_length=100)
    return tokenizer.decode(generated_output[0], skip_special_tokens=True)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({
    "pad_token": "<pad>",
    "bos_token": "<startofstring>",
    "eos_token": "<endofstring>"
})
tokenizer.add_tokens(["<bot>:"])

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.to(device)

chat_dataset = ChatDataset("./chat_data.json", tokenizer)
data_loader = DataLoader(chat_dataset, batch_size=64)

optimizer = Adam(model.parameters(), lr=1e-5)

checkpoint_path = "model_latest.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0

print("Training...")
train_model(data_loader, model, optimizer, start_epoch=start_epoch)

print("Infer from model: ")
while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        break
    print("Bot:", generate_response(user_input))