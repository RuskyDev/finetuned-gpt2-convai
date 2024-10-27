from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import tqdm
import torch
import json

class Dataset(Dataset):
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

def train_model(data_loader, model, optimizer, num_epochs=12):
    model.train()
    for epoch in tqdm.tqdm(range(num_epochs)):
        for input_ids, attention_mask in data_loader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            loss = model(input_ids, attention_mask=attention_mask, labels=input_ids).loss
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), "model_state.pt")
        print(generate_response("hello how are you"))

def generate_response(user_input):
    formatted_input = f"<startofstring> {user_input} <bot>: "
    tokenized_input = tokenizer(formatted_input, return_tensors="pt")
    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)
    generated_output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(generated_output[0])

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

chat_dataset = Dataset("./chat_data.json", tokenizer)
data_loader = DataLoader(chat_dataset, batch_size=64)

optimizer = Adam(model.parameters(), lr=1e-3)

print("Training...")
train_model(data_loader, model, optimizer)

print("Infer from model: ")
while True:
    user_input = input()
    print(generate_response(user_input))