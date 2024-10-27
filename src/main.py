from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import tqdm
import torch
import json

class ChatDataset(Dataset):
    def __init__(self, FilePath: str, Tokenizer):
        self.Conversations = self.load_data(FilePath)
        self.EncodedData = self.tokenize_data(Tokenizer)

    def load_data(self, FilePath: str):
        with open(FilePath, "r") as file:
            Data = json.load(file)
        
        Conversations = []
        for Item in Data:
            for Dialog in Item['dialog']:
                Conversations.append(Dialog['text'])
        
        return Conversations[:5000]

    def tokenize_data(self, Tokenizer):
        TokenizedData = Tokenizer(
            [f"<startofstring> {self.Conversations[i]} <bot>: {self.Conversations[i + 1]} <endofstring>"
             for i in range(len(self.Conversations) - 1)],
            max_length=40,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'InputIds': TokenizedData['input_ids'],
            'AttentionMask': TokenizedData['attention_mask']
        }

    def __len__(self):
        return len(self.Conversations) - 1

    def __getitem__(self, Index):
        return self.EncodedData['InputIds'][Index], self.EncodedData['AttentionMask'][Index]

def TrainModel(DatasetLoader, Model, Optimizer, NumEpochs=12):
    for Epoch in tqdm.tqdm(range(NumEpochs)):
        for InputIds, AttentionMask in DatasetLoader:
            InputIds = InputIds.to(Device)
            AttentionMask = AttentionMask.to(Device)
            Optimizer.zero_grad()
            Loss = Model(InputIds, attention_mask=AttentionMask, labels=InputIds).loss
            Loss.backward()
            Optimizer.step()
        torch.save(Model.state_dict(), "model_state.pt")
        print(GenerateResponse("hello how are you"))

def GenerateResponse(UserInput):
    FormattedInput = f"<startofstring> {User Input} <bot>: "
    TokenizedInput = Tokenizer(FormattedInput, return_tensors="pt")
    InputIds = TokenizedInput["input_ids"].to(Device)
    AttentionMask = TokenizedInput["attention_mask"].to(Device)
    GeneratedOutput = Model.generate(InputIds, attention_mask=AttentionMask)
    DecodedOutput = Tokenizer.decode(GeneratedOutput[0], skip_special_tokens=True)
    return DecodedOutput

Device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
Tokenizer.add_special_tokens({
    "pad_token": "<pad>", 
    "bos_token": "<startofstring>",
    "eos_token": "<endofstring>"
})
Tokenizer.add_tokens(["<bot>:"])

Model = GPT2LMHeadModel.from_pretrained("gpt2")
Model.resize_token_embeddings(len(Tokenizer))
Model = Model.to(Device)

ChatDatasetInstance = ChatDataset("./chat_data.json", Tokenizer)
DatasetLoader = DataLoader(ChatDatasetInstance, batch_size=64)

Model.train()
Optimizer = Adam(Model.parameters(), lr=1e-3)

print("Training .... ")
TrainModel(DatasetLoader, Model, Optimizer)

print("Infer from model: ")
while True:
    UserInput = input()
    print(GenerateResponse(UserInput))