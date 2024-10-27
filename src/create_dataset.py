import json
import re
from datasets import load_dataset

def clean_content(content):
    if isinstance(content, str):
        content = content.strip()
        # Remove unwanted characters including specific Unicode characters
        content = re.sub(r"[\"'.,!?;]", "", content)  # Remove specific punctuation
        content = re.sub(r"[\u2019\u2018]", "", content)  # Remove specific Unicode characters
        content = re.sub(r"[^\x00-\x7F]+", "", content)  # Remove all non-ASCII characters
        return content
    return ""

dataset_urls = [
    "AlekseyKorshuk/synthetic-romantic-characters",
    "AlekseyKorshuk/synthetic-friendly-characters",
    "AlekseyKorshuk/synthetic-fight-characters"
]

all_conversations = []

for url in dataset_urls:
    dataset = load_dataset(url)
    filtered_conversation = []
    for conversation in dataset['train']:
        for entry in conversation['conversation']:
            cleaned_content = clean_content(entry["content"])
            if cleaned_content:  # Only include non-empty cleaned content
                filtered_conversation.append({
                    "content": cleaned_content,
                    "role": entry["role"]
                })
    all_conversations.append({
        "dataset": url.split('/')[-1],
        "conversations": filtered_conversation
    })

with open('dataset.json', 'w') as json_file:
    json.dump(all_conversations, json_file, indent=4)

print("Conversations have been saved to dataset.json")