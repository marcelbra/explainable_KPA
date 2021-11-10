from datasets import load_dataset
from transformers import AutoTokenizer


# Load data
dataset = load_dataset('csv', data_files=['ArgKP_dataset.csv'])
# dataset["train"][0]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

s = 0