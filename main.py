from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


# Load data and tokenizer
dataset = load_dataset('csv', data_files=['ArgKP_dataset.csv'])
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")



tokenizer(dataset["train"][0]["argument"])
s = 0

def get_label(dataset):
    return [label["label"] for label in dataset["train"]]

def concat_and_pad(x,y):
    return x + y[1:] + [0] * (512 - len(x) - len(y) + 1)

def tokenize_function(examples):
    argument_tokenized = tokenizer(examples["argument"])
    key_point_tokenized = tokenizer(examples["key_point"])
    concat_tokenized = list(map(concat_and_pad, argument_tokenized["input_ids"], key_point_tokenized["input_ids"]))
    return argument_tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)
labels = get_label(dataset)


s = 0