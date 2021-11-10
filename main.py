from datasets import load_dataset
from transformers import AutoTokenizer


# Load data and tokenizer
dataset = load_dataset('csv', data_files=['ArgKP_dataset.csv'])
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")



tokenizer(dataset["train"][0]["argument"])
s = 0




def tokenize_function(examples):
    argument_tokenized = tokenizer(examples["argument"])
    key_point_tokenized = tokenizer(examples["key_point"])
    concat_tokenized = list(map(lambda x,y: x + y[1:] + [0]*(512-len(x)-len(y)+1),
                                argument_tokenized["input_ids"], key_point_tokenized["input_ids"]))

    #k = [1:]
    s = 0
    return argumet_tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)

s = 0