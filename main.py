"""
Current main workspace for explainable AI practical

By:
Marcel Braasch
Saransh Agarwal
Pınar Ayaz
"""

from copy import copy
from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm_notebook
import pandas as pd
import torch

df = pd.read_csv("../data/valid_df.csv", keep_default_na=False)
entailment_model = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
tokenizer = AutoTokenizer.from_pretrained(entailment_model)
model = AutoModelForSequenceClassification.from_pretrained(entailment_model).to("cuda:0")

MAX_LENGTH=256

def leave_one_out(hypothesis, premise):
    print(f"Hypothesis: {hypothesis}")
    print(f"Premise: {premise}\n")
    n = len(premise.split()) + len(hypothesis.split())
    true_score = compute_entailment(hypothesis, premise)["entail"]
    print(f"True entailment score: {true_score}\n")
    for i in range(n):
        premise_copy = copy(premise).split()
        hypothesis_copy = copy(hypothesis).split()
        # Drop word in the respective sequence
        index = None
        if i < len(premise_copy):
            dropped_word = premise_copy.pop(i)
            which = "premise"
        else:
            index = i - len(premise_copy) - 1
            dropped_word = hypothesis_copy.pop(index)
            which = "hypothesis"
        premise_copy = " ".join(premise_copy)
        hypothesis_copy = " ".join(hypothesis_copy)
        score = compute_entailment(hypothesis_copy, premise_copy)["entail"]
        print(f"Dropping word {index+1 if index else i+1} \"{dropped_word}\" in {which}.")
        print(f"Entailment score is: {score}")
        print(f"That's a difference of {true_score-score}.\n")

def compute_entailment(hypothesis, premise):
    tokenized_input_seq_pair = tokenizer.encode_plus(hypothesis, premise, max_length=MAX_LENGTH, return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).cuda()
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).cuda()
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).cuda()
    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None)
    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
    entailment_prob = predicted_probability[0]
    neutral_prob = predicted_probability[1]
    contradiction_prob = predicted_probability[2]
    result = {'entail':entailment_prob, 'neutral':neutral_prob, 'contradict':contradiction_prob}
    return result

def compute_entailment_preprocessed(tokenized_input_seq_pair):
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_input_seq_pair)
    input_ids = torch.Tensor(input_ids).long().unsqueeze(0).cuda()
    #token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).cuda()
    #attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).cuda()
    outputs = model(input_ids)#, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None)
    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
    entailment_prob = predicted_probability[0]
    neutral_prob = predicted_probability[1]
    contradiction_prob = predicted_probability[2]
    result = {'entail':entailment_prob, 'neutral':neutral_prob, 'contradict':contradiction_prob}
    return result


def load_data():
    data = pd.read_csv("../../argmining-21-keypoint-analysis-sharedtask-code/data/arg-kp/ArgKP_dataset.csv")

    j = 0
    mapping = defaultdict(list)
    for row in data.iterrows():
        if j == 15: break
        sample = row[1]
        argument = sample["argument"]
        key_point = sample["key_point"]
        mapping[argument].append(key_point)
        j += 1

    new_data = pd.DataFrame()
    args = []
    kps = []
    dropped = []
    scores = []

    for argument, key_points in tqdm_notebook(mapping.items()):
        for key_point in key_points:

            tokenized_input_seq_pair = tokenizer.encode_plus(argument, key_point, max_length=MAX_LENGTH,
                                                             return_token_type_ids=True, truncation=True)
            tokens = tokenized_input_seq_pair[0].tokens
            argument_tokens = tokens[:tokens.index("</s>")+1]
            key_point_tokens = tokens[tokens.index("</s>")+1:]
            current_argument = copy(argument_tokens)
            skip_words = ["<s>", "</s>", '`', ".", ","]

            i = 0
            for i in range(len(argument_tokens)):
                curr_drop = []
                current_argument = copy(argument_tokens)
                current_token = current_argument[i]
                if current_token not in skip_words:
                    curr_drop.append(current_argument[i])
                    current_argument[i] = "50264"  # <mask> token
                    while ((i+1) < len(tokens)
                           and not tokens[i+1].startswith('Ġ')  # marks new word
                           and tokens[i+1] not in skip_words):
                        i += 1
                        curr_drop.append(current_argument[i])
                        current_argument[i] = "50264"

                dropped.append(curr_drop)
                args.append(argument)
                kps.append(key_point)
                scores.append(compute_entailment_preprocessed(current_argument+key_point_tokens)["entail"])

    new_data["arg"] = args
    new_data["kp"] = kps
    new_data["dropped"] = dropped
    new_data["score"] = scores
    new_data.to_csv("/home/marcelbraasch/PycharmProjects/explainable_KPA/calculations.csv", index=False)

def unique(seq):
    """Order preserving unique function."""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def kendalls_tau():
    df = pd.read_csv("calculations.csv")
    args = list(set(df["arg"].tolist()))
    for arg in args:
        df_arg = df.loc[df.loc[:,'arg']==arg,:]
        dropped_words = unique(df_arg["dropped"].tolist())
        for word in dropped_words:
            df_arg_dropped = df_arg.loc[df_arg.loc[:,'dropped']==word,:]
            df_arg_dropped = df_arg_dropped.drop_duplicates()
            s = 0

"""
s1 = compute_entailment("a person has the right to end their suffering and if somebody takes pity on them and chooses to help, that person should not be punished.", "Assisted suicide reduces suffering")
s2 = compute_entailment("a person has the right to end their suffering and if somebody takes pity on them and chooses to help, that person should not be punished.", "Assisted suicide gives dignity to the person that wants to commit it")
s3 = compute_entailment("a person has the right to end their suffering and if somebody takes pity on them and chooses to help, that person should not be punished.", "The terminally ill would benefit from assisted suicide")
random = compute_entailment("a person has the right to end their suffering and if somebody takes pity on them and chooses to help, that person should not be punished.", "People should have the freedom to choose to end their life")
s = 0
"""