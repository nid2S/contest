from transformers import BartForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from datasets import load_metric
from typing import List
import transformers
import pandas as pd
import numpy as np
import torch
import re

# compute_metrics function
metric = load_metric('rouge')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)

# get max length
def get_TextMaxLen() -> int:
    # tokenizer = get_kobart_tokenizer()
    # data = pd.read_csv(r"D:\workspace\Git_project\contest\Korean_Summary\data\train_data.csv")
    # print("get data")
    # result = data["text"]
    # result = result.map(lambda text: text[3:] if re.sub(".\. ", "", text[:3]) == "" else text)
    # result = result.map(lambda text: re.sub("\[[^]]*]|\([^)]*\)", "", text))
    # result = result.map(lambda text: re.sub(r"[.,?!`~+=-_*""'']", "", text.lower()))
    # result = result.map(lambda text: tokenizer.encode(text))
    # print("preprocessing end")  # take long time
    # return max([len(text) for text in result.to_list()])
    return 853

# declare TrainDataset
class TrainDataset(Dataset):
    def __init__(self, train_data_):
        # dataset num = 266032
        train_data_ = train_data_.drop(train_data_[train_data_["summary"].isna()].index)
        train_data_ = train_data_.drop_duplicates(["summary"])
        train_data_ = train_data_.drop_duplicates(["text"])
        # # for fast training
        # train_data_ = train_data_[:int(len(train_data_)/10)]
        # remove Legal provisions (ex - 가. ~~~)
        train_data_["text"] = train_data_["text"].map(lambda text: text[3:] if re.sub(".\. ", "", text[:3]) == "" else text)
        train_data_["summary"] = train_data_["summary"].map(lambda text: text[3:] if re.sub(".\. ", "", text[:3]) == "" else text)
        # remove parenthesis
        train_data_["text"] = train_data_["text"].map(lambda text: re.sub("\[[^]]*]|\([^)]*\)", "", text))
        train_data_["summary"] = train_data_["summary"].map(lambda text: re.sub("\[[^]]*]|\([^)]*\)", "", text))
        # lower, remove non-korean/english/number/space
        train_data_["text"] = train_data_["text"].map(lambda text: re.sub(r"[\W]", " ", text.lower()).strip())
        train_data_["summary"] = train_data_["summary"].map(lambda text: re.sub(r"[\W]", "", text.lower()).strip())
        self.train_data = train_data_

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx: int):
        # trian_data's columns = {id, text, summary}
        _, text, summary = self.train_data.iloc[idx]
        return {"text": text, "label": summary}

# Declare Data Collator (move like tokenizer)
class TrainDataCollator(object):
    def __init__(self):
        self.max_len = get_TextMaxLen()
        self.tokenizer = get_kobart_tokenizer()

    def __call__(self, sequences):
        # expected >> List of {"text": text, "label": summary}
        texts = [sequence["text"] for sequence in sequences]
        labels = [sequence["label"] for sequence in sequences]

        tokened_texts = self.preprocessing(texts)
        tokened_labels = self.preprocessing(labels)

        # return tokened_texts["input_ids"], tokened_labels["input_ids"]
        return {'input_ids': tokened_texts.input_ids, "labels": tokened_labels["input_ids"]}

    def preprocessing(self, texts: List[str]):
        tokened_texts = self.tokenizer(texts, max_length=self.max_len,
                                       padding="max_length", truncation=True, return_tensors="pt")

        return tokened_texts


model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model(),
                                                     num_labels=len(get_kobart_tokenizer().get_vocab()))  # vocab_size = 30000

# load, split data (8:2)
data = pd.read_csv(r"D:\workspace\Git_project\contest\Korean_Summary\data\train_data.csv")
train_data, val_data = train_test_split(data, train_size=0.8, shuffle=True, random_state=1000)
# load dataset, collator
train_datasets = TrainDataset(train_data)
val_datasets = TrainDataset(val_data)
train_collator = TrainDataCollator()

args = TrainingArguments(
    output_dir=r"D:\workspace\Git_project\contest\Korean_Summary\model",
    evaluation_strategy=transformers.EvaluationStrategy.STEPS,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=4,
    seed=1000,
    load_best_model_at_end=True,
)
# define Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_datasets,
    eval_dataset=val_datasets,
    data_collator=train_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
trainer.train()
