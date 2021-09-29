# !pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
from transformers import BartModel, Trainer, TrainingArguments, EarlyStoppingCallback
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

# TODO check - is it/how it work
# declare TrainDataset
class TrainDataset(Dataset):
    def __init__(self, train_data_):
        # na, 중복 제거 후 데이터 저장
        # dataset num = 266032
        train_data_ = train_data_.drop(train_data_[train_data_["summary"].isna()].index)
        train_data_ = train_data_.drop_duplicates(["summary"])
        train_data_ = train_data_.drop_duplicates(["text"])
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
        self.tokenizer = get_kobart_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, sequences):
        # expected >> List of {"text": text, "label": summary}
        texts = [sequence["text"] for sequence in sequences]
        labels = [sequence["label"] for sequence in sequences]

        tokened_texts = self.preprocessing(texts)
        tokened_labels = self.preprocessing(labels)

        return {"input_ids": tokened_texts["input_ids"], "labels": tokened_labels["input_ids"]}

    def preprocessing(self, texts: List[str]):
        # convert to Serise for Easier preprocessing
        result = pd.Series(texts)
        # remove Legal provisions (ex - 가. ~~~)
        result = result.map(lambda text: text[3:] if re.sub(".\. ", "", text[:3]) == "" else text)
        # remove parenthesis
        result = result.map(lambda text: re.sub("\[[^]]*]|\([^)]*\)", "", text))
        # lower, remove non-korean/english/number/space
        result = result.map(lambda text: re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-z ]", "", text.lower()))
        # tokenize
        tokened_texts = self.tokenizer(result.to_list(), return_tensors="pt").to(self.device)

        return tokened_texts

class SummaryBartModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = get_kobart_tokenizer()
        self.bart = BartModel.from_pretrained(get_pytorch_kobart_model())
        self.softmax = torch.nn.Softmax(dim=len(self.tokenizer.get_vocab()))

    def forward(self, x):
        return self.softmax(self.bart(x)['last_hidden_state'])


# declare bart model
model = SummaryBartModel()  # model hidden_state = 768

# load, split data (8:2)
data = pd.read_csv("D:\\workspace\\Git_project\\contest\\Korean_Summary\\data\\train_data.csv")
train_data, val_data = train_test_split(data, train_size=0.8, shuffle=True, random_state=1000)
# load dataset, collator
train_datasets = TrainDataset(train_data)
val_datasets = TrainDataset(val_data)
train_collator = TrainDataCollator()
# define Training Arguments
args = TrainingArguments(
    output_dir="D:\\workspace\\Git_project\\contest\\Korean_Summary\\model",
    evaluation_strategy=transformers.EvaluationStrategy.STEPS,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=6,
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
# start train
trainer.train()


# prediction
class PredictDataset(Dataset):
    def __init__(self, predict_data_):
        self.predict_data = predict_data_

    def __len__(self):
        return len(self.predict_data)

    def __getitem__(self, idx: int):
        _, text, _ = self.predict_data.iloc[idx]
        return {"text": text}
class PredictDataCollator(object):
    def __init__(self):
        self.tokenizer = get_kobart_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, sequences):
        # expected >> List of {"text": text}
        texts = [sequence["text"] for sequence in sequences]
        tokened_texts = self.preprocessing(texts)
        return {"input_ids": tokened_texts["input_ids"]}

    def preprocessing(self, texts: List[str]):
        # convert to Serise for Easier preprocessing
        result = pd.Series(texts)
        # remove Legal provisions (ex - 가. ~~~)
        result = result.map(lambda text: text[3:] if re.sub(".\. ", "", text[:3]) == "" else text)
        # remove parenthesis
        result = result.map(lambda text: re.sub("\[[^]]*]|\([^)]*\)", "", text))
        # lower, convert non-korean/english/number to space, strip
        result = result.map(lambda text: re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-z]", " ", text.lower()).strip())
        # tokenize
        tokened_texts = self.tokenizer(result.to_list(), return_tensors="pt").to(self.device)

        return tokened_texts

def predict_sent(sent: str):
    # load trained model
    predict_model = BartModel.from_pretrained("D:\\workspace\\Git_project\\contest\\Korean_Summary\\model")
    tokenizer = get_kobart_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sent = sent[3:] if re.sub(".\. ", "", sent[:3]) == "" else sent
    sent = re.sub("\[[^]]*]|\([^)]*\)", "", sent)
    sent = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣0-9a-z ]", "", sent.lower())
    tokened_sent = tokenizer.encode(sent, return_tensors="pt").to(device)

    predicts = np.argmax(predict_model(tokened_sent), axis=-1)
    return tokenizer.batch_decode(predicts)


def predict_sents(sents: pd.DataFrame):
    predict_model = BartModel.from_pretrained("D:\\workspace\\Git_project\\contest\\Korean_Summary\\model")
    predict_datacollator = PredictDataCollator()
    predict_datasets = PredictDataset(sents)

    predict_args = TrainingArguments(
        output_dir='/content/drive/MyDrive/A_Rai/outputs/Bert',
        evaluation_strategy=transformers.EvaluationStrategy.STEPS,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=4,
        seed=1000,
        load_best_model_at_end=True,
    )

    predict_trainer = Trainer(
        model=predict_model,
        args=predict_args,
        train_dataset=train_datasets,
        eval_dataset=val_datasets,
        data_collator=predict_datacollator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    return np.argmax(predict_trainer.predict(predict_datasets), axis=-1)

