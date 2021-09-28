# !pip install git+https://github.com/SKT-AI/KoBART#egg=kobart
import transformers
from transformers import BartModel, Trainer, TrainingArguments, EarlyStoppingCallback
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from datasets import load_metric
import pandas as pd
import numpy as np
import re

# 학습 중 정확도를 계산할 함수
metric = load_metric('rouge')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)

# TODO preprocessing
# 데이터셋, 데이터 Collator선언
class TrainDataset(Dataset):
    def __init__(self, train_data):
        self.train_data = train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # trian_data's columns = {id, text, summary}
        _, text, summary = self.train_data.iloc[idx]
        return {"text": text, "label": summary}
class TrainDataCollator:
    def __init__(self):
        self.tokenizer = get_kobart_tokenizer()

    def __call__(self, sequences):
        # sequences = {"text": text, "label": summary}
        texts = [sequence["text"] for sequence in sequences]
        labels = [sequence["label"] for sequence in sequences]

        tokened_titles = self.tokenizer(texts, return_tensors="pt")
        tokened_labels = self.tokenizer(labels, return_tensors="pt")

        return {"input_ids": tokened_titles["input_ids"], "labels": tokened_labels["input_ids"]}


# 모델 선언
model = BartModel.from_pretrained(get_pytorch_kobart_model())
# 데이터를 가져와 train, val로 나눔(8:2)
data = pd.read_csv("D:\\workspace\\Git_project\\contest\\Korean_Summary\\data\\train_data.csv")
train_data, val_data = train_test_split(data, train_size=0.8, shuffle=True, random_state=1000)
# 나뉜 데이터들을 데이터셋에 넣음, train_collator호출
train_datasets = TrainDataset(train_data)
val_datasets = TrainDataset(val_data)
train_collator = TrainDataCollator()
# 트레이너 설정 정의
args = TrainingArguments(
    output_dir="D:\\workspace\\Git_project\\contest\\Korean_Summary\\model",
    evaluation_strategy=transformers.EvaluationStrategy.STEPS,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=6,
    num_train_epochs=4,
    seed=1000,
    load_best_model_at_end=True,
)
# 트레이너 정의
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_datasets,
    eval_dataset=val_datasets,
    data_collator=train_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
)
# 학습시작
trainer.train()
