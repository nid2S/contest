from transformers import AutoTokenizer, AutoConfig
import tensorflow as tf
import pandas as pd
import json

def make_dataset():
    file_name = ""

    raw_data = json.load(open('./data/raw_data/'+file_name+".json", "r+", encoding="utf-8"))
    # raw_data = open('./data/raw_data/'+file_name+".txt", "r+", encoding="utf-8").read()


class Preprocesser:
    def __init__(self):
        self.RANDOM_SEED = 10
        # HyperParam
        self.batch_size = 16
        self.embedding_dim = 128
        self.input_dim = 0
        self.output_dim = 0
        # data
        self.data_num = 0
        self.PREMODEL_NAME = ""
        self.trainData = pd.read_csv("./data/train.txt", sep="\t", names=[])
        self.validationData = pd.read_csv("./data/validation.txt", sep="\t", names=[])
        self.testData = pd.read_csv("./data/test.txt", sep="\t", names=[])
        # tokenizers
        self.HFtokenizer = AutoTokenizer.from_pretrained(self.PREMODEL_NAME)

    def getTrainData(self):
        pass

    def getValidationData(self):
        pass

    def getTestData(self):
        pass

    def id2Token(self, ids: list[int]):
        pass

    def preprocessing(self, text: str):
        pass


