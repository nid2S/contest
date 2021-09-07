import os
import re
import pandas as pd
from hgtk.text import decompose
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class Preprocesser:
    def __init__(self):
        self.pad_len = 0
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.word_to_index = dict()
        self.index_to_topic = dict([(row["topic_idx"], row["topic"]) for (idx, row) in
                                    pd.read_csv("./dataset/topic_dict.csv").iterrows()])
        self._get_data()

    def _get_data(self):
        if not (os.path.isfile("./dataset/train_data.csv") and os.path.isfile("./dataset/test_data.csv")
                and os.path.isfile("./dataset/topic_dict.csv")):
            raise FileNotFoundError("Essential data file is NOT exsiting.")

        self.train = pd.read_csv("./dataset/train_data.csv")
        self.test = pd.read_csv("./dataset/test_data.csv")

        # make word_vocab + train preprocessing
        word_vocab = set()
        self.train["encoded_title"] = []
        for i, title in self.train["title"].items():
            title = decompose(re.sub(r'/W', r' ', title.lower()), compose_code="")
            self.train["encoded_title"].values[i] = [char for char in title]
            for char in title:
                word_vocab.add(char)

        # test preprocessing
        self.test["encoded_title"] = []
        for i, title in self.test["title"].items():
            title = decompose(re.sub(r'/W', r' ', title.lower()), compose_code="")
            self.test["encoded_title"].values[i] = [char for char in title]

        # set word_vocab
        self.word_to_index = dict([(char, i+1) for i, char in enumerate(sorted(list(word_vocab)))])
        self.word_to_index["OOV"] = len(self.word_to_index)+1


        # get max_len, padding, to_categorical
        self.max_len = max([len(encoded_title) for encoded_title in self.train["encoded_title"].values])
        self.train["encoded_title"].values = pad_sequences(self.train["encoded_title"].values, maxlen=self.pad_len, padding="post")
        self.train["encoded_title"].values = to_categorical(self.train["encoded_title"].values)
        self.test["encoded_title"].values = pad_sequences(self.test["encoded_title"].values, maxlen=self.pad_len, padding="post")
        self.test["encoded_title"].values = to_categorical(self.test["encoded_title"].values)

    def preprocessing(self, text: str) -> list[list[list[str]]]:
        # [sent > char(one-hot encoded)]
        # 소문자화, 비문자(한자는 남음)제거, 자모단위로 분리 후 리스트로 분리
        text = decompose(re.sub(r'/W', r' ', text.lower()), compose_code="")
        text = [char for char in text]

        encoded_text = []
        for char in text:
            try:
                encoded_text.append(self.word_to_index[char])
            except KeyError:
                encoded_text.append(self.word_to_index["OOV"])

        encoded_text = pad_sequences([encoded_text], maxlen=self.pad_len, padding="post")
        encoded_text = to_categorical(encoded_text)

        return encoded_text
