import os
import re
import pandas as pd
from string import punctuation
from hgtk.text import decompose


def pad_sequences(encoded_vectors, maxlen: int) -> list[list[float]]:
    padded_text = []
    for vector in encoded_vectors:
        padded_text.append(vector+([0]*(maxlen-len(vector))))
    return padded_text


class Preprocesser:
    def __init__(self):
        self.pad_len = 0
        self.vocab_size = 0
        self.train_origin = pd.DataFrame()
        self.train_data = []
        self.test_origin = pd.DataFrame()
        self.test_data = []
        self.word_to_index = dict()
        self.index_to_topic = dict()
        self._get_data()

    def _get_data(self):
        if os.path.isdir("./HTK_TopicClassification"):
            os.chdir("./HKT_TopicClassification")
        if not (os.path.isfile("./dataset/train_data.csv") and os.path.isfile("./dataset/test_data.csv")
                and os.path.isfile("./dataset/topic_dict.csv")):
            raise FileNotFoundError("Essential data file is NOT existing.")

        self.train_origin = pd.read_csv("./dataset/train_data.csv")  # data_num = 45654
        self.test_origin = pd.read_csv("./dataset/test_data.csv")  # data_num = 9131
        self.test_origin["topic_idx"] = pd.read_csv("./dataset/sample_submission.csv")["topic_idx"]  # add label
        self.index_to_topic = dict([(row["topic_idx"], row["topic"]) for (_, row) in
                                    pd.read_csv("./dataset/topic_dict.csv").iterrows()])
        print("loaded data")

        # make word_vocab
        word_vocab = set()
        for i, title in self.train_origin["title"].items():
            title = decompose(re.sub(r'/W', r' ', title.lower()), compose_code="")
            title = re.sub(f"[{punctuation}]", "", title)
            for char in title:
                word_vocab.add(char)
        self.word_to_index = dict([(char, i+1) for i, char in enumerate(sorted(list(word_vocab)))])
        self.word_to_index["OOV"] = len(self.word_to_index) + 1
        self.vocab_size = len(self.word_to_index) + 1  # 91
        print("made word_vocab")

        # train preprocessing
        train_encoded = []
        for i, title in self.train_origin["title"].items():
            title = decompose(re.sub(r'/W', r' ', title.lower()), compose_code="")
            title = re.sub(f"[{punctuation}]", "", title)
            encoded_text = []
            for char in [char for char in title]:
                try:
                    encoded_text.append(self.word_to_index[char])
                except KeyError:
                    encoded_text.append(self.word_to_index["OOV"])
            train_encoded.append(encoded_text)
        print("train_preprocessed")

        # test preprocessing
        test_encoded = []
        for i, title in self.test_origin["title"].items():
            title = decompose(re.sub(r'/W', r' ', title.lower()), compose_code="")
            title = re.sub(f"[{punctuation}]", "", title)
            encoded_text = []
            for char in [char for char in title]:
                try:
                    encoded_text.append(self.word_to_index[char])
                except KeyError:
                    encoded_text.append(self.word_to_index["OOV"])
            test_encoded.append(encoded_text)
        print("test_preprocessed")

        # get pad_len
        self.pad_len = max([len(encoded_title) for encoded_title in train_encoded])  # 96

        # train padding, one-hot encoding
        pad = pad_sequences(train_encoded, maxlen=self.pad_len)
        self.train_data = self.to_categorical(pad)
        print("train_padding, one-hot encoded")

        # train padding, one-hot encoding
        pad = pad_sequences(test_encoded, maxlen=self.pad_len)
        self.test_data = self.to_categorical(pad)
        print("complete to dataLoad")

    def preprocessing(self, text: str) -> list[list[list[int]]]:
        # return : [[char(one-hot encoded)]]
        # 소문자화, 비문자(한자는 남음)제거, 구두점 제거, 자모단위로 분리 후 리스트로 분리
        # 정수인코딩, 패딩, 원-핫-인코딩
        text = decompose(re.sub(r'/W', r' ', text.lower()), compose_code="")
        text = re.sub(f"[{punctuation}]", "", text)
        text = [char for char in text]

        encoded_text = []
        for char in text:
            try:
                encoded_text.append(self.word_to_index[char])
            except KeyError:
                encoded_text.append(self.word_to_index["OOV"])

        encoded_text = pad_sequences([encoded_text], maxlen=self.pad_len)
        encoded_text = self.to_categorical(encoded_text)

        return encoded_text

    def to_categorical(self, padded_text) -> list[list[list[int]]]:
        encoded_vector = []
        for sent in padded_text:
            encoded_sent = []
            for char_idx in sent:
                base_vec = [0]*self.vocab_size
                base_vec[char_idx] = 1
                encoded_sent.append(base_vec)
            encoded_vector.append(encoded_sent)
        return encoded_vector
