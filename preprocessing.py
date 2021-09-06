import tensorflow as tf
import pandas as pd
import re
import os
from hgtk.text import decompose
from sklearn.model_selection import train_test_split


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

    def preprocessing(self, text: str) -> list[list[list[str]]]:
        # sent > word > char(one-hot encoded)
        # 소문자화, 비문자(한자는 남음)제거, 자모단위로 분리 후 리스트로 분리
        text = decompose(re.sub(r'/W', r' ', text.lower()), "")
        text = [[char for char in word] for word in text.split()]

        encoded_text = []
        for word in text:
            encoded_word = []
            for char in word:
                try:
                    encoded_word.append(self.word_to_index[char])
                except KeyError:
                    encoded_word.append(self.word_to_index["OOV"])
            encoded_text.append(encoded_word)

        encoded_text = tf.keras.preprocessing.sequence.pad_sequences(encoded_text, maxlen=self.pad_len)
        encoded_text = tf.keras.utils.to_categorical(encoded_text)

        return encoded_text


