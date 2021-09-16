import tensorflow as tf
import pandas as pd

from transformers import ElectraTokenizerFast, TFElectraForSequenceClassification
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

RANDOM_SEED = 7777
PRETRAINED_MODEL_NAME = "monologg/koelectra-base-v3-discriminator"


def get_data() -> (tf.data.Dataset, tf.data.Dataset):
    train_data = pd.read_csv("../dataset/train_data.csv")
    tokenizer = ElectraTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME)
    x_data = []
    Y_data = []
    # train데이터를 받아옴
    for _, (_, title, topic_idx) in train_data.iterrows():
        x_data.append(title)
        Y_data.append(topic_idx)
    # train, eval 분리 | 허용되는 입력이 list, ndarray, pd.DataFrame, scipy-sparse matrices 뿐이라 먼저 진행됨.
    x_train, x_val, Y_train, Y_val = train_test_split(x_data, Y_data, train_size=0.8, shuffle=True,
                                                      random_state=RANDOM_SEED)
    # train 토큰화 + 정수인코딩 + 패딩
    x_train = tokenizer.batch_encode_plus(x_train, padding=True, truncation=True)
    x_val = tokenizer.batch_encode_plus(x_val, padding=True, truncation=True)
    # train 텐서화
    encoded_x_train = dict()
    for key, values in x_train.items():
        encoded_x_train[key] = tf.convert_to_tensor(values)
    # val 텐서화
    encoded_x_val = dict()
    for key, values in x_val.items():
        encoded_x_val[key] = tf.convert_to_tensor(values)
    # label 원핫인코딩 + 텐서화
    Y_train = to_categorical(Y_train)
    Y_val = to_categorical(Y_val)
    # 데이터셋 제작
    train_dataset = tf.data.Dataset.from_tensor_slices((encoded_x_train, Y_train)).shuffle(1000).batch(16)
    val_dataset = tf.data.Dataset.from_tensor_slices((encoded_x_val, Y_val)).shuffle(1000).batch(16)

    return train_dataset, val_dataset
