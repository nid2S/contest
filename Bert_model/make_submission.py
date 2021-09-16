import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import ElectraTokenizerFast


# make submission
model = tf.keras.models.load_model("../model/TC_model")
test_data = pd.read_csv("../dataset/test_data.csv")
tokenizer = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")  # PRETRAINED_MODEL_NAME

# test데이터를 받아옴
index_list = []
test_x = []
for i, row in test_data.iterrows():
    index_list.append(row["index"])
    test_x.append(row["title"])

# test 토큰화 + 정수인코딩 + 패딩
encoded_test_data = tokenizer.batch_encode_plus(test_x, padding="max", truncation=True, max_length=30, return_tensors="tf")
# 예측 결과 도출
prediction = np.argmax(model.predict(encoded_test_data["input_ids"]).logits, axis=1)
# 인덱스와 함께 데이터프레임으로 제작 후 csv파일로 저장
df = pd.DataFrame()
df["index"] = pd.Series(index_list)
df["topic_idx"] = pd.Series(prediction)
df.to_csv("./submission.csv", index=False, header=False, columns=["index", "topic_idx"])
