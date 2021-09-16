from transformers import ElectraTokenizerFast
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import tkinter
import re

# topic_dict를 가져옴
topic_dict = pd.read_csv("../dataset/topic_dict.csv")
topic_dict = dict([(row["topic_idx"], row["topic"]) for _, row in topic_dict.iterrows()])
# tokinezer를 가져옴
tokenizer = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")
# model을 가져옴
model = load_model("../model/TC_model")

# 메인 화면(윈도우)설정
win = tkinter.Tk()
win.title("topic classification")
win.geometry("800x450+100+50")
win.option_add("*Font", "맑은고딕 20")
# 메인 타이틀 설정
tkinter.Label(win, text="뉴스기사 토픽분류", pady=10).pack(side="top")
# 최 하단 분석버튼 설정
tkinter.Button(win, text="분석", command=lambda: prediction()).pack(side="bottom", pady="7") # command, text
# 분석 결과가 나올 라벨 설정
lb = tkinter.Label(win, text="", width=30)
lb.pack(side="bottom", pady="10")
# 분석할 문장(뉴스토픽)을 입력할 엔트리 설정
entry = tkinter.Entry(win,  width=50)
entry.insert(0, "여기에 뉴스 제목을 입력하세요.")
entry.pack(side="bottom", pady="10")


# 문장분석 함수 정의
def prediction():
    text = entry.get()
    entry.delete(0, len(text))
    if len(re.sub("/W", "", text)) < 2:
        lb.config(text="잘못된 입력입니다.")
    else:
        encoded_text = tokenizer(text)["input_ids"]  # remoce []?
        prediction = model.predict(encoded_text["input_ids"]).logits  # input other data? (dataset, tensor)
        prediction = np.argmax(prediction)
        lb.config(text="주제 분석결과 : " + topic_dict[prediction])

# 메인 루프 실행
win.mainloop()
win.destroy()
