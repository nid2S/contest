import pandas as pd
import re

def Train():
    # Train
    # get first train data
    t1 = pd.read_csv("./dataset/EmotionalTalkCorpusTraining1.txt", sep="\t", encoding="949")
    drop_columns = ["번호", "연령", "value", "성별", "상황키워드", "신체질환", '시스템응답1', '사람문장2', '시스템응답2', '사람문장3', '시스템응답3']
    t1.drop(drop_columns, axis=1, inplace=True)
    # get second train data
    t2 = pd.read_csv("./dataset/EmotionalTalkCorpusTraining2.txt", sep="\t", encoding="949")
    drop_columns = ["번호", "연령", "성별", "상황키워드", "신체질환", '시스템응답1', '사람문장2', '시스템응답2', '사람문장3', '시스템응답3', '사람문장4', '시스템응답4']
    t2.drop(drop_columns, axis=1, inplace=True)
    t2.dropna(inplace=True)
    # put EachData together | columns = ['감정_대분류', '감정_소분류', '사람문장1']
    train_data = t1.append(t2)
    train_data.drop_duplicates(["사람문장1"], inplace=True)
    train_data = train_data.reset_index().drop(["index"], axis=1)
    # get label_dict
    b_dict = dict([(row["Emotion"], row["index"]) for _, row in pd.read_csv("./contest/KDN/dataset/EmotionB.txt", sep="\t").iterrows()])
    d_dict = dict([(row["Emotion"], row["index"]) for _, row in pd.read_csv("./contest/KDN/dataset/EmotionD.txt", sep="\t").iterrows()])

    # preprocessing
    for idx, row in train_data.iterrows():
        # 감정 대분류
        pass
        # 감정 소분류
        row["감정_소분류"] = row["감벙_대분류"] + "_" + row["감정대분류"]
        # 사람문장1
        pass





def make_label_dict():
    t1 = pd.read_csv("./dataset/EmotionalTalkCorpusTraining1.txt", sep="\t", encoding="949")
    drop_columns = ["번호", "연령", "value", "성별", "상황키워드", "신체질환", '시스템응답1', '사람문장2', '시스템응답2', '사람문장3', '시스템응답3']
    t1.drop(drop_columns, axis=1, inplace=True)
    # get second train data
    t2 = pd.read_csv("./dataset/EmotionalTalkCorpusTraining2.txt", sep="\t", encoding="949")
    drop_columns = ["번호", "연령", "성별", "상황키워드", "신체질환", '시스템응답1', '사람문장2', '시스템응답2', '사람문장3', '시스템응답3', '사람문장4',
                    '시스템응답4']
    t2.drop(drop_columns, axis=1, inplace=True)
    t2.dropna(inplace=True)
    # put EachData together | columns = ['감정_대분류', '감정_소분류', '사람문장1']
    train_data = t1.append(t2)
    train_data.drop_duplicates(["사람문장1"], inplace=True)
    train_data = train_data.reset_index().drop(["index"], axis=1)

    b_list = []
    d_list = []
    for b, d in zip(train_data["감정_대분류"], train_data["감정_소분류"]):
        if b not in b_list:
            b_list.append(b)
        if b + "_" + d not in d_list:
            d_list.append(b + "_" + d)

    b_list.sort()
    d_list.sort()

    b = pd.DataFrame()
    d = pd.DataFrame()
    b["index"] = pd.Series(range(len(b_list)))
    b["Emotion"] = pd.Series(b_list)
    d["index"] = pd.Series(range(len(d_list)))
    d["Emotion"] = pd.Series(d_list)

    b.to_csv("./dataset/EmotionB.txt", sep="\t", index=False)
    d.to_csv("./dataset/EmotionD.txt", sep="\t", index=False)
