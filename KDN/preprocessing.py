from transformers import BertTokenizerFast
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd

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


def train():
    # 88970
    # get first train data
    t1 = pd.read_csv("./dataset/EmotionalTalkCorpusTraining1.txt", sep="\t", encoding="949")
    drop_columns = ["번호", "연령", "value", "성별", "상황키워드", "신체질환", '시스템응답1', '사람문장2', '시스템응답2', '사람문장3', '시스템응답3']
    t1.drop(drop_columns, axis=1, inplace=True)
    t1.dropna(inplace=True)
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
    b_dict = dict([(row["Emotion"], row["index"]) for _, row in pd.read_csv("./dataset/EmotionB.txt", sep="\t").iterrows()])
    d_dict = dict([(row["Emotion"], row["index"]) for _, row in pd.read_csv("./dataset/EmotionD.txt", sep="\t").iterrows()])

    # preprocessing
    for idx, row in train_data.iterrows():
        # 감정_소분류 | 형에 맞게 변환 후 정수 인코딩
        row["감정_소분류"] = d_dict[row["감정_대분류"]+"_"+row["감정_소분류"]]
        # 감정_대분류 | 정수 인코딩
        row["감정_대분류"] = b_dict[row["감정_대분류"]]

    train_data.to_csv("./dataset/Train.txt", sep="\t", index=False, header=["emotion_b", "emotion_d", "sent"])

def validation():
    # 13983
    # get first validation data
    v1 = pd.read_csv("./dataset/EmotionalTalkCorpusValidation1.txt", sep="\t", encoding="949")
    drop_columns = ["번호", "연령", "value", "성별", "상황키워드", "신체질환", '시스템응답1', '사람문장2', '시스템응답2', '사람문장3', '시스템응답3']
    v1.drop(drop_columns, axis=1, inplace=True)
    v1.dropna(inplace=True)
    # get second validation data
    v2 = pd.read_csv("./dataset/EmotionalTalkCorpusValidation2.txt", sep="\t", encoding="949")
    drop_columns = ["번호", "연령", "성별", "상황키워드", "신체질환", '시스템응답1', '사람문장2', '시스템응답2', '사람문장3', '시스템응답3', '사람문장4', '시스템응답4']
    v2.drop(drop_columns, axis=1, inplace=True)
    v2.dropna(inplace=True)
    # put EachData together | columns = ['감정_대분류', '감정_소분류', '사람문장1']
    val_data = v1.append(v2)
    val_data.drop_duplicates(["사람문장1"], inplace=True)
    val_data = val_data.reset_index().drop(["index"], axis=1)
    # get label_dict
    b_dict = dict(
        [(row["Emotion"], row["index"]) for _, row in pd.read_csv("./dataset/EmotionB.txt", sep="\t").iterrows()])
    d_dict = dict(
        [(row["Emotion"], row["index"]) for _, row in pd.read_csv("./dataset/EmotionD.txt", sep="\t").iterrows()])

    # preprocessing
    for idx, row in val_data.iterrows():
        # 감정_소분류 | 형에 맞게 변환 후 정수 인코딩
        row["감정_소분류"] = d_dict[row["감정_대분류"] + "_" + row["감정_소분류"]]
        # 감정_대분류 | 정수 인코딩
        row["감정_대분류"] = b_dict[row["감정_대분류"]]

    val_data.to_csv("./dataset/Validation.txt", sep="\t", index=False, header=["emotion_b", "emotion_d", "sent"])

def test():
    # 9376
    # get test data
    test_data = pd.read_csv("./dataset/EmotionalTalkCorpusTest.txt", sep="\t", encoding="949")
    drop_columns = ["연령", "성별", "상황키워드", '기계문장1', '사람문장2', '기계문장2', '사람문장3', '기계문장3']
    unnamed_columns = ['Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18']
    # columns = ['감정_대분류', '감정_소분류', '사람문장1']
    test_data.drop(drop_columns + unnamed_columns, axis=1, inplace=True)
    test_data.dropna(inplace=True)
    test_data.drop_duplicates(["사람문장1"], inplace=True)
    test_data = test_data.reset_index().drop(["index"], axis=1)
    # get label_dict
    b_dict = dict([(row["Emotion"], row["index"]) for _, row in pd.read_csv("./dataset/EmotionB.txt", sep="\t").iterrows()])
    d_dict = dict([(row["Emotion"], row["index"]) for _, row in pd.read_csv("./dataset/EmotionD.txt", sep="\t").iterrows()])
    # preprocessing
    for idx, row in test_data.iterrows():
        # 감정_소분류 | 형에 맞게 변환 후 정수 인코딩
        # '당황_고립된(당황한)', '당황_혼란스러운(당황한)', '당황_괴로워하는' 제거
        row["감정_소분류"] = row["감정_소분류"].replace("(당황한)", "")
        row["감정_소분류"] = row["감정_소분류"].replace("괴로워하는", "괴로워 하는")
        try:
            row["감정_소분류"] = d_dict[row["감정_대분류"] + "_" + row["감정_소분류"]]
        except KeyError:
            # '당황_분노', '분노_구역질나는', '분노_스트레스 받는', '분노_희생된', '불안_스트레스받는',
            # '상처_외로운', '상처_충격받은', '슬픔_상처', '슬픔_성가신', '슬픔_열등감', '슬픔_외로운' 의 경우, 해당 열 제거
            test_data.drop(idx, inplace=True)
        # 감정_대분류 | 정수 인코딩
        row["감정_대분류"] = b_dict[row["감정_대분류"]]

    test_data.to_csv("./dataset/Test.txt", sep="\t", index=False, header=["emotion_b", "emotion_d", "sent"])


class datasetGetter:
    def __init__(self):
        self.MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
        self.RANDOM_SEED = 1000
        self.max_len = 100
        self.batch_size = 16
        self.tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)

    # 토큰화, 정수인코딩, 패딩, 텐서화
    def getTrainDataset(self) -> tf.data.Dataset:
        train_data = pd.read_csv("./dataset/Train.txt", sep="\t")
        train_x = self.tokenizer.batch_encode_plus(train_data.sent.to_list(), padding="max_length",
                                                   truncation=True, max_length=self.max_len, return_tensors="tf")
        train_Y1 = to_categorical(train_data["emotion_b"].to_list())
        train_Y2 = to_categorical(train_data["emotion_d"].to_list())

        return tf.data.Dataset.from_tensors((train_x, [train_Y1, train_Y2])).shuffle(1000, seed=self.RANDOM_SEED).batch(self.batch_size)

    def getValidationDataset(self) -> tf.data.Dataset:
        val_data = pd.read_csv("./dataset/Validation.txt", sep="\t")
        val_x = self.tokenizer.batch_encode_plus(val_data.sent.to_list(), padidng="max_lenth",
                                                 truncation=True, max_length=self.max_len, return_tensors="tf")

        val_Y1 = to_categorical(val_data["emotion_b"].to_list())
        val_Y2 = to_categorical(val_data["emotion_d"].to_list())

        return tf.data.Dataset.from_tensors((val_x, [val_Y1, val_Y2])).shuffle(1000, seed=self.RANDOM_SEED).batch(self.batch_size)

    def getTestDataset(self, sent: str = None):
        """ if sent is None, return basic Test Dataset."""
        # data is sentence. case of wanna put document in model, seperate to sentence > get average sentences' vector.
        if sent is None:
            test_data = pd.read_csv("./dataset/Test.txt", sep="\t")
            test_x = self.tokenizer.batch_encode_plus(test_data.sent.to_list(), padidng="max_lenth",
                                                      truncation=True, max_length=self.max_len, return_tensors="tf")
            return tf.data.Dataset.from_tensors((test_x, )).shuffle(1000, seed=self.RANDOM_SEED).batch(self.batch_size)

        else:
            return self.tokenizer(sent)

    def getMaxLen(self) -> int:
        """ get train_data's maxlen(92). model's maxlen is set 100"""
        # 92. setting 100.
        train_data = pd.read_csv("./dataset/Train.txt", sep="\t")
        train_x = self.tokenizer.batch_encode_plus(train_data.sent.to_list(), padding=True, truncation=True)
        return len(train_x[0])
