from transformers import TFMobileBertModel, BertTokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import pandas as pd

MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
# TODO 전처리 생각, None 채우기
# get data

input_layer = tf.keras.Input(input_shape=None)
bert_layer = TFMobileBertModel.from_pretrained(MODEL_NAME)(input_layer)
output_layer1 = tf.keras.layers.Dense(6, activation="softmax")(bert_layer)
output_layer2 = tf.keras.layers.Dense(60, activation="softmax")(bert_layer)

optim = tf.keras.optimizers.Adam(learning_rate=1e-5)
model = tf.keras.Model(inputs=input_layer, outputs=[output_layer1, output_layer2])
model.compile(optimizer=optim, loss="cetegorical_crossentropy", metrics="accuray")
model.fit(x=None,
          y=[None, None],
          epochs=None,
          shuffle=True,
          validation_data=(None, [None, None]),
          callbacks=[EarlyStopping(monitor="val_loss", patience=3),
                     ModelCheckpoint(filepath="./model", monitor='val_accuracy', mode='max', save_best_only=True)])
model.save("./model/best_model")
