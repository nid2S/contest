from transformers import TFMobileBertModel, BertTokenizerFast
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import pandas as pd

# TODO 전처리 생각, None 채우기
# get data

input_layer = tf.keras.layers.InputLayer(batch_input_shape=None)
bert_layer = TFMobileBertModel.from_pretrained("monologg/koelectra-base-v3-discriminator")(input_layer)
output_layer1 = tf.keras.layers.Dense(6, activation="softmax")(bert_layer)
output_layer2 = tf.keras.layers.Dense(60, activation="softmax")(bert_layer)

optim = tf.keras.optimizers.Adam(learning_rate=0.001)
model = tf.keras.Model(inputs=input_layer, outputs=[output_layer1, output_layer2])
model.compile(optimizer=optim, loss="sparse_cetegorical_crossentropy", metrics="accuray")
model.fit(x=None,
          y=[None, None],
          epochs=None,
          shuffle=True,
          validation_data=(None, [None, None]),
          callbacks=[EarlyStopping(monitor="val_loss", patience=3),
                     ModelCheckpoint(filepath="./model", monitor='val_accuracy', mode='max', save_best_only=True)])
model.save("./model/best_model")
