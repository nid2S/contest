from preprocessing import datasetGetter
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import TFMobileBertModel
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
dg = datasetGetter()

# make model
# batch_size=16/32, lr(Adam)=5/3/2e-5, epoch=3/4

class SentenceClassifier(tf.keras.Model):
    def __init__(self):
        super(SentenceClassifier, self).__init__()
        self.bert_layer = TFMobileBertModel.from_pretrained(MODEL_NAME)
        self.dense_layer = tf.keras.layers.Dense(512, activation="gelu")
        self.output_layer1 = tf.keras.layers.Dense(6, activation="softmax")
        self.output_layer2 = tf.keras.layers.Dense(60, activation="softmax")

    def call(self, x, *args, **kwargs):
        x = self.bert_layer(x)
        # x = x.last_hidden_state
        # x = tf.convert_to_tensor(np.mean(x, axis=1))
        x = x.pooler_output
        x = self.dense_layer(x)
        y1 = self.output_layer1(x)
        y2 = self.output_layer2(x)

        return y1, y2


model = SentenceClassifier()

optim = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optim, loss="categorical_crossentropy", metrics="accuracy")
hist = model.fit(dg.getTrainDataset(),
                 batch_size=dg.batch_size,
                 epochs=4,
                 shuffle=True,
                 validation_data=dg.getValidationDataset(),
                 callbacks=[EarlyStopping(monitor="val_loss", patience=3),
                            ModelCheckpoint(filepath="./model/best_model", monitor='val_accuracy', mode='max', save_best_only=True)])
model.save("./model/last_model")

# show history
plt.plot(range(1, 5), hist.history["loss"], "r", label="loss")
plt.plot(range(1, 5), hist.history["accuracy"], "b", label="accuracy")
plt.xlabel("epoch")
plt.ylabel("loss/accuracy")
plt.xticks([1, 2, 3, 4])
plt.xlim(0.9, 4.1)
plt.legend()
plt.show()
