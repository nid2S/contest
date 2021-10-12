from preprocessing import datasetGetter
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import TFMobileBertModel
import matplotlib.pyplot as plt
import tensorflow as tf

MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
dg = datasetGetter()

# make model
# batch_size=16/32, lr(Adam)=5/3/2e-5, epoch=3/4
bert_layer = TFMobileBertModel.from_pretrained(MODEL_NAME)
output_layer1 = tf.keras.layers.Dense(6, activation="softmax")(bert_layer)
output_layer2 = tf.keras.layers.Dense(60, activation="softmax")(bert_layer)
model = tf.keras.Model(inputs=bert_layer, outputs=[output_layer1, output_layer2])

optim = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optim, loss="cetegorical_crossentropy", metrics="accuray")
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
