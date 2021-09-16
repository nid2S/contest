import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import TFElectraForSequenceClassification
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from HKT_TopicClassification.Bert_model import get_dataset

RANDOM_SEED = 7777
PRETRAINED_MODEL_NAME = "monologg/koelectra-base-v3-discriminator"

tf.random.set_seed(RANDOM_SEED)

train_dataset, val_dataset = get_dataset.get_data()

# 모델 선언
model = TFElectraForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=7)

# batch_size=16/32, lr(Adam)=5/3/2e-5, epoch=3/4
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics="accuracy",
)

hist = model.fit(
    train_dataset,
    epochs = 4,
    batch_size = 32,
    validation_data = val_dataset,
    callbacks = [EarlyStopping(monitor="val_loss", patience=2),
                 ModelCheckpoint("../model/best_model", monitor="val_accuracy", mode="max", save_best_only=True)]
)

model.save("./Model/TC_model")

# loss, accuracy 그래프
plt.plot(range(1, 5), hist.history["loss"], "r", label="loss")
plt.plot(range(1, 5), hist.history["accuracy"], "b", label="accuracy")
plt.xlabel("epoch")
plt.ylabel("loss/accuracy")
plt.xticks([1, 2, 3, 4])
plt.xlim(0.9, 4.1)
plt.legend()
plt.show()
