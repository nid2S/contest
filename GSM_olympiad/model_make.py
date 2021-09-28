from preprocessing import Preprocesser
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 시간효율성 | 모델 성능(전처리)
RANDOM_SEED = 7777
tf.random.set_seed(RANDOM_SEED)

p = Preprocesser()

train_data, val_data, train_label, val_label = \
    train_test_split(p.train_data, p.train_origin["topic_idx"].values.tolist(),  # non-one-hot vector > error(shape(ndim:1))
                     train_size=0.8, shuffle=True, random_state=RANDOM_SEED)
test_data = p.test_data
test_label = p.test_origin["topic_idx"].values.tolist()


def CNN(batch_size: int, kernel_size: int, dr: float, padding: str, use_bias: bool, verbose: int = 1):
    """padding : valid(padding), same(non-padding)
        kernel_size=2, stride=1, padding="valid", use_bias=False, pool_size=2,
        batch_size=128, dropout_rate=0.3, learning_rate=0.1"""
    # Embedding - required non-one-hot encoded vector(?) | units setting/fit | KFold
    # model_CNN.add(layers.Embedding(input_shape=(p.pad_len, p.vocab_size), output_dim=64))
    model_CNN = tf.keras.Sequential()
    model_CNN.add(layers.InputLayer(input_shape=(p.pad_len, p.vocab_size), batch_size=batch_size))
    model_CNN.add(layers.Conv1D(128, kernel_size=kernel_size, padding=padding, use_bias=use_bias))
    model_CNN.add(layers.MaxPooling1D(padding=padding))
    model_CNN.add(layers.Dropout(rate=dr))

    model_CNN.add(layers.Dense(64))
    model_CNN.add(layers.Conv1D(32, kernel_size=kernel_size, padding=padding, use_bias=use_bias))
    model_CNN.add(layers.MaxPooling1D(padding=padding))
    model_CNN.add(layers.Dropout(rate=dr))

    model_CNN.add(layers.Dense(16, activation="relu"))
    model_CNN.add(layers.Dense(7, activation="softmax"))

    optim = tf.keras.optimizers.Adam(learning_rate=0.1)
    model_CNN.compile(optimizer=optim, loss="sparse_categorical_crossentropy", metrics="accuracy")

    model_CNN.fit(epochs=500, x=train_data, y=train_label, verbose=verbose, validation_data=(val_data, val_label),
                  callbacks={EarlyStopping(monitor="val_loss", patience=2)})

    return model_CNN.evaluate(test_data, test_label, batch_size=batch_size)


def RNN(batch_size: int, use_bias: bool, use_attention: bool, use_LSTM: bool, bidirectional: bool, verbose: int = 1):
    """ if use_LSTM is False, use GRU.
    batch_size=128, learning_rate=0.1, dropout_rate=0, use_bias=True, use_attention=False,
    use_LSTM=True, bidirectional=False"""

    input_tensor = tf.keras.Input(shape=(p.pad_len, p.vocab_size), batch_size=batch_size)
    # x = layers.Embedding(input_dim=p.pad_len, output_dim=128)(input_tensor)  # required non-ont-hot-vector
    if use_LSTM:
        RNN_c = layers.LSTM(64, activation="tanh", use_bias=use_bias)
    else:
        RNN_c = layers.GRU(64, activation="tanh", use_bias=use_bias)
    if bidirectional:
        RNN_c = layers.Bidirectional(RNN_c)

    x = RNN_c(input_tensor)

    if use_attention:
        x = layers.Attention(x)

    x = layers.Dense(32, activation="relu")(x)
    Y = layers.Dense(7, activation="softmax")(x)

    model_RNN = tf.keras.models.Model(input_tensor, Y)

    optim = tf.keras.optimizers.Adam(learning_rate=0.1)
    model_RNN.compile(optimizer=optim, loss="sparse_categorical_crossentropy", metrics="accuracy")

    model_RNN.fit(epochs=500, x=train_data, y=train_label, verbose=verbose, validation_data=(val_data, val_label),
                  callbacks={EarlyStopping(monitor="val_loss", patience=2)})

    return model_RNN.evaluate(test_data, test_label, batch_size=batch_size)


CNN_hyper_params = {
    "batch_size": [64, 128, 256],
    "kernel_size": [2, 3, 4],
    "dr": [0.2, 0.3, 0.4],
    "padding": ["vaild", "same"],
    "use_bias": [True, False],
}
RNN_hyper_params = {
    "batch_size": [64, 128, 256],
    "use_bias": [True, False],
    "use_LSTM": [True, False],
    "use_attention": [True, False],
    "bidirectional": [True, False],
}

best_accuracy = 0
best_param = {"CNN": [], "RNN": []}
result = pd.DataFrame(columns=["model_type", "time", "accuracy", "hyper_params"])
# CNN search
for batch_size in CNN_hyper_params["batch_size"]:
    for kernel_size in CNN_hyper_params["kernel_size"]:
        for dr in CNN_hyper_params["dr"]:
            for padding in CNN_hyper_params["padding"]:
                for use_bias in CNN_hyper_params["use_bias"]:
                    hyper_params = {"batch_size": batch_size, "kernel_size": kernel_size, "dr": dr, "padding": padding, "use_bias": use_bias}
                    time_ = time.time()
                    (_, accuracy) = CNN(**hyper_params)
                    time_ = time.time() - time_
                    result.append({"model_type": "CNN", "accuracy": accuracy, "time": time_, "hyper_params": hyper_params})
                    if accuracy >= best_accuracy:
                        best_accuracy = accuracy
                        best_param["CNN"] = hyper_params
# RNN search
for batch_size in RNN_hyper_params["batch_size"]:
    for use_bias in RNN_hyper_params["use_bias"]:
        for use_LSTM in RNN_hyper_params["use_LSTM"]:
            for use_attention in RNN_hyper_params["use_attention"]:
                for bidirectional in RNN_hyper_params["bidirectional"]:
                    hyper_params = {"batch_size": batch_size, "use_bias": use_bias, "use_LSTM": use_LSTM, "use_attention": use_attention, "bidirectional": bidirectional}
                    time_ = time.time()
                    (_, accuracy) = RNN(**hyper_params)
                    time_ = time.time() - time_
                    result.append({"model_type": "RNN", "accuracy": accuracy, "time": time_, "hyper_params": hyper_params})
                    if accuracy >= best_accuracy:
                        best_accuracy = accuracy
                        best_param["RNN"] = hyper_params

result.sort_values(["accuracy"], inplace=True, ascending=False)

with open("model/result.csv", "w+") as f, open("model/best_result.csv", "w+") as f1:
    f.write(result.__str__())
    f1.write(f1.__str__())
