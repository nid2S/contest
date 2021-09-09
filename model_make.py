from preprocessing import Preprocesser
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 7777
tf.random.set_seed(RANDOM_SEED)

p = Preprocesser()

train_data = p.train_data
train_label = p.train_origin["topic_idx"].values
test_data = p.test_data
test_label = p.test_origin["topic_idx"].values


def CNN(batch_size: int, kernel_size: int, stride: int, activation: str, padding: str, use_bias: bool,
        pool_size: int, dropout_rate: float, learning_rate: float, epoch: int = 500, verbose: int = 1):
    """padding : valid(padding), same(non-padding)
        kernel_size=2, stride=1, activation="relu", padding="valid", use_bias=False, pool_size=2,
        batch_size=128, dropout_rate=0.3, learning_rate=0.1"""
    # Embedding - required non-one-hot encoded vector(?) | unit(pram) setting/fit
    # model_CNN.add(layers.Embedding(input_shape=(p.pad_len, p.vocab_size), output_dim=64))
    model_CNN = tf.keras.Sequential()
    model_CNN.add(layers.InputLayer(input_shape=(p.pad_len, p.vocab_size), batch_size=batch_size))
    model_CNN.add(layers.Conv1D(128, kernel_size=kernel_size, strides=stride, activation=activation, padding=padding, use_bias=use_bias))
    model_CNN.add(layers.MaxPooling1D(pool_size=pool_size, padding=padding))
    model_CNN.add(layers.Dropout(rate=dropout_rate))

    model_CNN.add(layers.Dense(64, activation=activation))
    model_CNN.add(layers.Conv1D(32, kernel_size=kernel_size, strides=stride, activation=activation, padding=padding, use_bias=use_bias))
    model_CNN.add(layers.MaxPooling1D(pool_size=pool_size, padding=padding))
    model_CNN.add(layers.Dropout(rate=dropout_rate))

    model_CNN.add(layers.Dense(16, activation=activation))
    model_CNN.add(layers.Dense(7, activation="softmax"))

    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model_CNN.compile(optimizer=optim, loss="sparse_categorical_crossentropy", metrics="accuracy")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)  # ?
    for train_idx, val_idx in skf.split(train_data, train_label):
        train_t = train_data[train_idx[:]]
        label_t = train_label[train_idx[:]]
        train_val = train_data[val_idx[:]]
        label_val = train_label[val_idx[:]]

        model_CNN.fit(epochs=epoch, x=train_t, y=label_t, verbose=verbose, validation_data=(train_val, label_val),
                      callbacks={EarlyStopping(monitor="val_loss", patience=2)})

    return model_CNN.evaluate(test_data, test_label, batch_size=batch_size)


def RNN(batch_size: int, learning_rate: float, dropout_rate: float, use_bias: bool,
        rnn_act: str, dense_act: str, use_LSTM: bool, bidirectional: bool, epoch: int = 500, verbose: int = 1):
    """ if use_LSTM is False, use GRU.
    batch_size=128, learning_rate=0.1, dropout_rate=0, use_bias=True,
    rnn_act="tanh", dense_act="relu", use_LSTM=True, bidirectional=False"""
    x = layers.Input(shape=(p.pad_len, p.vocab_size), batch_size=batch_size)

    if use_LSTM:
        RNN_c = layers.LSTM(128, activation=rnn_act, recurrent_activation="sigmoid", use_bias=use_bias,
                            dropout=dropout_rate, recurrent_dropout=0.)
    else:
        RNN_c = layers.GRU(128, activation=rnn_act, recurrent_activation="sigmoid", use_bias=use_bias,
                           dropout=dropout_rate, recurrent_dropout=0.)
    if bidirectional:
        RNN_c = layers.Bidirectional(RNN_c)

    x = RNN_c(x)

    x = layers.Dense(64, activation=dense_act)(x)
    Y = layers.Dense(7, activation="softmax")(x)

    model_RNN = tf.keras.models.Model(x, Y)

    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model_RNN.compile(optimizer=optim, loss="sparse_categorical_crossentropy", metrics="accuracy")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, val_idx in skf.split(train_data, train_label):
        train_t = train_data[train_idx[:]]
        label_t = train_label[train_idx[:]]
        train_val = train_data[val_idx[:]]
        label_val = train_label[val_idx[:]]

        model_RNN.fit(epochs=epoch, x=train_t, y=label_t, verbose=verbose, validation_data=(train_val, label_val),
                      callbacks={EarlyStopping(monitor="val_loss", patience=2)})

    return model_RNN.evaluate(test_data, test_label, batch_size=batch_size)




