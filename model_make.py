from preprocessing import Preprocesser
import tensorflow as tf
import pandas as pd

p = Preprocesser()

train_data = p.train["encoded_title"]


