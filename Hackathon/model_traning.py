from preprocessing import Preprocesser
from transformers import AutoModel
import tensorflow as tf

p = Preprocesser()
model = AutoModel.from_pretrained(p.PREMODEL_NAME)

# finetuning
