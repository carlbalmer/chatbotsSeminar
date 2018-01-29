import os
import random

import tensorflow as tf
import pandas as pd
import re
import time

tf.flags.DEFINE_float(
  "noise_probability", 0.0, "Probability to replace a utterance in the context with noise")

tf.flags.DEFINE_string(
  "input_dir", os.path.abspath("./data"),
  "Input directory containing original CSV data files (default = './data')")

tf.flags.DEFINE_string(
  "output_dir", os.path.abspath("./data/train"),
  "Output directory for noisy training files (default = './data/train')")

FLAGS = tf.flags.FLAGS
TRAIN_PATH = os.path.join(FLAGS.input_dir, "train.csv")
P = FLAGS.noise_probability


if __name__ == "__main__":
    train_data = pd.read_csv(TRAIN_PATH)
    unfolded_context = [re.split(r'(__eou__ __eot__|__eou__)', row)[:-1] for row in train_data['Context']]
    utterances = [split for row in train_data['Context'] for split in re.split(r'__eou__ __eot__|__eou__', row)[:-1]]
    random.shuffle(utterances)

    start_time = time.time()
    out = []
    for row in unfolded_context:
        out.append([utterances.pop() if
                    x != '__eou__' and x != '__eou__ __eot__' # don't replace the delimiters
                    and random.random() <= P != 0.0 # replace the utterances with probability P
                    and row.index(x) != len(row)-2 # don't replace the last utterance
                    else x
                    for x in row])

    train_data['Context'] = ["".join(x) for x in out]

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    train_data.to_csv(os.path.join(FLAGS.output_dir, "train.csv"), index=False)