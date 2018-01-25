import os
import random

import pandas as pd
import re

TRAIN_PATH = os.path.join('./data', "train.csv")
P = 0.1

def get_list_of_utterances(dataframe):
    utterances = []
    for row in dataframe['Context']:
        utterances.extend(re.split(r'__eou__ __eot__|__eou__', row)[:-1])
    return utterances


def noise_context(context, replacements):
    context_by_utterances = re.split(r'(__eou__ __eot__|__eou__)', context)[:-1]
    for i in range(len(context_by_utterances)-2):
        if context_by_utterances[i] != '__eou__' and context_by_utterances[i] != '__eou__ __eot__':
            if random.random() <= P:
                if random.random() <= 0.1:
                    context_by_utterances[i] = ''
                else:
                    context_by_utterances[i] = replacements[random.randint(0, len(replacements)-1)]

if __name__ == "__main__":
    for P in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]:
        train_data = pd.read_csv(TRAIN_PATH)
        utterances = get_list_of_utterances(train_data)
        for row in train_data['Context']:
            noise_context(row, utterances)
        train_data.to_csv(os.path.join('./data', "train_p" + str(P) + ".csv"))