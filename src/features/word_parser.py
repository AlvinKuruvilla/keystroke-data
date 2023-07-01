import os
import pandas as pd
import spacy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from features.linguistic_features import estimate_shannon_entropy, plot_multi_bar_graph


def plot_heatmap(matrix, title=None):
    sns.heatmap(matrix, linewidth=0.5).set_title(title)
    plt.savefig(title)


def get_user_by_platform(user_id, platform_id):
    df = pd.read_csv(
        os.path.join(os.getcwd(), "cleaned.csv"),
        dtype={
            "key": str,
            "press_time": np.float64,
            "release_time": np.float64,
            "platform_id": np.uint8,
            "session_id": np.uint8,
            "user_ids": np.uint8,
        },
    )
    return df[(df["user_ids"] == user_id) & (df["platform_id"] == platform_id)]


# https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
def remove_invalid_keystrokes(df):
    # A helper function that takes as input a dataframe, and return a new
    # dataframe no longer containing rows with the string "<0>"
    return df.loc[df["key"] != "<0>"]


# This cell contains helper functions for word level feature generation.
# Most of these functions are currently unused but we may want to keep them
# around for future feature extraction
# The main function of note is the get_words() function which parses the raw
# sentences we generate from users' typing logs and extract words using the
# the spacy tokenizer


def clean_letters(letters):
    return [item.strip("'") for item in letters]


class SentenceParser:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path

    def path(self):
        return self.csv_file_path

    def as_df(self):
        return pd.read_csv(
            os.path.join(os.getcwd(), "cleaned.csv"),
            dtype={
                "key": str,
                "press_time": np.float64,
                "release_time": np.float64,
                "platform_id": np.uint8,
                "session_id": np.uint8,
                "user_ids": np.uint8,
            },
        )

    def letters(self, as_list: bool = False):
        if as_list == True:
            return list(remove_invalid_keystrokes(self.as_df()).iloc[:, 1])
        elif as_list == False:
            return remove_invalid_keystrokes(self.as_df())

    def make_sentences(self, raw_df):
        ignorable = ["Key.cmd", "Key.tab", "Key.shift", "Key.shift_r"]
        filtered = []
        keys = clean_letters(remove_invalid_keystrokes(raw_df).loc[:, "key"])
        for i, key in enumerate(keys):
            filtered.append(key)

        filtered_list = []
        for i in range(len(filtered) - 1):
            if filtered[i] == "Key.backspace":
                if not filtered_list:
                    filtered_list.pop()
            else:
                filtered_list.append(filtered[i])
        filtered = filtered_list

        sentence = ""
        for i, key in enumerate(filtered):
            if i + 1 < len(filtered) and i - 1 >= 0:
                next_index = (i + 1) % len(filtered)
                if not key in ignorable:
                    if key == "Key.enter" or key == "." or key == "!" or key == "?":
                        sentence += "\n"
                        continue
                    elif key == "Key.space":
                        sentence += " "
                        continue
                    elif key == "Key.backspace" or key == "Key.ctrl":
                        continue
                    elif filtered[i - 1] == "Key.ctrl":
                        sentence += "\n"
                        continue
                    elif filtered[next_index] == "Key.backspace":
                        continue
                    sentence += key.strip()
        return sentence

    def get_words(self, data_df):
        tokenized_words = []
        sentences = self.make_sentences(data_df)
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentences)
        for token in doc:
            tokenized_words.append(token.text)
        return tokenized_words
