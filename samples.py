import os
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm


def remove_invalid_keystrokes(df):
    # A helper function that takes as input a dataframe, and return a new
    # dataframe no longer containing rows with the string "<0>"
    for index, row in df.iterrows():
        # print(row[1])
        if row[1] == "<0>":
            # print("HERE")
            df.drop(index=index, inplace=True)
    return df


def balance_lists(release, press):
    if len(press) > len(release):
        while len(release) < len(press):
            press.pop()
    else:
        while len(release) > len(press):
            release.pop()
    return (release, press)


def get_KHT_features(df):
    processed_df = remove_invalid_keystrokes(df)
    unique_keys = processed_df.iloc[:, 1].unique()
    features = defaultdict(list)
    for key in unique_keys:
        rows_for_key = processed_df.loc[processed_df[1] == key]
        press_rows_for_key = rows_for_key.loc[rows_for_key[0] == "P"][2].tolist()
        release_rows_for_key = rows_for_key.loc[rows_for_key[0] == "R"][2].tolist()
        # TODO: This line will cause a crash if he timing arrays are not the same length, how do we want to handle that?
        # The ```balance_list``` function is a temporary solution
        balanced_release, balanced_press = balance_lists(
            release_rows_for_key, press_rows_for_key
        )
        print("Key:", key)
        print("Release:", balanced_release)
        print("Press:", balanced_press)
        result = np.subtract(balanced_release, balanced_press)
        for element in result:
            # FIXME: Check if this condition is needed due to a data issue or a problem in the code (this could be because of the list balancing causing misalignments in the subtraction)
            if element < 0:
                continue
            features[key].append(element)
    return features


def unique_kit_keypairs(df):
    processed_df = remove_invalid_keystrokes(df)

    all_keys = processed_df.iloc[:, 1]
    print(all_keys)
    pairs = []
    for first, second in zip(all_keys, all_keys[1:]):
        if not first == second:
            pairs.append(first + second)
    return pairs


if __name__ == "__main__":
    files = os.listdir(os.path.join(os.getcwd(), "Facebook"))
    df = pd.read_csv(os.path.join(os.getcwd(), "samples", "s1.csv"), header=None)
    data = get_KHT_features(df)
    key_pairs = unique_kit_keypairs(df)
    print(key_pairs)
    # for f in tqdm(files):
    #     df = pd.read_csv(os.path.join(os.getcwd(), "Facebook", f), header=None)
    #     print(f)
    #     data = get_KHT_features(df)
    #     print(data)
