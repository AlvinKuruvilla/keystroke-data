import os
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import islice


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
        pair = []
        if not first == second:
            pair.append(first)
            pair.append(second)
            pairs.append(pair)
    return pairs


def new_kit(base_df, feature_type):
    # FIXME: Therre is a bug where the previosuly used rows are not set to visited, so if any repeat keys show up the
    # algorithm will choose the first time they appear from the begining of the file
    df = remove_invalid_keystrokes(base_df)
    pairs = unique_kit_keypairs(df)
    if feature_type == 1:
        first_event_type = "R"
        second_event_type = "P"
    elif feature_type == 2:
        first_event_type = "R"
        second_event_type = "R"
    elif feature_type == 3:
        first_event_type = "P"
        second_event_type = "P"
    elif feature_type == 4:
        first_event_type = "R"
        second_event_type = "R"
    df["visited"] = False
    features = defaultdict(list)
    for key_pair in pairs:
        first_key_search_res = df.loc[
            (df[0] == first_event_type)
            & (df["visited"] == False)
            & (df[1] == key_pair[0])
        ]
        second_key_search_res = df.loc[
            (df[0] == second_event_type)
            & (df["visited"] == False)
            & (df[1] == key_pair[1])
        ]
        if first_key_search_res.empty or second_key_search_res.empty:
            features[key_pair[0] + key_pair[1]].append([])
            continue
        first_key_index = first_key_search_res.index[0]
        first_key_search_res = first_key_search_res.iloc[0]
        second_key_index = second_key_search_res.index[0]
        second_key_search_res = second_key_search_res.iloc[0]
        print(first_key_index, second_key_index)
        input()
        df.at[first_key_index, "visited"] = True
        first_timing = first_key_search_res[2]
        df.at[second_key_index, "visited"] = True
        second_timing = second_key_search_res[2]
        features[key_pair[0] + key_pair[1]].append(second_timing - first_timing)
        print(df)

    return features


if __name__ == "__main__":
    files = os.listdir(os.path.join(os.getcwd(), "Facebook"))
    df = pd.read_csv(os.path.join(os.getcwd(), "samples", "s1.csv"), header=None)
    data = get_KHT_features(df)
    key_pairs = unique_kit_keypairs(df)
    # print(key_pairs)
    # for key_pair in key_pairs:
    #     feats = kit_features(key_pair, df, 2)
    #     print(key_pair)
    #     print(feats)
    print(new_kit(df, 1))
    # for f in tqdm(files):
    #     df = pd.read_csv(os.path.join(os.getcwd(), "Facebook", f), header=None)
    #     print(f)
    #     data = get_KHT_features(df)
    #     print(data)
