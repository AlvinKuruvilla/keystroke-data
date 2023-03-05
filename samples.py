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


def kit_features(keypair, base_df, feature_type):
    df = remove_invalid_keystrokes(base_df)
    df["visited"] = False
    features = defaultdict(list)
    if feature_type == 1:
        for index, row in df.iterrows():
            # A shortcu/hack to prevent the algorithm from searching for a "corresponding match" more than 2 rows away from the initial match
            max_hops = 0
            if row[1] == keypair[0] and row[0] == "P" and row["visited"] == False:
                # print("FIRST MATCH:")
                # print(row)
                # input()
                release_time = row[2]
                row["visited"] = True
                for next_index, next_row in islice(df.iterrows(), index, None):
                    if max_hops < 2:
                        max_hops += 1
                        if (
                            next_row[1] == keypair[1]
                            and next_row[0] == "R"
                            and next_row["visited"] == False
                        ):
                            # print("CORRESPONDING MATCH:")
                            # print(next_row)
                            # input()
                            press_time = next_row[2]
                            next_row["visited"] = True
                            features[keypair[0] + keypair[1]].append(
                                press_time - release_time
                            )
                            # After we find our first match we don't want to keep looking for rows with the second key
                            break
    elif featre_type == 2:
        for index, row in df.iterrows():
            # A shortcu/hack to prevent the algorithm from searching for a "corresponding match" more than 2 rows away from the initial match
            max_hops = 0
            if row[1] == keypair[0] and row[0] == "R" and row["visited"] == False:
                # print("FIRST MATCH:")
                # print(row)
                # input()
                release_time = row[2]
                row["visited"] = True
                for next_index, next_row in islice(df.iterrows(), index, None):
                    if max_hops < 2:
                        max_hops += 1
                        if (
                            next_row[1] == keypair[1]
                            and next_row[0] == "R"
                            and next_row["visited"] == False
                        ):
                            # print("CORRESPONDING MATCH:")
                            # print(next_row)
                            # input()
                            press_time = next_row[2]
                            next_row["visited"] = True
                            features[keypair[0] + keypair[1]].append(
                                press_time - release_time
                            )
                            # After we find our first match we don't want to keep looking for rows with the second key
                            break
    elif feature_type == 3:
        for index, row in df.iterrows():
            # A shortcu/hack to prevent the algorithm from searching for a "corresponding match" more than 2 rows away from the initial match
            max_hops = 0
            if row[1] == keypair[0] and row[0] == "p" and row["visited"] == False:
                # print("FIRST MATCH:")
                # print(row)
                # input()
                release_time = row[2]
                row["visited"] = True
                for next_index, next_row in islice(df.iterrows(), index, None):
                    if max_hops < 2:
                        max_hops += 1
                        if (
                            next_row[1] == keypair[1]
                            and next_row[0] == "P"
                            and next_row["visited"] == False
                        ):
                            # print("CORRESPONDING MATCH:")
                            # print(next_row)
                            # input()
                            press_time = next_row[2]
                            next_row["visited"] = True
                            features[keypair[0] + keypair[1]].append(
                                press_time - release_time
                            )
                            # After we find our first match we don't want to keep looking for rows with the second key
                            break
    if feature_type == 4:
        for index, row in df.iterrows():
            # A shortcu/hack to prevent the algorithm from searching for a "corresponding match" more than 2 rows away from the initial match
            max_hops = 0
            if row[1] == keypair[0] and row[0] == "R" and row["visited"] == False:
                # print("FIRST MATCH:")
                # print(row)
                # input()
                release_time = row[2]
                row["visited"] = True
                for next_index, next_row in islice(df.iterrows(), index, None):
                    if max_hops < 2:
                        max_hops += 1
                        if (
                            next_row[1] == keypair[1]
                            and next_row[0] == "P"
                            and next_row["visited"] == False
                        ):
                            # print("CORRESPONDING MATCH:")
                            # print(next_row)
                            # input()
                            press_time = next_row[2]
                            next_row["visited"] = True
                            features[keypair[0] + keypair[1]].append(
                                press_time - release_time
                            )
                            # After we find our first match we don't want to keep looking for rows with the second key
                            break
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


if __name__ == "__main__":
    files = os.listdir(os.path.join(os.getcwd(), "Facebook"))
    df = pd.read_csv(os.path.join(os.getcwd(), "samples", "s1.csv"), header=None)
    data = get_KHT_features(df)
    key_pairs = unique_kit_keypairs(df)
    print(key_pairs)
    for key_pair in key_pairs:
        feats = kit_features(key_pair, df, 1)
        print(key_pair)
        print(feats)
    # for f in tqdm(files):
    #     df = pd.read_csv(os.path.join(os.getcwd(), "Facebook", f), header=None)
    #     print(f)
    #     data = get_KHT_features(df)
    #     print(data)
