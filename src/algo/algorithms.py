import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_new_format():
    df = pd.read_csv(
        os.path.join(os.getcwd(), "samples", "fpd_new_session_no_nans.csv"),
    )
    df = df[["direction", "key", "time", "user_ids"]]
    df = df.astype({"direction": str, "key": str, "time": float, "user_ids": str})
    return remove_invalid_keystrokes(df)


# https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
def remove_invalid_keystrokes(df):
    # A helper function that takes as input a dataframe, and return a new
    # dataframe no longer containing rows with the string "<0>"
    return df.loc[df["key"] != "<0>"]


def balance_lists(release, press):
    length_diff = len(press) - len(release)
    if length_diff > 0:
        press = press[:-length_diff]
    elif length_diff < 0:
        release = release[: len(release) + length_diff]
    return (release, press)


def get_KHT_features(df, use_new_dataset=True):
    processed_df = remove_invalid_keystrokes(df)
    print(processed_df)

    unique_keys = processed_df.iloc[:, 1].unique()

    features = defaultdict(list)
    for key in tqdm(unique_keys):
        if use_new_dataset is False:
            rows_for_key = processed_df.loc[processed_df[1] == key]
        else:
            rows_for_key = processed_df.loc[processed_df["key"] == key]
        if use_new_dataset is False:
            press_rows_for_key = rows_for_key.loc[rows_for_key[0] == "P"][2].tolist()
            release_rows_for_key = rows_for_key.loc[rows_for_key[0] == "R"][2].tolist()
        else:
            press_rows_for_key = rows_for_key.loc[rows_for_key["direction"] == "P"][
                "time"
            ].tolist()
            release_rows_for_key = rows_for_key.loc[rows_for_key["direction"] == "R"][
                "time"
            ].tolist()

        # TODO: This line will cause a crash if he timing arrays are not the same length, how do we want to handle that?
        # The ```balance_list``` function is a temporary solution
        balanced_release, balanced_press = balance_lists(
            release_rows_for_key, press_rows_for_key
        )
        # print("Key:", key)
        # print("Release:", balanced_release)
        # print("Press:", balanced_press)
        result = np.subtract(balanced_release, balanced_press)
        for element in result:
            # FIXME: Check if this condition is needed due to a data issue or a problem in the code (this could be because of the list balancing causing misalignments in the subtraction)
            if element < 0:
                continue
            features[key].append(element)
    return features


def select_in_df(df, start, end):
    return df[start:end].values.tolist()


def sliding_window_KHT(processed_df, window_size):
    features = defaultdict(list)
    for i, g in processed_df.groupby(processed_df.index // 4):
        print("Index:", i)
        unique_keys = g.iloc[:, 1].unique()
        for key in unique_keys:
            rows_for_key = g.loc[g["key"] == key]
            press_rows_for_key = rows_for_key.loc[rows_for_key["direction"] == "P"][
                "time"
            ].tolist()
            release_rows_for_key = rows_for_key.loc[rows_for_key["direction"] == "R"][
                "time"
            ].tolist()
            try:
                result = np.subtract(release_rows_for_key, press_rows_for_key)
            except ValueError:
                print("BAD")
                print(g)
                continue
            for element in result:
                # FIXME: Check if this condition is needed due to a data issue or a problem in the code (this could be because of the list balancing causing misalignments in the subtraction)
                if element < 0:
                    continue
                features[key].append(element)
    return features


def unique_kit_keypairs(df):
    processed_df = remove_invalid_keystrokes(df)

    keys = list(processed_df.iloc[:, 1])
    pairs = [
        (keys[i], keys[i + 1]) for i in range(len(keys) - 1) if keys[i] != keys[i + 1]
    ]
    return set(pairs)


def kit_features(df, feature_type, use_new_dataset=True):
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
        first_event_type = "P"
        second_event_type = "R"
    df["visited"] = False
    features = defaultdict(list)
    for key_pair in tqdm(pairs):
        if not use_new_dataset:
            first_key_search_res = df.loc[
                (df[0] == first_event_type)
                & (df["visited"] is False)
                & (df[1] == key_pair[0])
            ]
            second_key_search_res = df.loc[
                (df[0] == second_event_type)
                & (df["visited"] is False)
                & (df[1] == key_pair[1])
            ]
        else:
            # print("First event:", first_event_type)
            # print("Second event:", second_event_type)
            # print("Key:", key_pair[0], key_pair[1])
            # print(
            #     df.loc[
            #         (df["key"] == key_pair[0])
            #         & (df["direction"] == first_event_type)
            #         & (~df["visited"])
            #     ]
            # )
            first_key_search_res = df.loc[
                (df["direction"] == first_event_type)
                & (~df["visited"])
                & (df["key"] == key_pair[0])
            ]
            second_key_search_res = df.loc[
                (df["direction"] == second_event_type)
                & (~df["visited"])
                & (df["key"] == key_pair[1])
            ]
        if first_key_search_res.empty or second_key_search_res.empty:
            features[key_pair[0] + key_pair[1]].append([])
            continue
        first_key_index = first_key_search_res.index[0]
        first_key_search_res = first_key_search_res.iloc[0]
        second_key_index = second_key_search_res.index[0]
        second_key_search_res = second_key_search_res.iloc[0]
        df.at[first_key_index, "visited"] = True
        first_timing = first_key_search_res[2]
        df.at[second_key_index, "visited"] = True
        second_timing = second_key_search_res[2]
        features[key_pair[0] + key_pair[1]].append(second_timing - first_timing)

    return features


def sliding_window_KIT(processed_df, feature_type):
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
        first_event_type = "P"
        second_event_type = "R"
    df["visited"] = False
    features = defaultdict(list)
    for i, g in processed_df.groupby(processed_df.index // 4):
        print(i)
        pairs = unique_kit_keypairs(g)
        for key_pair in pairs:
            first_key_search_res = df.loc[
                (df["direction"] == first_event_type)
                & (~df["visited"])
                & (df["key"] == key_pair[0])
            ]
            second_key_search_res = df.loc[
                (df["direction"] == second_event_type)
                & (~df["visited"])
                & (df["key"] == key_pair[1])
            ]
            if first_key_search_res.empty or second_key_search_res.empty:
                features[key_pair[0] + key_pair[1]].append([])
                continue
            first_key_index = first_key_search_res.index[0]
            first_key_search_res = first_key_search_res.iloc[0]
            second_key_index = second_key_search_res.index[0]
            second_key_search_res = second_key_search_res.iloc[0]
            df.at[first_key_index, "visited"] = True
            first_timing = first_key_search_res[2]
            df.at[second_key_index, "visited"] = True
            second_timing = second_key_search_res[2]
            features[key_pair[0] + key_pair[1]].append(second_timing - first_timing)

    return features


def compare_algos():
    df = get_new_format()
    res1 = get_KHT_features(df)
    df = pd.read_csv(os.path.join(os.getcwd(), "samples", "s1.csv"), header=None)
    res2 = get_KHT_features(df, False)
    return res1 == res2

if __name__ == "__main__":
    df = get_new_format()
    cols = df.columns
    # sliding_window_KHT(df, 4)
    # input()
    # print(cols)
    # sliding_window_KIT(df, 4)
    # input()
    user_ids = list(df["user_ids"].unique())
    for user_id in user_ids:
        sub_df = df[df["user_ids"] == user_id]
        print(sub_df.shape)
        input()
        data = sliding_window_KIT(sub_df, 1)
        with open(
            os.path.join(
                os.getcwd(), "features", "kit", f"KIT_1_for_{str(user_id)}.json"
            ),
            "w",
        ) as f:
            json.dump(data, f)
        break
