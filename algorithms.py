import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_new_format():
    df = pd.read_csv(
        os.path.join(os.getcwd(), "samples", "fpd_new_session_no_nans.csv"),
        usecols=["key", "direction", "time"],
    )
    df = df[["direction", "key", "time"]]
    # TODO: Turn on when working with the actual dataset file
    # df["direction"] = df["direction"].apply(lambda x: "P" if x == 1 else "R")
    print(df)
    df = df.astype({"direction": str, "key": str, "time": float})
    print(df.dtypes)
    input("New dataset")
    return df


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


def kit_features(base_df, feature_type, use_new_dataset=True):
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


def remove_outliers_for_dictionary_data(data):
    data.sort()

    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    IQR = q3 - q1
    upper = q3 + (1.5 * IQR)
    lower = q1 - (1.5 * IQR)
    return [element for element in data if not element < lower or not element > upper]


def compare_algos():
    df = get_new_format()
    res1 = get_KHT_features(df)
    df = pd.read_csv(os.path.join(os.getcwd(), "samples", "s1.csv"), header=None)
    res2 = get_KHT_features(df, False)
    return res1 == res2


if __name__ == "__main__":
    df = get_new_format()

    data = get_KHT_features(df)
    # processed_KHT_data = {}
    # for key in list(data.keys()):
    #     res = remove_outliers_for_dictionary_data(data[key])
    #     processed_KHT_data[key] = res
    # print("Outlier Removed Data:")
    # print(processed_KHT_data)
    # input("KHT")

    # TODO: Spawn distinct threads to run each of the KIT flight calculations in parallel
    # https://stackoverflow.com/questions/46301933/how-to-wait-till-all-threads-finish-their-work
    print(kit_features(df, 1))
    # for f in tqdm(files):
    #     df = pd.read_csv(os.path.join(os.getcwd(), "Facebook", f), header=None)
    #     print(f)
    #     data = get_KHT_features(df)
    #     print(data)
    #     print(data)
