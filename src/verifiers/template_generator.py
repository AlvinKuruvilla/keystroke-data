import os
import pandas as pd
from collections import defaultdict
def read_compact_format():
    return pd.read_csv(os.path.join(os.getcwd(), "cleaned.csv"))


def get_compact_data_by_user_and_platform_id(user_id, platform_id):
    df = read_compact_format()
    return df[(df["user_ids"] == user_id) & (df["platform_id"] == platform_id)]


def create_kht_template_for_user_and_platform_id(user_id, platform_id):
    df = get_compact_data_by_user_and_platform_id(user_id, platform_id)
    kht_dict = defaultdict(list)
    for i, row in df.iterrows():
        kht_dict[row["key"]].append(row["release_time"] - row["press_time"])
    return kht_dict


def create_kht_verification_attempt_for_user_and_platform_id(user_id):
    df = read_compact_format()
    df = df[(df["user_ids"] == user_id) & ~(df["platform_id"] == 1)]
    kht_dict = defaultdict(list)
    for i, row in df.iterrows():
        kht_dict[row["key"]].append(row["release_time"] - row["press_time"])
    return kht_dict


def create_kit_flight_template_for_user_and_platform(user_id, kit_feature_type):
    df = read_compact_format()
    df = df[(df["user_ids"] == user_id) & (df["platform_id"] == 1)]
    kit_dict = defaultdict(list)
    for i, g in df.groupby(df.index // 2):
        if g.shape[0] == 1:
            continue
        key = g.iloc[0]["key"] + g.iloc[1]["key"]
        initial_press = g.iloc[0]["press_time"]
        second_press = g.iloc[1]["press_time"]
        initial_release = g.iloc[0]["release_time"]

        second_release = g.iloc[1]["release_time"]
        if kit_feature_type == 1:
            kit_dict[key].append(float(second_press) - float(initial_release))
        elif kit_feature_type == 2:
            kit_dict[key].append(float(second_release) - float(initial_release))
        elif kit_feature_type == 3:
            kit_dict[key].append(float(second_press) - float(initial_press))
        elif kit_feature_type == 4:
            kit_dict[key].append(float(second_release) - float(initial_press))
    return kit_dict


def create_kit_flight_verification_attempt_for_user_and_platform(
    user_id, kit_feature_type
):
    df = read_compact_format()
    df = df[(df["user_ids"] == user_id) & ~(df["platform_id"] == 1)]
    kit_dict = defaultdict(list)
    for i, g in df.groupby(df.index // 2):
        if g.shape[0] == 1:
            continue
        key = g.iloc[0]["key"] + g.iloc[1]["key"]
        initial_press = g.iloc[0]["press_time"]
        second_press = g.iloc[1]["press_time"]
        initial_release = g.iloc[0]["release_time"]

        second_release = g.iloc[1]["release_time"]
        if kit_feature_type == 1:
            kit_dict[key].append(float(second_press) - float(initial_release))
        elif kit_feature_type == 2:
            kit_dict[key].append(float(second_release) - float(initial_release))
        elif kit_feature_type == 3:
            kit_dict[key].append(float(second_press) - float(initial_press))
        elif kit_feature_type == 4:
            kit_dict[key].append(float(second_release) - float(initial_press))
    return kit_dict
