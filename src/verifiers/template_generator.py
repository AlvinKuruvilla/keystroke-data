import os
import pandas as pd
from collections import defaultdict

def read_compact_format():
    return pd.read_csv(os.path.join(os.getcwd(), 'cleaned.csv'))
def get_compact_data_by_user_and_platform_id(user_id, platform_id):
    df = read_compact_format()
    return df[(df["user_ids"] == user_id)& (df["platform_id"] == platform_id)]
def create_kht_template_for_user_and_platform_id(user_id, platform_id):
    df = get_compact_data_by_user_and_platform_id(user_id, platform_id)
    kht_dict = defaultdict(list)
    for i, row in df.iterrows():
        kht_dict[row["key"]].append(row["release_time"]  - row["press_time"])
    return kht_dict
def create_kht_verification_attempt_for_user_and_platform_id(user_id):
    df = read_compact_format()
    df = df[(df["user_ids"] == user_id)& ~(df["platform_id"] == 1)]
    kht_dict = defaultdict(list)
    for i, row in df.iterrows():
        kht_dict[row["key"]].append(row["release_time"]  - row["press_time"])
    return kht_dict

