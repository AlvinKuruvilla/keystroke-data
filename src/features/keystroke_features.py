from collections import defaultdict


def create_kht_data_from_df(df):
    kht_dict = defaultdict(list)
    for i, row in df.iterrows():
        kht_dict[row["key"]].append(row["release_time"] - row["press_time"])
    return kht_dict


def create_kit_data_from_df(df, kit_feature_type):
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
