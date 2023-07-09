from collections import defaultdict


def create_kht_data_from_df(df):
    kht_dict = defaultdict(list)
    for i, row in df.iterrows():
        kht_dict[row["key"]].append(row["release_time"] - row["press_time"])
    return kht_dict


def create_kit_data_from_df(df, kit_feature_type):
    kit_dict = defaultdict(list)
    # We must use this to get the max_row because we are looping by the index and if we use the length the index will be off
    # The length of the dataframe != row indices
    max_rows = df.index[-1]
    # print(max_rows)
    for i, row in df.iterrows():
        current_row = row
        if i < max_rows - 1:
            next_row = df.loc[i + 1]
            # print(row)
            # print(next_row)
            # input()
            key = current_row["key"] + next_row["key"]
            initial_press = current_row["press_time"]
            second_press = next_row["press_time"]
            initial_release = current_row["release_time"]
            second_release = next_row["release_time"]
            if kit_feature_type == 1:
                kit_dict[key].append(float(second_press) - float(initial_release))
            elif kit_feature_type == 2:
                kit_dict[key].append(float(second_release) - float(initial_release))
            elif kit_feature_type == 3:
                kit_dict[key].append(float(second_press) - float(initial_press))
            elif kit_feature_type == 4:
                kit_dict[key].append(float(second_release) - float(initial_press))
    # input("Comparing FS results")
    return kit_dict
