import math
import os
import pandas as pd
import numpy as np
from tqdm import tqdm 
# https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
def remove_invalid_keystrokes(df):
    # A helper function that takes as input a dataframe, and return a new
    # dataframe no longer containing rows with the string "<0>"
    return df.loc[df["key"] != "<0>"]

def get_new_format():
    df = pd.read_csv(
        os.path.join(os.getcwd(), "samples", "fpd_new_session_no_nans.csv"),
    )
    df = df[["direction", "key", "time", "user_ids"]]
    df = df.astype({"direction": str, "key": str, "time": float, "user_ids": str})
    return remove_invalid_keystrokes(df)

def find_pairs(list1, list2):
    # Sort the lists to optimize the algorithm
    list1.sort()
    list2.sort()

    result = []

    # Find the smaller list and its length
    if len(list1) >= len(list2):
        smaller_list = list2.copy()
        larger_list = list1.copy()
        larger_list.append(math.inf)

        ind_s = 0
        ind_l = 0

        while ind_s < len(smaller_list):
            if float(larger_list[ind_l]) <= float(smaller_list[ind_s]) and ind_l < len(larger_list) - 1:
                ind_l += 1

            else:
                # print(larger_list[ind_l-1])
                if (larger_list[ind_l - 1] != math.inf):
                    result.append((int(larger_list[ind_l - 1]), int(smaller_list[ind_s])))
                    smaller_list.remove((smaller_list[ind_s]))
                    larger_list.remove((larger_list[ind_l - 1]))
                ind_s += 1

            if ind_l == len(larger_list):
                break

        return result, larger_list, smaller_list

    else:
        smaller_list = list1.copy()
        larger_list = list2.copy()

        ind_s = 0
        ind_l = 0

        while ind_s < len(smaller_list):
            if float(larger_list[ind_l]) <= float(smaller_list[ind_s]) and ind_l < len(larger_list) - 1:
                ind_l += 1

            else:
                result.append((int(smaller_list[ind_s]), int(larger_list[ind_l])))
                smaller_list.remove((smaller_list[ind_s]))
                larger_list.remove((larger_list[ind_l]))
                ind_s += 1

            if ind_l == len(larger_list):
                break
        return result, smaller_list, larger_list
def closest_value(input_list, input_value):
  i = (np.abs(input_list - input_value)).argmin()
  return input_list[i]

if __name__ == "__main__":
    df = get_new_format()
    cols = df.columns
    user_ids = list(df["user_ids"].unique())
    keys = list(df["key"].unique())
    times = np.asarray(list(df["time"]))
    kht = {}
    khtimings = []
    for user_id in tqdm(user_ids):
        for key in keys:
            press_df = df[(df["user_ids"] == user_id) & (df['direction'] == 'P') & (key == df['key'])]
            release_df = df[(df["user_ids"] == user_id) & (df['direction'] == 'R') & (key == df['key'])]
            press_times = press_df['time'].tolist()
            release_times = release_df['time'].tolist()
            timing_pairs,_,_ = find_pairs(press_times, release_times)
            # for pair in timing_pairs:
            #     print(pair[1])
            #     print(pair[0])
            #     print(closest_value(times, pair[1]))
            #     print(closest_value(times, pair[0]))
            #      input() 
            khtimings = [closest_value(times, pair[1])-closest_value(times, pair[0]) for pair in timing_pairs]
            # print(f'letter: {key}, corrected_kht: {khtimings}') # dig in this further.
            kht[key] = khtimings
            
            

