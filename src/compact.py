import os
import pandas as pd
from tqdm import tqdm
import csv

# https://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
def remove_invalid_keystrokes(df):
    # A helper function that takes as input a dataframe, and return a new
    # dataframe no longer containing rows with the string "<0>"
    return df.loc[df["key"] != "<0>"]

def get_new_format():
    df = pd.read_csv(
        os.path.join(os.getcwd(), "samples", "fpd_new_session_no_nans.csv"),
    )
    # df = df[["direction", "key", "time", "user_ids"]]
    df = df.astype({"direction": str, "key": str, "time": float, "user_ids": str})
    return remove_invalid_keystrokes(df)
def find_pairs(df):
    df["visited"] = False
    # Iterate over the keys and find the "P" and "R" pairs for each key
    for _, row in tqdm(df.iterrows()):
        if row["direction"] == "R":
            continue
        key = row["key"]
        potential_release_matches = df[(~df["visited"])&(df["direction"]=="R")& (df["key"]==key)& (df["session_ids"]==row["session_ids"])& (df["platform_ids"]==row["platform_ids"])& (df["user_ids"]==row["user_ids"])]
        if len(potential_release_matches) > 0:
            first_row = potential_release_matches.iloc[0]
            first_row_index = first_row.name
            df.loc[first_row_index, "visited"] = True
            if row["session_ids"] == first_row["session_ids"] and row["platform_ids"] == first_row["platform_ids"] and row["user_ids"] == first_row["user_ids"]:
                entry = [key, row["time"], first_row["time"], first_row["platform_ids"], first_row["session_ids"], first_row["user_ids"]]
                # with open("cleaned.csv", 'a', encoding='UTF8') as f:
                #     writer = csv.writer(f)
                #     writer.writerow(entry)
            else:
                print("NON MATCHING PAIR OF ROWS")
                print(row)
                print(first_row)
                input()
        else:
            print("No remaining matches found - skip key")

df = get_new_format()
find_pairs(df)
