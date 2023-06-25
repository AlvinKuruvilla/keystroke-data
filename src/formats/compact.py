import os
import csv
import pandas as pd
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
    # df = df[["direction", "key", "time", "user_ids"]]
    df = df.astype({"direction": str, "key": str, "time": float, "user_ids": int})
    return remove_invalid_keystrokes(df)


# TODO: Lots of correctness checks need to be done
def find_pairs_potentially_optimized(df):
    df["visited"] = False
    key_groups = df.groupby("key")  # Group the dataframe by key character

    for row in tqdm(df.itertuples(index=False), total=df.shape[0]):
        if row.direction == "R":
            continue
        key = row.key
        potential_release_matches = key_groups.get_group(key).query(
            "~visited & direction == 'R' & platform_ids == @row.platform_ids & session_ids == @row.session_ids & user_ids == @row.user_ids"
        )

        if len(potential_release_matches) > 0:
            first_row = potential_release_matches.iloc[0]
            first_row_index = first_row.name
            df.at[first_row_index, "visited"] = True
            if (
                row.session_ids == first_row.session_ids
                and row.platform_ids == first_row.platform_ids
                and row.user_ids == first_row.user_ids
            ):
                entry = [
                    key,
                    row.time,
                    first_row.time,
                    row.platform_ids,
                    row.session_ids,
                    row.user_ids,
                ]

                with open("cleaned.csv", "a", encoding="UTF8") as f:
                    writer = csv.writer(f)
                    writer.writerow(entry)
            else:
                print("NON MATCHING PAIR OF ROWS")
                print(row)
                print(first_row)
                input()
        else:
            print("No remaining matches found - skip key")


def find_pairs(df):
    df["visited"] = False
    # Iterate over the keys and find the "P" and "R" pairs for each key
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row["direction"] == "R":
            continue
        key = row["key"]
        potential_release_matches = df[
            (~df["visited"])
            & (df["direction"] == "R")
            & (df["key"] == key)
            & (df["session_ids"] == row["session_ids"])
            & (df["platform_ids"] == row["platform_ids"])
            & (df["user_ids"] == row["user_ids"])
        ]
        if len(potential_release_matches) > 0:
            first_row = potential_release_matches.iloc[0]
            first_row_index = first_row.name
            df.loc[first_row_index, "visited"] = True
            if (
                row["session_ids"] == first_row["session_ids"]
                and row["platform_ids"] == first_row["platform_ids"]
                and row["user_ids"] == first_row["user_ids"]
            ):
                entry = [
                    key,
                    row["time"],
                    first_row["time"],
                    first_row["platform_ids"],
                    first_row["session_ids"],
                    first_row["user_ids"],
                ]
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


def read_csv_file(filename):
    columns = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # skip header
        for row in csv_reader:
            columns.append(row)
    return columns


def find_matching_indices(path: str):
    rows = read_csv_file(path)
    visited = set()
    matches = []
    direction_index = 1
    key_index = 2
    for i, row in tqdm(enumerate(rows), total=len(rows)):
        if row[direction_index] == "R":
            continue
        if i in visited:
            continue
        key = row[key_index]
        platform_id = row[4]
        user_id = row[5]
        session_id = row[6]
        for j in range(i + 1, len(rows)):
            if j in visited:
                continue  # Skip visited columns
            if rows[j][direction_index] == "P":
                continue
            if (
                rows[j][direction_index] == "R"
                and rows[j][key_index] == key
                and rows[j][4] == platform_id
                and rows[j][5] == user_id
                and rows[j][6] == session_id
            ):
                matches.append((i, j))
                # print(i, j)
                # print(rows[i])
                # print(rows[j])
                # input()
                visited.add(i)
                visited.add(j)
                # I think its safe to break out of the loop because if we
                # end up here that means we found a corresponding 'j' release
                # row to the 'i' press row, so do not need to keep looking
                break
    return matches


def write_pairs_to_file(pairs, path):
    rows = read_csv_file(path)
    for pair in pairs:
        i = pair[0]
        j = pair[1]
        key = rows[i][2]
        press_time = rows[i][3]
        release_time = rows[j][3]
        platform = rows[j][4]
        user = rows[j][5]
        session = rows[j][6]
        entry = [key, press_time, release_time, platform, user, session]
        with open("out.csv", "a", encoding="UTF8") as f:
            writer = csv.writer(f)
            writer.writerow(entry)


# df = get_new_format()
# find_pairs(df)
cols = read_csv_file(
    os.path.join(os.getcwd(), "samples", "fpd_new_session_no_nans.csv"),
)
# print(cols)
matching_indices = find_matching_indices(
    os.path.join(os.getcwd(), "samples", "fpd_new_session_no_nans.csv"),
)
write_pairs_to_file(
    matching_indices,
    os.path.join(os.getcwd(), "samples", "fpd_new_session_no_nans.csv"),
)
