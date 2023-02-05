import os
import pandas as pd
from collections import defaultdict
import shutil


def remake_folders() -> None:
    shutil.rmtree(os.path.join(os.getcwd(), "Facebook"))
    shutil.rmtree(os.path.join(os.getcwd(), "Instagram"))
    shutil.rmtree(os.path.join(os.getcwd(), "Twitter"))
    os.mkdir(os.path.join(os.getcwd(), "Facebook"))
    os.mkdir(os.path.join(os.getcwd(), "Instagram"))
    os.mkdir(os.path.join(os.getcwd(), "Twitter"))


def find_all_unique_string_prefixes(train_files):
    # Find all unique string prefixes so we can order the files accordingly
    # For example one of our unique prefixes should be t_05 allowing us to find
    # "t_05_1, "t_05_2, "t_05_3" returned in a list so we can then combine them
    # in order
    prefixes = set()
    for f in train_files:
        filename = os.path.splitext(f)[0]
        # We know for sure that in our dataset there are no double digit session
        # numbers, so since we already removed the extension we only need to remove
        # the one digit session id and the underscore before it
        prefix = filename[:-2]
        prefixes.add(prefix)
    return prefixes


def all_matching_files(train_files, prefixes):
    ordered_files = []
    res = defaultdict(list)
    final_matches = []
    for prefix in prefixes:
        matches = [s for s in train_files if prefix in s]
        for hit in matches:
          prefix_length = len(prefix)
          if hit[prefix_length] == "_":
            final_matches.append(hit)
        res[prefix].extend(final_matches)
        final_matches.clear()
    for k, values in res.items():
        sorted_values = sorted(values)
        ordered_files.append(sorted_values)
    return ordered_files


def get_platform_and_prefix(file_set):
    f = file_set[0]
    filename = os.path.splitext(f)[0]
    prefix = filename[:-2]
    first = filename[0]
    if first.upper() == "F":
        return ("Facebook", prefix)
    elif first.upper() == "I":
        return ("Instagram", prefix)
    elif first.upper() == "T":
        return ("Twitter", prefix)
    else:
        raise Exception("Unknown platform")


remake_folders()

os.chdir('Train/')
files = os.listdir(os.getcwd())
unique_prefixes = find_all_unique_string_prefixes(files)
sorted_files = all_matching_files(files, unique_prefixes)
df_list = []
# Now combine the dataframes of the csvs and write it out to a new csv
# in the correct platform folder
for file_set in sorted_files:
    platform_folder, combined_filename = get_platform_and_prefix(file_set)
    # print("Platform:", platform_folder)
    # print("Filename:", combined_filename)
    # input("FILE_SET")
    # Read CSV files from List
    for file in file_set:
        try:
          print(file)
          df = pd.read_csv(file, header=None, engine="python")
          input("HANG")
        except ValueError:
          print("Error on file:", file)
          input("HELLO")
        df_list.append(df)
    for df in df_list:
        df.dropna(inplace=True)
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(
        f"../{platform_folder}/{combined_filename}.csv", index=False, header=False
    )
    df_list.clear()