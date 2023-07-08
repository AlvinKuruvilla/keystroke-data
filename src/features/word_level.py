from collections import defaultdict
from tqdm import tqdm


def word_hold(word_list, raw_df):
    wh = defaultdict(list)
    # The word_list needs to be in the same order as they would
    # sequentially appear in the dataframe
    raw_df["visited"] = False
    # Convert all keys to lowercase because the data has mixed case so the
    # comparisons could fail
    # raw_df["key"] = raw_df["key"].str.lower()
    for word in tqdm(word_list):
        first_letter = word[0]
        # print(first_letter)
        potential_release_matches = raw_df[
            (~raw_df["visited"]) & (raw_df["key"].str.strip("'") == first_letter)
        ]
        if len(potential_release_matches) > 0:
            first_row = potential_release_matches.iloc[0]
            first_row_index = first_row.name
            # TODO: How to account for shift and other non-printing keys that could appear in between?
            # The ending bound is exclusive
            raw_df.loc[first_row_index : first_row_index + len(word), "visited"] = True
            press_time = raw_df.iloc[first_row_index]["press_time"]
            release_time = raw_df.iloc[first_row_index + len(word)]["release_time"]
            wh[word].append(release_time - press_time)
    return wh
