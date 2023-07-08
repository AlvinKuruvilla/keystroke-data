import statistics
import os
from collections import defaultdict
from tqdm import tqdm
from features.word_parser import SentenceParser

from verifiers.heatmap import VerifierType, get_user_by_platform, Verifiers


def word_hold(word_list, raw_df):
    wh = defaultdict(list)
    # The word_list needs to be in the same order as they would
    # sequentially appear in the dataframe
    raw_df["visited"] = False
    for word in word_list:
        first_letter = word[0]
        # print(first_letter)
        potential_release_matches = raw_df[
            (~raw_df["visited"]) & (raw_df["key"].str.strip("'") == first_letter)
        ]
        if len(potential_release_matches) > 0:
            first_row = potential_release_matches.iloc[0]
            first_row_index = first_row.name
            # print(first_row_index)
            # print(raw_df.loc[first_row_index])
            # input()
            # TODO: How to account for shift and other non-printing keys that could appear in between?
            # The ending bound is exclusive
            raw_df.loc[first_row_index : first_row_index + len(word), "visited"] = True
            press_time = raw_df.loc[first_row_index]["press_time"]
            release_time = raw_df.loc[first_row_index + len(word)]["release_time"]
            wh[word].append(release_time - press_time)
    return wh


def word_unigraph_feature(word_list, raw_df):
    w_mean = defaultdict(list)
    w_median = defaultdict(list)
    w_stdev = defaultdict(list)
    # The word_list needs to be in the same order as they would
    # sequentially appear in the dataframe
    raw_df["visited"] = False
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
            i = first_row_index
            unigraph_times = []
            while i < first_row_index + len(word):
                unigraph = raw_df.iloc[i]["release_time"] - raw_df.iloc[i]["press_time"]
                unigraph_times.append(unigraph)
                i += 1
            w_mean[word].append(statistics.mean(unigraph_times))
            w_median[word].append(statistics.median(unigraph_times))
            try:
                w_stdev[word].append(statistics.stdev(unigraph_times))
            except statistics.StatisticsError:
                w_stdev[word].append(10)
    return (w_mean, w_median, w_stdev)


def word_hold_matrix(
    enroll_platform_id,
    probe_platform_id,
    enroll_session_id,
    probe_session_id,
    verifier_type,
):
    if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
        raise ValueError("Platform ID must be between 1 and 3")

    matrix = []
    # TODO: We have to do a better job of figuring out how many users there
    # are automatically so we don't need to keep changing it manually
    for i in tqdm(range(1, 26)):
        df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
        sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
        word_list = sp.get_words(df)
        enrollment = word_hold(word_list, df)
        row = []
        # TODO: We have to do a better job of figuring out how many users there
        # are automatically so we don't need to keep changing it manually
        for j in range(1, 26):
            df = get_user_by_platform(j, probe_platform_id, probe_session_id)
            word_list = sp.get_words(df)
            probe = word_hold(word_list, df)
            v = Verifiers(enrollment, probe)
            if verifier_type == VerifierType.Absolute:
                row.append(v.get_abs_match_score())
            elif verifier_type == VerifierType.Similarity:
                row.append(v.get_weighted_similarity_score())
            elif verifier_type == VerifierType.SimilarityUnweighted:
                row.append(v.get_similarity_score())
            elif verifier_type == VerifierType.Itad:
                row.append(v.itad_similarity())
            else:
                raise ValueError("Unknown VerifierType {}".format(verifier_type))
        matrix.append(row)
    return matrix
