from warnings import simplefilter
from sklearn.feature_selection import mutual_info_classif as MIC
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from scipy.stats import entropy
import statistics
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_selection import *
from enum import Enum
from tqdm import tqdm
from taaled import ld
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import kurtosis, skew
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from functools import lru_cache
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from scipy.stats import gaussian_kde
import math


class FeatureType(Enum):
    KHT = 1
    KIT = 2
    WORD_LEVEL = 3


def remove_invalid_keystrokes(df):
    # A helper function that takes as input a dataframe, and return a new
    # dataframe no longer containing rows with the string "<0>"
    for index, row in df.iterrows():
        # print(row[1])
        if row[1] == "<0>":
            # print("HERE")
            df.drop(index=index, inplace=True)
    return df


def make_gender_df():
    demographics = pd.read_csv("Filtered_Demographics.csv")
    return demographics


def make_bbmass_gender_df():
    demographics = pd.read_csv("BBMASS_Demographics.csv")
    return demographics


@lru_cache(maxsize=None)
def path_to_platform(path: str):
    filename = os.path.basename(path)
    return filename.split("_")[0]


@lru_cache(maxsize=None)
def path_to_class(path: str):
    platform = path_to_platform(path)
    classification = 0
    if platform.upper() == "F" or platform.upper() == "I":
        classification = 0
        return str(classification)
    elif platform.upper() == "T":
        classification = 1
        return str(classification)


@lru_cache(maxsize=None)
def path_to_id(path: str):
    print("PATH:", path)
    filename = os.path.basename(path)
    return filename.split("_")[1]


@lru_cache(maxsize=None)
def bbmass_path_to_id(path: str):
    filename = os.path.basename(path)
    # We choose 4 to the end of the string because the bbmass files are of the
    # format UserXXXX, so we only want the XXXX part
    end = len(filename)
    id = filename[4 : end - 4]
    return id


@lru_cache(maxsize=None)
def path_to_session(path: str):
    filename = os.path.basename(path)
    return filename.split("_")[2]


def split_into_four(df: pd.DataFrame):
    ret = []
    i = 4
    # df.index represents the number of rows in the dataframe
    for j in range(0, len(df.index), 4):
        ret.append(df.iloc[j:i])
        i += 4
    return ret


def merge_dataframes(df_list):
    return pd.concat(df_list, axis=0, ignore_index=True)


# Handles strings like "<0>""
def remove_invalid_keystrokes_from_data_list(data):
    for i in range(0, len(data)):
        df = data[i]
        for row in df.itertuples():
            # print(row[2])
            if row[2] == "<0>":
                # print("HERE")
                num = int(row.Index)
                # print(num)
                rem = df.drop(index=num)
                data[i] = rem
    # After removing the weird values the size of each dataframe element is
    # smaller so we need to coalesce. Re-partitioning will be the job of
    # subsequent methods that use the return value of this method
    return merge_dataframes(data)


################################################################################################################################
# KHT
# The functions in this cell are helpers to generate KHT dictionarie using
# user data dataframes


def get_KHT(keys_in_pipeline, search_key, search_key_timing):
    mask = np.ones(len(keys_in_pipeline))
    keys_in_pipeline = np.asarray(keys_in_pipeline)
    for i, row in enumerate(keys_in_pipeline):
        key = row[1]
        timing = row[2]
        # print("Key:", key)
        # print("Timing:", timing)
        if search_key == key:
            mask[i] = 0
            kht = int(float(search_key_timing)) - int(float(timing))
            # print(search_key_timing, float(timing))
            # print(kht)
            non_zero_indices = np.nonzero(mask)
            if len(non_zero_indices) > 0:
                keys_in_pipeline = keys_in_pipeline[non_zero_indices]
            else:
                keys_in_pipeline = []
            # print(
            #     key,
            #     int(float(search_key_timing)),
            #     int(float(timing)),
            #     int(float(search_key_timing)) - int(float(timing)),
            # )
            return keys_in_pipeline, kht

    return keys_in_pipeline, None


@lru_cache(maxsize=None)
def event_to_int(event: str) -> int:
    if event == "P":
        return 0
    elif event == "R":
        return 1


# Make sure that the direction of the event is an integer rather than "P" or "R"
def conform_to_int(data):
    result = []
    for row_idx in data:
        result.append([event_to_int(row_idx[0]), row_idx[1], row_idx[2]])
    return result


# function to get KHT feature dictionary for a given user
def get_KHT_features(data):
    feature_dictionary = {}
    keys_in_pipeline = []

    for row_idx in range(len(data)):
        keys_in_pipeline = list(data)
        # print("keys_in_pipeline:", keys_in_pipeline)
        curr_key = data[row_idx][1]
        # print("curr_key:", curr_key)
        # print("Action:", data[row_idx][0])
        curr_direction = event_to_int(data[row_idx][0])
        curr_timing = data[row_idx][2]
        if curr_direction == 0:
            keys_in_pipeline.append([curr_key, curr_timing])

        if curr_direction == 1:
            keys_in_pipeline, curr_kht = get_KHT(
                conform_to_int(keys_in_pipeline), curr_key, curr_timing
            )
            if curr_kht is None:
                continue
            else:
                if curr_key in list(feature_dictionary.keys()):
                    feature_dictionary[curr_key].append(curr_kht)
                else:
                    feature_dictionary[curr_key] = []
                    feature_dictionary[curr_key].append(curr_kht)

    return feature_dictionary


def kht_from_dataframe(df: pd.DataFrame):
    user_data = df.values
    user_feat_dict = get_KHT_features(user_data)
    return user_feat_dict


def get_kht_features_for_file(user_file_path: str):
    data_frame = pd.read_csv(user_file_path)
    user_data = data_frame.values
    user_feat_dict = get_KHT_features(user_data)
    return user_feat_dict


def get_all_users_features_KHT(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        if ".csv" in user_file and not user_file.startswith("."):
            print(user_file)
            data_frame = pd.read_csv(directory + user_file)
            processed_df = remove_invalid_keystrokes(data_frame)
            user_data = processed_df.values
            user_feat_dict = get_KHT_features(user_data)
            users_feat_dict[i + 1] = user_feat_dict
    return users_feat_dict


################################################################################################################################
# KIT
# The functions in this cell are helpers to generate KIT dictionarie using
# user data dataframes
def get_timings_KIT(keys_in_pipeline, search_key, search_key_timing):
    mask = np.ones(len(keys_in_pipeline))
    keys_in_pipeline = np.asarray(keys_in_pipeline)
    for row in keys_in_pipeline:
        try:
            i = int(row[0])
        except TypeError:
            print("Encountered error with:", i, "in get_timings_KIT")
            input("HANG")
        key = str(row[1])
        timing = str(row[2])
        if search_key == key:
            mask[i] = 0
            non_zero_indices = np.nonzero(mask)

            if len(non_zero_indices) > 0:
                keys_in_pipeline = keys_in_pipeline[non_zero_indices]
            else:
                keys_in_pipeline = []

            return keys_in_pipeline, timing, search_key_timing
    return keys_in_pipeline, None, None


def get_dataframe_KIT(data):
    """Input: data  Output: Dataframe with (key, press_time, release_time)"""
    keys_in_pipeline = []
    result_key = []
    press = []
    release = []
    for row_idx in range(len(data)):
        keys_in_pipeline = list(data)
        # print("Keys in pipeline: ", keys_in_pipeline)
        curr_key = data[row_idx][1]
        curr_direction = event_to_int(data[row_idx][0])
        # print("Key direction: ", curr_direction)
        curr_timing = data[row_idx][2]
        if curr_direction == 0:
            keys_in_pipeline.append([curr_key, curr_timing])

        if curr_direction == 1:
            # print("Pipeline Keys:", conform_to_int(keys_in_pipeline))
            keys_in_pipeline, curr_start, curr_end = get_timings_KIT(
                conform_to_int(keys_in_pipeline), curr_key, curr_timing
            )
            if curr_start is None:
                continue
            else:
                result_key.append(curr_key)
                press.append(curr_start)
                release.append(curr_end)

    resultant_data_frame = pd.DataFrame(
        list(zip(result_key, press, release)),
        columns=["Key", "Press_Time", "Release_Time"],
    )
    # print("Result:", resultant_data_frame)
    return resultant_data_frame


def get_KIT_features_F1(data):
    """Input: keystroke data, Output: Dictionary of (next_key_press - current_key_release)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][2]
        next_timing = data[row_idx + 1][1]
        if math.isnan(int(float(next_timing)) - int(float(curr_timing))):
            print("NAN FOUND: get_KIT_features_F1")
            print("Encountered Key: ", curr_key)
            print("Next Key:", next_key)
        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


def get_KIT_features_F1_from_file(profile_path):
    features = {}
    df = pd.read_csv(profile_path)
    data_frame = get_dataframe_KIT(df.values)
    user_data = data_frame.values
    features = get_KIT_features_F1(user_data)
    return features


# function to get Flight2 KIT feature dictionary for a given user
def get_KIT_features_F2(data):
    """Input: keystroke data, Output: Dictionary of (next_key_press - current_key_press)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][1]
        next_timing = data[row_idx + 1][1]
        if math.isnan(int(float(next_timing)) - int(float(curr_timing))):
            print("NAN FOUND: get_KIT_features_F2")
            print("Encountered Key: ", curr_key)
            print("Next Key:", next_key)

        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


# function to get Flight3 KIT feature dictionary for a given user
def get_KIT_features_F3(data):
    """Input: keystroke data, Output: Dictionary of (next_key_release - current_key_release)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][2]
        next_timing = data[row_idx + 1][2]
        if math.isnan(int(float(next_timing)) - int(float(curr_timing))):
            print("NAN FOUND: get_KIT_features_F3")
            print("Encountered Key: ", curr_key)
            print("Next Key:", next_key)

        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


# function to get Flight3 KIT feature dictionary for a given user
def get_KIT_features_F4(data):
    """Input: keystroke data, Output: Dictionary of (next_key_release - current_key_press)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][1]
        next_timing = data[row_idx + 1][2]
        if math.isnan(int(float(next_timing)) - int(float(curr_timing))):
            print("NAN FOUND: get_KIT_features_F4")
            print("Encountered Key: ", curr_key)
            print("Next Key:", next_key)

        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


def kit_from_dataframe(df: pd.DataFrame, kit_feature_index: int):
    df = get_dataframe_KIT(df.values)
    user_data = df.values
    assert 1 <= kit_feature_index <= 4
    if kit_feature_index == 1:
        return get_KIT_features_F1(user_data)
    elif kit_feature_index == 2:
        return get_KIT_features_F2(user_data)
    elif kit_feature_index == 3:
        return get_KIT_features_F3(user_data)
    elif kit_feature_index == 4:
        return get_KIT_features_F4(user_data)


def get_all_users_features_KIT(directory):
    users_feat_dict_f1 = {}
    users_feat_dict_f2 = {}
    users_feat_dict_f3 = {}
    users_feat_dict_f4 = {}
    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        if ".csv" in user_file and not user_file.startswith("."):
            data_frame = pd.read_csv(directory + user_file)
            # print("Read:", data_frame)
            data_frame = get_dataframe_KIT(data_frame.values)
            processed_df = remove_invalid_keystrokes(data_frame)
            # print("DataFrame:", data_frame)
            user_data = processed_df.values
            # print("User Data:", user_data)

            user_feat_dict_f1 = get_KIT_features_F1(user_data)
            users_feat_dict_f1[i + 1] = user_feat_dict_f1

            user_feat_dict_f2 = get_KIT_features_F2(user_data)
            users_feat_dict_f2[i + 1] = user_feat_dict_f2

            user_feat_dict_f3 = get_KIT_features_F3(user_data)
            users_feat_dict_f3[i + 1] = user_feat_dict_f3

            user_feat_dict_f4 = get_KIT_features_F4(user_data)
            users_feat_dict_f4[i + 1] = user_feat_dict_f4
        else:
            print("File skipped", user_file)
    return (
        users_feat_dict_f1,
        users_feat_dict_f2,
        users_feat_dict_f3,
        users_feat_dict_f4,
    )


################################################################################################################################
# BBMAS Word Level Features
# For word level features we make a map of all of the individual letters and take an average of the timing values for their associated value
# Then we do PP, PR, RP, RR calculations using those values for the first and letters respectively
# Each column will do stored in the form of ""WORD"+ PP/PR/RP/RR"

""" function to return word level statistics for each extracted word.
These features are as follows:
1) Word hold time
2) Average, Standard Deviation and Median of all key hold times in the word
3) Average, Standard Deviation and Median of all flight 1 features for all digraphs in the word
4) Average, Standard Deviation and Median of all flight 2 features for all digraphs in the word
5) Average, Standard Deviation and Median of all flight 3 features for all digraphs in the word
6) Average, Standard Deviation and Median of all flight 4 features for all digraphs in the word
"""


def bbmas_get_advanced_word_level_features(words_in_pipeline):
    def get_word_hold(words_in_pipeline):
        res = int(float(words_in_pipeline[-1][2])) - int(float(words_in_pipeline[0][1]))
        if math.isnan(res):
            print("NAN FOUND get_word_hold (bbmas)")
            print(int(float(words_in_pipeline[-1][2])))
            print(int(float(words_in_pipeline[0][1])))
            input()
        return int(float(words_in_pipeline[-1][2])) - int(
            float(words_in_pipeline[0][1])
        )

    def get_avg_std_median_key_hold(words_in_pipeline):
        key_holds = []
        for _, press, release in words_in_pipeline:
            key_holds.append(int(float(release)) - int(float(press)))
        if math.isnan(np.mean(key_holds)):
            print("NAN FOUND get_avg_std_median_key_hold, avg - bbmas")
            input()

        if math.isnan(np.std(key_holds)):
            print("NAN FOUND get_avg_std_median_key_hold, std - bbmas")
            input()
        if math.isnan(np.median(key_holds)):
            print("NAN FOUND get_avg_std_median_key_hold, median - bbmas")
            input()
        return np.mean(key_holds), np.std(key_holds), np.median(key_holds)

    def get_avg_std_median_flights(words_in_pipeline):
        flights_1 = []
        flights_2 = []
        flights_3 = []
        flights_4 = []
        if len(words_in_pipeline) > 1:
            for i in range(len(words_in_pipeline) - 1):
                k1_r = words_in_pipeline[i][2]
                k1_p = words_in_pipeline[i][1]
                k2_r = words_in_pipeline[i + 1][2]
                k2_p = words_in_pipeline[i + 1][1]
                flights_1.append(int(float(k2_p)) - int(float(k1_r)))
                flights_2.append(int(float(k2_r)) - int(float(k1_r)))
                flights_3.append(int(float(k2_p)) - int(float(k1_p)))
                flights_4.append(int(float(k2_r)) - int(float(k1_p)))
        # TODO: What is the best way to handle a case like this,
        # since this means our words_in_pipeline is something like:
        # [['a',1496842340672, 1496842340002 ]]
        # In a case like this, there is no second key to use
        else:
            k1_r = words_in_pipeline[0][2]
            k1_p = words_in_pipeline[0][1]
            k2_r = words_in_pipeline[0][2]
            k2_p = words_in_pipeline[0][1]
            flights_1.append(int(float(k2_p)) - int(float(k1_r)))
            flights_2.append(int(float(k2_r)) - int(float(k1_r)))
            flights_3.append(int(float(k2_p)) - int(float(k1_p)))
            flights_4.append(int(float(k2_r)) - int(float(k1_p)))

        return (
            np.mean(flights_1),
            np.std(flights_1),
            np.median(flights_1),
            np.mean(flights_2),
            np.std(flights_2),
            np.median(flights_2),
            np.mean(flights_3),
            np.std(flights_3),
            np.median(flights_3),
            np.mean(flights_4),
            np.std(flights_4),
            np.median(flights_4),
        )

    wh = get_word_hold(words_in_pipeline)
    avg_kh, std_kh, median_kh = get_avg_std_median_key_hold(words_in_pipeline)
    (
        avg_flight1,
        std_flight1,
        median_flight1,
        avg_flight2,
        std_flight2,
        median_flight2,
        avg_flight3,
        std_flight3,
        median_flight3,
        avg_flight4,
        std_flight4,
        median_flight4,
    ) = get_avg_std_median_flights(words_in_pipeline)
    if math.isnan(wh):
        print("NAN FOUND get_advanced_word_level_features, wh - bbmas")
        input()
    if math.isnan(avg_kh):
        print("NAN FOUND get_advanced_word_level_features, avg_kh - bbmas")
        input()
    if math.isnan(std_kh):
        print("NAN FOUND get_advanced_word_level_features, std_kh - bbmas")
        input()
    if math.isnan(median_kh):
        print("NAN FOUND get_advanced_word_level_features, median_kh - bbmas")
        input()
    if math.isnan(avg_flight1):
        print("NAN FOUND get_advanced_word_level_features, avg_flight1 - bbmas")
        print(words_in_pipeline)
        input()
    if math.isnan(std_flight1):
        print("NAN FOUND get_advanced_word_level_features, std_flight1 - bbmas")
        input()
    if math.isnan(median_flight1):
        print("NAN FOUND get_advanced_word_level_features, median_flight1 - bbmas")
        input()

    if math.isnan(avg_flight2):
        print("NAN FOUND get_advanced_word_level_features, avg_flight2 - bbmas")
        input()
    if math.isnan(std_flight2):
        print("NAN FOUND get_advanced_word_level_features, std_flight2 - bbmas")
        input()
    if math.isnan(median_flight2):
        print("NAN FOUND get_advanced_word_level_features, median_flight2 - bbmas")
        input()

    if math.isnan(avg_flight3):
        print("NAN FOUND get_advanced_word_level_features, avg_flight3 - bbmas")
        input()
    if math.isnan(std_flight3):
        print("NAN FOUND get_advanced_word_level_features, std_flight3 - bbmas")
        input()
    if math.isnan(median_flight3):
        print("NAN FOUND get_advanced_word_level_features, median_flight3 - bbmas")
        input()

    if math.isnan(avg_flight4):
        print("NAN FOUND get_advanced_word_level_features, avg_flight4 - bbmas")
        input()
    if math.isnan(std_flight4):
        print("NAN FOUND get_advanced_word_level_features, std_flight4 - bbmas")
        input()
    if math.isnan(median_flight4):
        print("NAN FOUND get_advanced_word_level_features, median_flight4 - bbmas")
        input()

    return [
        wh,
        avg_kh,
        std_kh,
        median_kh,
        avg_flight1,
        std_flight1,
        median_flight1,
        avg_flight2,
        std_flight2,
        median_flight2,
        avg_flight3,
        std_flight3,
        median_flight3,
        avg_flight4,
        std_flight4,
        median_flight4,
    ]


# get KIT feature based on current key and timing values
def bbmass_get_timings_KIT(keys_in_pipeline, search_key, search_key_timing):
    mask = np.ones(len(keys_in_pipeline))
    keys_in_pipeline = np.asarray(keys_in_pipeline)
    for i, (key, timing) in enumerate(keys_in_pipeline):
        if search_key == key:
            mask[i] = 0
            non_zero_indices = np.nonzero(mask)

            if len(non_zero_indices) > 0:
                keys_in_pipeline = keys_in_pipeline[non_zero_indices]
            else:
                keys_in_pipeline = []

            return keys_in_pipeline, timing, search_key_timing
    return keys_in_pipeline, None, None


# function to get KIT data frame with key, press_time, release_time for a given user
def bbmass_get_dataframe_KIT(data):
    """Input: data  Output: Dataframe with (key, press_time, release_time)"""
    feature_dictionary = {}
    keys_in_pipeline = []
    result_key = []
    press = []
    release = []
    for row_idx in range(len(data)):
        keys_in_pipeline = list(keys_in_pipeline)
        curr_key = data[row_idx][1]
        curr_direction = data[row_idx][2]
        curr_timing = data[row_idx][3]

        if curr_direction == 0:
            keys_in_pipeline.append([curr_key, curr_timing])

        if curr_direction == 1:
            keys_in_pipeline, curr_start, curr_end = bbmass_get_timings_KIT(
                keys_in_pipeline, curr_key, curr_timing
            )
            if curr_start is None:
                continue
            else:
                result_key.append(curr_key)
                press.append(curr_start)
                release.append(curr_end)

    resultant_data_frame = pd.DataFrame(
        list(zip(result_key, press, release)),
        columns=["Key", "Press_Time", "Release_Time"],
    )
    return resultant_data_frame


# get KHT feature based on current key and timing values
def bbmass_get_KHT(keys_in_pipeline, search_key, search_key_timing):
    mask = np.ones(len(keys_in_pipeline))
    keys_in_pipeline = np.asarray(keys_in_pipeline)

    for i, (key, timing) in enumerate(keys_in_pipeline):
        if search_key == key:
            mask[i] = 0
            kht = int(float(search_key_timing)) - int(float(timing))
            non_zero_indices = np.nonzero(mask)
            if len(non_zero_indices) > 0:
                keys_in_pipeline = keys_in_pipeline[non_zero_indices]
            else:
                keys_in_pipeline = []
            return keys_in_pipeline, kht

    return keys_in_pipeline, None


def bbmass_get_KHT_features(data):
    feature_dictionary = {}
    keys_in_pipeline = []

    for row_idx in range(len(data)):
        keys_in_pipeline = list(keys_in_pipeline)
        curr_key = data[row_idx][1]
        curr_direction = data[row_idx][2]
        curr_timing = data[row_idx][3]

        if curr_direction == 0:
            keys_in_pipeline.append([curr_key, curr_timing])

        if curr_direction == 1:
            keys_in_pipeline, curr_kht = bbmass_get_KHT(
                keys_in_pipeline, curr_key, curr_timing
            )
            if curr_kht is None:
                continue
            else:
                if curr_key in list(feature_dictionary.keys()):
                    feature_dictionary[curr_key].append(curr_kht)
                else:
                    feature_dictionary[curr_key] = []
                    feature_dictionary[curr_key].append(curr_kht)

    return feature_dictionary


def bbmass_kht_from_dataframe(df: pd.DataFrame):
    user_data = df.values
    user_feat_dict = bbmass_get_KHT_features(user_data)
    return user_feat_dict


# function to get the advanced word level features of every user
# ignore_keys = ['Key.ctrl', 'Key.shift', 'Key.tab', 'Key.down', 'Key.up', 'Key.left', 'Key.right']
# delimiter_keys = ["Key.space", '.', ',', "Key.enter"]
def bbmas_get_advanced_word_features(processed_data):
    words_in_pipeline = []
    feature_dictionary = {}

    ignore_keys = [
        "LCTRL",
        "RSHIFT",
        "TAB",
        "DOWN",
        "LSHIFT",
        "LEFT",
        "CAPSLOCK",
        "NUM",
    ]
    delimiter_keys = ["SPACE", ".", ",", "RETURN"]

    for row_idx in range(len(processed_data)):
        curr_key = processed_data[row_idx][1]
        curr_press = processed_data[row_idx][2]
        curr_release = processed_data[row_idx][3]

        if curr_key in ignore_keys:
            continue

        if curr_key in delimiter_keys:
            if len(words_in_pipeline) > 0:
                advanced_word_features = bbmas_get_advanced_word_level_features(
                    words_in_pipeline
                )
                key_word = ""
                for char, _, _ in words_in_pipeline:
                    key_word = key_word + str(char)

                if key_word in list(feature_dictionary.keys()):
                    feature_dictionary[key_word].append(advanced_word_features)
                else:
                    feature_dictionary[key_word] = []
                    feature_dictionary[key_word].append(advanced_word_features)
            words_in_pipeline = []
            continue

        if curr_key == "BACKSPACE":
            words_in_pipeline = words_in_pipeline[:-1]
            continue

        words_in_pipeline.append([curr_key, curr_press, curr_release])

    return feature_dictionary


def bbmas_get_all_users_features_advanced_word(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        print("File:", directory + user_file)
        data_frame = pd.read_csv(directory + user_file)
        user_data = data_frame.values
        processed_data = bbmass_get_dataframe_KIT(user_data)
        processed_data = np.c_[np.arange(len(processed_data)), processed_data]
        processed_data = processed_data[np.argsort(processed_data[:, 2])]
        user_feat_dict = bbmas_get_advanced_word_features(processed_data)
        users_feat_dict[i + 1] = user_feat_dict

    return users_feat_dict


# desktop_advanced_word_features = bbmas_get_all_users_features_advanced_word('Desktop/')


################################################################################################################################
# Our Dataset's Word Level Features

""" function to return word level statistics for each extracted word.
These features are as follows:
1) Word hold time
2) Average, Standard Deviation and Median of all key hold times in the word
3) Average, Standard Deviation and Median of all flight 1 features for all digraphs in the word
4) Average, Standard Deviation and Median of all flight 2 features for all digraphs in the word
5) Average, Standard Deviation and Median of all flight 3 features for all digraphs in the word
6) Average, Standard Deviation and Median of all flight 4 features for all digraphs in the word
"""


def get_advanced_word_level_features(words_in_pipeline):
    def get_word_hold(words_in_pipeline):
        if math.isnan(
            int(float(words_in_pipeline[-1][2])) - int(float(words_in_pipeline[0][1]))
        ):
            print("NAN FOUND get_word_hold")
            input()
        return int(float(words_in_pipeline[-1][2])) - int(
            float(words_in_pipeline[0][1])
        )

    def get_avg_std_median_key_hold(words_in_pipeline):
        key_holds = []
        for _, press, release in words_in_pipeline:
            key_holds.append(int(float(release)) - int(float(press)))
        if math.isnan(np.mean(key_holds)):
            print("NAN FOUND get_avg_std_median_key_hold, avg")
            input()

        if math.isnan(np.std(key_holds)):
            print("NAN FOUND get_avg_std_median_key_hold, std")
            input()
        if math.isnan(np.median(key_holds)):
            print("NAN FOUND get_avg_std_median_key_hold, median")
            input()
        return np.mean(key_holds), np.std(key_holds), np.median(key_holds)

    def get_avg_std_median_flights(words_in_pipeline):
        flights_1 = []
        flights_2 = []
        flights_3 = []
        flights_4 = []
        if len(words_in_pipeline) > 1:
            for i in range(len(words_in_pipeline) - 1):
                k1_r = words_in_pipeline[i][2]
                k1_p = words_in_pipeline[i][1]
                k2_r = words_in_pipeline[i + 1][2]
                k2_p = words_in_pipeline[i + 1][1]
                flights_1.append(int(float(k2_p)) - int(float(k1_r)))
                flights_2.append(int(float(k2_r)) - int(float(k1_r)))
                flights_3.append(int(float(k2_p)) - int(float(k1_p)))
                flights_4.append(int(float(k2_r)) - int(float(k1_p)))
        # TODO: What is the best way to handle a case like this,
        # since this means our words_in_pipeline is something like:
        # [['a',1496842340672, 1496842340002 ]]
        # In a case like this, there is no second key to use
        else:
            k1_r = words_in_pipeline[0][2]
            k1_p = words_in_pipeline[0][1]
            k2_r = words_in_pipeline[0][2]
            k2_p = words_in_pipeline[0][1]
            flights_1.append(int(float(k2_p)) - int(float(k1_r)))
            flights_2.append(int(float(k2_r)) - int(float(k1_r)))
            flights_3.append(int(float(k2_p)) - int(float(k1_p)))
            flights_4.append(int(float(k2_r)) - int(float(k1_p)))
        if math.isnan(np.mean(flights_1)):
            print("NAN FOUND get_avg_std_median_flights, avg flights_1")
            print(words_in_pipeline)
            input()
        if math.isnan(np.std(flights_1)):
            print("NAN FOUND get_avg_std_median_flights, std flights_1")
            input()
        if math.isnan(np.median(flights_1)):
            print("NAN FOUND get_avg_std_median_flights, median flights_1")
            input()
        if math.isnan(np.mean(flights_2)):
            print("NAN FOUND get_avg_std_median_flights, avg flights_2")
            input()
        if math.isnan(np.std(flights_2)):
            print("NAN FOUND get_avg_std_median_flights, std flights_2")
            input()
        if math.isnan(np.median(flights_2)):
            print("NAN FOUND get_avg_std_median_flights, median flights_2")
            input()
        if math.isnan(np.mean(flights_3)):
            print("NAN FOUND get_avg_std_median_flights, avg flights_3")
            input()
        if math.isnan(np.std(flights_3)):
            print("NAN FOUND get_avg_std_median_flights, std flights_3")
            input()
        if math.isnan(np.median(flights_3)):
            print("NAN FOUND get_avg_std_median_flights, median flights_3")
            input()
        if math.isnan(np.mean(flights_4)):
            print("NAN FOUND get_avg_std_median_flights, avg flights_4")
            input()
        if math.isnan(np.std(flights_4)):
            print("NAN FOUND get_avg_std_median_flights, std flights_4")
            input()
        if math.isnan(np.median(flights_4)):
            print("NAN FOUND get_avg_std_median_flights, median flights_4")
            input()

        return (
            np.mean(flights_1),
            np.std(flights_1),
            np.median(flights_1),
            np.mean(flights_2),
            np.std(flights_2),
            np.median(flights_2),
            np.mean(flights_3),
            np.std(flights_3),
            np.median(flights_3),
            np.mean(flights_4),
            np.std(flights_4),
            np.median(flights_4),
        )

    wh = get_word_hold(words_in_pipeline)
    avg_kh, std_kh, median_kh = get_avg_std_median_key_hold(words_in_pipeline)
    (
        avg_flight1,
        std_flight1,
        median_flight1,
        avg_flight2,
        std_flight2,
        median_flight2,
        avg_flight3,
        std_flight3,
        median_flight3,
        avg_flight4,
        std_flight4,
        median_flight4,
    ) = get_avg_std_median_flights(words_in_pipeline)
    if math.isnan(wh):
        print("NAN FOUND get_advanced_word_level_features, wh")
        input()
    if math.isnan(avg_kh):
        print("NAN FOUND get_advanced_word_level_features, avg_kh")
        input()
    if math.isnan(std_kh):
        print("NAN FOUND get_advanced_word_level_features, std_kh")
        input()
    if math.isnan(median_kh):
        print("NAN FOUND get_advanced_word_level_features, median_kh")
        input()
    if math.isnan(avg_flight1):
        print("NAN FOUND get_advanced_word_level_features, avg_flight1")
        print(words_in_pipeline)
        input()
    if math.isnan(std_flight1):
        print("NAN FOUND get_advanced_word_level_features, std_flight1")
        input()
    if math.isnan(median_flight1):
        print("NAN FOUND get_advanced_word_level_features, median_flight1")
        input()

    if math.isnan(avg_flight2):
        print("NAN FOUND get_advanced_word_level_features, avg_flight2")
        input()
    if math.isnan(std_flight2):
        print("NAN FOUND get_advanced_word_level_features, std_flight2")
        input()
    if math.isnan(median_flight2):
        print("NAN FOUND get_advanced_word_level_features, median_flight2")
        input()

    if math.isnan(avg_flight3):
        print("NAN FOUND get_advanced_word_level_features, avg_flight3")
        input()
    if math.isnan(std_flight3):
        print("NAN FOUND get_advanced_word_level_features, std_flight3")
        input()
    if math.isnan(median_flight3):
        print("NAN FOUND get_advanced_word_level_features, median_flight3")
        input()

    if math.isnan(avg_flight4):
        print("NAN FOUND get_advanced_word_level_features, avg_flight4")
        input()
    if math.isnan(std_flight4):
        print("NAN FOUND get_advanced_word_level_features, std_flight4")
        input()
    if math.isnan(median_flight4):
        print("NAN FOUND get_advanced_word_level_features, median_flight4")
        input()

    return [
        wh,
        avg_kh,
        std_kh,
        median_kh,
        avg_flight1,
        std_flight1,
        median_flight1,
        avg_flight2,
        std_flight2,
        median_flight2,
        avg_flight3,
        std_flight3,
        median_flight3,
        avg_flight4,
        std_flight4,
        median_flight4,
    ]


# function to get the advanced word level features of every user
def get_advanced_word_features(processed_data):
    words_in_pipeline = []
    feature_dictionary = {}

    ignore_keys = [
        "Key.ctrl",
        "Key.shift",
        "Key.tab",
        "Key.down",
        "Key.up",
        "Key.left",
        "Key.right",
    ]
    delimiter_keys = ["Key.space", ".", ",", "Key.enter"]

    for row_idx in range(len(processed_data)):
        curr_key = processed_data[row_idx][1]
        curr_key = curr_key.replace('"', "")
        curr_press = processed_data[row_idx][2]
        curr_release = processed_data[row_idx][3]

        if curr_key in ignore_keys:
            continue

        if curr_key in delimiter_keys:
            if len(words_in_pipeline) > 0:
                advanced_word_features = get_advanced_word_level_features(
                    words_in_pipeline
                )
                key_word = ""
                for char, _, _ in words_in_pipeline:
                    key_word = key_word + str(char)

                if key_word in list(feature_dictionary.keys()):
                    feature_dictionary[key_word].append(advanced_word_features)
                else:
                    feature_dictionary[key_word] = []
                    feature_dictionary[key_word].append(advanced_word_features)
            words_in_pipeline = []
            continue

        if curr_key == "Key.backspace":
            words_in_pipeline = words_in_pipeline[:-1]
            continue

        words_in_pipeline.append([curr_key, curr_press, curr_release])

    return feature_dictionary


def get_all_users_features_advanced_word(directory):
    users_feat_dict = {}

    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        print("User File:", user_file)
        data_frame = pd.read_csv(directory + user_file)
        user_data = data_frame.values
        processed_data = get_dataframe_KIT(user_data)
        processed_data = np.c_[np.arange(len(processed_data)), processed_data]
        # print(processed_data)
        # input("HANG")
        # print(processed_data)
        # input("DATA")
        # processed_data = processed_data[np.argsort(processed_data[:, 2])]
        # print(processed_data)
        # input("HANG")
        user_feat_dict = get_advanced_word_features(processed_data)
        users_feat_dict[i + 1] = user_feat_dict

    return users_feat_dict


def generate_word_level_df(use_bbmass: bool = True):
    if use_bbmass:
        desktop_advanced_word_features = bbmas_get_all_users_features_advanced_word(
            "Desktop/"
        )
    else:
        # TEMP - change dir
        desktop_advanced_word_features = get_all_users_features_advanced_word(
            "Facebook/"
        )
    return pd.DataFrame.from_dict(desktop_advanced_word_features)


# df = generate_word_level_df(False)
# print(df.isnull().sum().sum())

################################################################################################################################
# Word Level Dataframe Feature Generation


def flatten(l):
    return [item for sublist in l for item in sublist]


def unique_word_level_words(use_bbmass: bool = True):
    word_features = []
    words = []
    curr_dict_index = 1
    final_words_set = set()
    if use_bbmass:
        desktop_advanced_word_features = bbmas_get_all_users_features_advanced_word(
            "Desktop/"
        )
    else:
        # TEMP - change dir
        desktop_advanced_word_features = get_all_users_features_advanced_word(
            "Facebook/"
        )
    final_features_dict = defaultdict(list)
    user_ids = list(desktop_advanced_word_features.keys())
    for user_id in user_ids:
        word_features.append(desktop_advanced_word_features[user_id])
    unique_words = []
    for word_dict in word_features:
        words.append(list(set(list(word_dict.keys()))))
    for word_chunk in words:
        for word in word_chunk:
            final_words_set.add(word)
    words = list(final_words_set)
    return words


def get_all_word_level_words():
    # XXX: MAKE SURE THAT IF WE USE BBMAS AGAIN WE UNCOMMENT THIS LINE
    # bbmas_words = set(unique_word_level_words(True))
    fp_words = set(unique_word_level_words(False))
    return list(fp_words)


def generate_word_level_features_df(use_bbmass: bool = True):
    word_features = []
    words = []
    curr_dict_index = 1
    final_words_set = set()
    if use_bbmass:
        desktop_advanced_word_features = bbmas_get_all_users_features_advanced_word(
            "Desktop/"
        )
    else:
        # TEMP - change dir
        desktop_advanced_word_features = get_all_users_features_advanced_word(
            "Facebook/"
        )
    final_features_dict = defaultdict(list)
    user_ids = list(desktop_advanced_word_features.keys())
    for user_id in user_ids:
        word_features.append(desktop_advanced_word_features[user_id])
    words = get_all_word_level_words()
    for word in words:
        for word_feature_dict_element in word_features:
            if word_feature_dict_element.get(word) != None:
                word_features_timings = flatten(word_feature_dict_element.get(word))
                final_features_dict[word + "_word_hold"].append(
                    float(word_features_timings[0])
                )
                final_features_dict[word + "_mean_key_hold"].append(
                    float(word_features_timings[1])
                )
                final_features_dict[word + "_std_key_hold"].append(
                    float(word_features_timings[2])
                )
                final_features_dict[word + "_median_key_hold"].append(
                    float(word_features_timings[3])
                )
                final_features_dict[word + "_mean_flight1"].append(
                    float(word_features_timings[4])
                )
                final_features_dict[word + "_std_flight1"].append(
                    float(word_features_timings[5])
                )
                final_features_dict[word + "_median_flight1"].append(
                    float(word_features_timings[6])
                )
                final_features_dict[word + "_mean_flight2"].append(
                    float(word_features_timings[7])
                )
                final_features_dict[word + "_std_flight2"].append(
                    float(word_features_timings[8])
                )
                final_features_dict[word + "_median_flight2"].append(
                    float(word_features_timings[9])
                )
                final_features_dict[word + "_mean_flight3"].append(
                    float(word_features_timings[10])
                )
                final_features_dict[word + "_std_flight3"].append(
                    float(word_features_timings[11])
                )
                final_features_dict[word + "_median_flight3"].append(
                    float(word_features_timings[12])
                )
                final_features_dict[word + "_mean_flight4"].append(
                    float(word_features_timings[13])
                )
                final_features_dict[word + "_std_flight4"].append(
                    float(word_features_timings[14])
                )
                final_features_dict[word + "_median_flight4"].append(
                    float(word_features_timings[15])
                )
            else:
                final_features_dict[word + "_word_hold"].append(float(0))
                final_features_dict[word + "_mean_key_hold"].append(float(0))
                final_features_dict[word + "_std_key_hold"].append(float(0))
                final_features_dict[word + "_median_key_hold"].append(float(0))
                final_features_dict[word + "_mean_flight1"].append(float(0))
                final_features_dict[word + "_std_flight1"].append(float(0))
                final_features_dict[word + "_median_flight1"].append(float(0))
                final_features_dict[word + "_mean_flight2"].append(float(0))
                final_features_dict[word + "_std_flight2"].append(float(0))
                final_features_dict[word + "_median_flight2"].append(float(0))
                final_features_dict[word + "_mean_flight3"].append(float(0))
                final_features_dict[word + "_std_flight3"].append(float(0))
                final_features_dict[word + "_median_flight3"].append(float(0))
                final_features_dict[word + "_mean_flight4"].append(float(0))
                final_features_dict[word + "_std_flight4"].append(float(0))
                final_features_dict[word + "_median_flight4"].append(float(0))

    word_level_features_df = pd.DataFrame(final_features_dict)
    # word_level_features_df = word_level_features_df.drop(columns=word_level_features_df.columns[word_level_features_df.eq(0).mean()>0.5])
    return word_level_features_df


# df = generate_word_level_features_df(False)
# print(df)

################################################################################################################################
# Statistics Features
def flatten(l):
    return [item for sublist in l for item in sublist]


def get_min_timing(data):
    if math.isnan(min(data)):
        print("NAN Found in get_min_timing")
    return min(data)


def get_max_timing(data):
    if math.isnan(max(data)):
        print("NAN Found in get_max_timing")
    return max(data)


def first_quartile(data):
    if math.isnan(np.quantile(data, 0.25)):
        print("NAN Found in first_quartile")
    return np.quantile(data, 0.25)


def third_quartile(data):
    if math.isnan(np.quantile(data, 0.75)):
        print("NAN Found in third_quartile")
    return np.quantile(data, 0.75)


def get_mean_timings(data):
    if math.isnan(sum(data) / len(data)):
        print("NAN Found in get_mean_timings")
    mean_value = sum(data) / len(data)
    return mean_value


def get_median_timing(data):
    if math.isnan(statistics.median(data)):
        print("NAN Found in get_median_timings")
    return statistics.median(data)


def get_mode_timing(data):
    return statistics.mode(data)


def get_standard_deviation_timing(data):
    if math.isnan(statistics.stdev(data)):
        print("NAN Found in get_standard_deviation_timing")
    return statistics.stdev(data)


def get_stistical_features(data):
    mean_value = get_mean_timings(data)
    median_value = get_median_timing(data)
    mode_value = get_mode_timing(data)
    stdev_value = get_standard_deviation_timing(data)
    return (mean_value, median_value, mode_value, stdev_value)


################################################################################################################################
# Keystroke Feature Generators

# In the subsequent cells are utility functions to take the training dataset and
# generate KHT, KIT, and combined dataframes.
# Each of these, or the combined dfs can be put into the machine larning pipelines
# The combined simply does a concatenation of the KHT and KIT dfs

# TEMP - to acccount for the change in how are files are made in the facebook
# direcotry we remove the session from the df columns and the dictionary
# and change how we look at the ID's and get the associated gender information
# by using os.path.basename() to strip the extension
def generate_kht_features_df(use_train=True):
    demographics = make_gender_df()
    # TEMP - change dir
    if use_train:
        dir = "Facebook/"
    else:
        dir = "Desktop/"
    final_df = pd.DataFrame(
        columns=[
            "ID",
            "Gender",
            "Platform",
            "Key(s)",
            "Mean",
            "Median",
            "Standard Deviation",
            "First Quartile",
            "Third Quartile",
            "Class",
        ]
    )
    user_id = 1
    holder = []
    classification = 0
    user_files = os.listdir(dir)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        if ".csv" in user_file and not user_file.startswith("."):
            path = dir + user_file
            try:
                df = pd.read_csv(path, header=None)
            except pd.errors.EmptyDataError:
                print(user_file, "is empty")
                continue
            data = split_into_four(df)
            df = remove_invalid_keystrokes_from_data_list(data)
            user_data_kht = kht_from_dataframe(df)
            print(user_file)
            # print(user_data_kht)
            for key, timings in user_data_kht.items():
                time_mean = get_mean_timings(timings)
                time_median = get_median_timing(timings)
                fquartile = first_quartile(timings)
                tquartile = third_quartile(timings)
                try:
                    time_stdev = get_standard_deviation_timing(timings)
                except statistics.StatisticsError:
                    # print("ERROR: Less than 2 points to make a standard deviation variance")
                    time_stdev = 0
                user_id = path_to_id(user_file)
                platform = path_to_platform(user_file)
                # session = path_to_session(user_file)[0]
                classification = path_to_class(user_file)
                gender = demographics.loc[
                    demographics["ID"] == int(os.path.splitext(user_id)[0])
                ].iloc[0]["Gender"]
                temp = {
                    "ID": int(os.path.splitext(user_id)[0]),
                    "Gender": [gender],
                    "Platform": [platform],
                    # "Session":[session],
                    "Key(s)": [key],
                    "Mean": [float(time_mean)],
                    "Median": [float(time_median)],
                    "Standard Deviation": [float(time_stdev)],
                    "First Quartile": [float(fquartile)],
                    "Third Quartile": [float(tquartile)],
                    "Class": [classification],
                }
                # print(temp)

                partial = pd.DataFrame.from_dict(temp)
                # display(partial)
                final_df = final_df.append(partial, ignore_index=True)
    # print(final_df)
    # final_df = final_df.drop(final_df[final_df["Standard Deviation"]==0].index)
    # print(final_df)
    # input()
    return final_df


# TODO: For f_29 the values are correct in the raw file however, when looking
# at the file it is rounded off so the kht values are all 0
# I think this is also happening to some of the other files

# TEMP - to acccount for the change in how are files are made in the facebook
# direcotry we remove the session from the df columns and the dictionary
# and change how we look at the ID's and get the associated gender information
# by using os.path.basename() to strip the extension


def generate_kit_features_df(use_train=True):
    demographics = make_gender_df()
    if use_train:
        dir = "Facebook/"
    else:
        dir = "Test/"
    f1 = pd.DataFrame()
    f2 = pd.DataFrame()
    f3 = pd.DataFrame()
    f4 = pd.DataFrame()
    user_id = 1
    directory = dir
    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        if ".csv" in user_file and not user_file.startswith("."):
            path = directory + user_file
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                print(user_file, "is empty")
                input("Empty File")
                continue
            user_data_kit_f1 = kit_from_dataframe(df, 1)
            user_data_kit_f2 = kit_from_dataframe(df, 2)
            user_data_kit_f3 = kit_from_dataframe(df, 3)
            user_data_kit_f4 = kit_from_dataframe(df, 4)
            for key, timings in user_data_kit_f1.items():
                time_mean = get_mean_timings(timings)
                if math.isnan(time_mean):
                    print("NAN FOUND: f1 time_mean, generate_kit_features_df")

                time_median = get_median_timing(timings)
                if math.isnan(time_median):
                    print("NAN FOUND: f1 time_median, generate_kit_features_df")

                fquartile = first_quartile(timings)
                if math.isnan(fquartile):
                    print("NAN FOUND: f1 fquartile, generate_kit_features_df")

                tquartile = third_quartile(timings)
                if math.isnan(tquartile):
                    print("NAN FOUND: f1 tquartile, generate_kit_features_df")
                try:
                    time_stdev = get_standard_deviation_timing(timings)
                except statistics.StatisticsError:
                    # print("ERROR: Less than 2 points to make a standard deviation variance")
                    time_stdev = 0
                user_id = path_to_id(user_file)
                gender = demographics.loc[
                    demographics["ID"] == int(os.path.splitext(user_id)[0])
                ].iloc[0]["Gender"]
                os.path.splitext(user_id)[0]
                temp = {
                    "ID": [user_id],
                    "Gender": [gender],
                    "Key(s)": [key],
                    "F1_Mean": [time_mean],
                    "F1_Median": [time_median],
                    "F1_Standard Deviation": [time_stdev],
                    "F1_First Quartile": [fquartile],
                    "F1_Third Quartile": [tquartile],
                }
                # print(temp)
                partial = pd.DataFrame.from_dict(temp)
                # display(partial)
                f1 = f1.append(partial, ignore_index=True)

            for key, timings in user_data_kit_f2.items():
                time_mean = get_mean_timings(timings)
                if math.isnan(time_mean):
                    print("NAN FOUND: f2 time_mean, generate_kit_features_df")

                time_median = get_median_timing(timings)
                if math.isnan(time_median):
                    print("NAN FOUND: f2 time_median, generate_kit_features_df")

                fquartile = first_quartile(timings)
                if math.isnan(fquartile):
                    print("NAN FOUND: f2 fquartile, generate_kit_features_df")

                tquartile = third_quartile(timings)
                if math.isnan(tquartile):
                    print("NAN FOUND: f2 tquartile, generate_kit_features_df")
                try:
                    time_stdev = get_standard_deviation_timing(timings)
                except statistics.StatisticsError:
                    # print("ERROR: Less than 2 points to make a standard deviation variance")
                    time_stdev = 0
                temp = {
                    # "ID":[user_id],
                    # "Gender": [gender],
                    # "Key(s)":[key],
                    "F2_Mean": [time_mean],
                    "F2_Median": [time_median],
                    "F2_Standard Deviation": [time_stdev],
                    "F2_First Quartile": [fquartile],
                    "F2_Third Quartile": [tquartile],
                }
                # print(temp)
                partial = pd.DataFrame.from_dict(temp)
                # display(partial)
                f2 = f2.append(partial, ignore_index=True)
            for key, timings in user_data_kit_f3.items():
                time_mean = get_mean_timings(timings)
                if math.isnan(time_mean):
                    print("NAN FOUND: f3 time_mean, generate_kit_features_df")

                time_median = get_median_timing(timings)
                if math.isnan(time_median):
                    print("NAN FOUND: f3 time_median, generate_kit_features_df")

                fquartile = first_quartile(timings)
                if math.isnan(fquartile):
                    print("NAN FOUND: f3 fquartile, generate_kit_features_df")

                tquartile = third_quartile(timings)
                if math.isnan(tquartile):
                    print("NAN FOUND: f3 tquartile, generate_kit_features_df")
                try:
                    time_stdev = get_standard_deviation_timing(timings)
                except statistics.StatisticsError:
                    # print("ERROR: Less than 2 points to make a standard deviation variance")
                    time_stdev = 0
                temp = {
                    # "ID":[user_id],
                    # "Gender": [gender],
                    # "Key(s)":[key],
                    "F3_Mean": [time_mean],
                    "F3_Median": [time_median],
                    "F3_Standard Deviation": [time_stdev],
                    "F3_First Quartile": [fquartile],
                    "F3_Third Quartile": [tquartile],
                }
                # print(temp)

                partial = pd.DataFrame.from_dict(temp)
                # display(partial)
                f3 = f3.append(partial, ignore_index=True)

            for key, timings in user_data_kit_f4.items():
                time_mean = get_mean_timings(timings)
                if math.isnan(time_mean):
                    print("NAN FOUND: f4 time_mean, generate_kit_features_df")

                time_median = get_median_timing(timings)
                if math.isnan(time_median):
                    print("NAN FOUND: f4 time_median, generate_kit_features_df")

                fquartile = first_quartile(timings)
                if math.isnan(fquartile):
                    print("NAN FOUND: f4 fquartile, generate_kit_features_df")

                tquartile = third_quartile(timings)
                if math.isnan(tquartile):
                    print("NAN FOUND: f4 tquartile, generate_kit_features_df")
                try:
                    time_stdev = get_standard_deviation_timing(timings)
                except statistics.StatisticsError:
                    # print("ERROR: Less than 2 points to make a standard deviation variance")
                    time_stdev = 0
                temp = {
                    # "ID":[user_id],
                    # "Gender": [gender],
                    # "Key(s)":[key],
                    "F4_Mean": [time_mean],
                    "F4_Median": [time_median],
                    "F4_Standard Deviation": [time_stdev],
                    "F4_First Quartile": [fquartile],
                    "F4_Third Quartile": [tquartile],
                }
                # print(temp)
                partial = pd.DataFrame.from_dict(temp)
                # display(partial)
                f4 = f4.append(partial, ignore_index=True)
    # f1 = f1.drop(f1[f1["F1_Standard Deviation"]==0].index)
    # f2 = f2.drop(f2[f2["F2_Standard Deviation"]==0].index)
    # f3 = f3.drop(f3[f3["F3_Standard Deviation"]==0].index)
    # f4 = f4.drop(f4[f4["F4_Standard Deviation"]==0].index)
    final_df = pd.concat([f1, f2, f3, f4], axis=1)
    return final_df


# df = generate_kit_features_df()
# print(df.shape)


def bbmass_generate_kht_features_df():
    final_df = pd.DataFrame(
        columns=[
            "ID",
            "Key(s)",
            "Mean",
            "Median",
            "Standard Deviation",
            "First Quartile",
            "Third Quartile",
            "Gender",
        ]
    )
    bbmass_demographics = make_bbmass_gender_df()
    dir = "Desktop/"
    user_files = os.listdir(dir)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        if ".csv" in user_file and not user_file.startswith("."):
            path = dir + user_file
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                print(user_file, "is empty")
                continue
            user_data_kht = bbmass_kht_from_dataframe(df)
            for key, timings in user_data_kht.items():
                time_mean = get_mean_timings(timings)
                time_median = get_median_timing(timings)
                fquartile = first_quartile(timings)
                tquartile = third_quartile(timings)
                try:
                    time_stdev = get_standard_deviation_timing(timings)
                except statistics.StatisticsError:
                    # print("ERROR: Less than 2 points to make a standard deviation variance")
                    time_stdev = 0
                user_id = bbmass_path_to_id(user_file)
                gender = bbmass_demographics.loc[
                    bbmass_demographics["User ID"] == int(user_id)
                ].iloc[0]["Gender"]
                temp = {
                    "ID": [user_id],
                    "Key(s)": [key],
                    "Mean": [time_mean],
                    "Median": [time_median],
                    "Standard Deviation": [time_stdev],
                    "First Quartile": [fquartile],
                    "Third Quartile": [tquartile],
                    "Gender": [gender],
                }
                partial = pd.DataFrame.from_dict(temp)
                # display(partial)
                final_df = final_df.append(partial, ignore_index=True)
    final_df = final_df.drop(final_df[final_df["Standard Deviation"] == 0].index)
    return final_df


# df = bbmass_generate_kht_features_df()

# function to get Flight1 KIT feature dictionary for a given user
def bbmass_get_KIT_features_F1(data):
    """Input: keystroke data, Output: Dictionary of (next_key_press - current_key_release)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][2]
        next_timing = data[row_idx + 1][1]

        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


# function to get Flight2 KIT feature dictionary for a given user
def bbmass_get_KIT_features_F2(data):
    """Input: keystroke data, Output: Dictionary of (next_key_press - current_key_press)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][1]
        next_timing = data[row_idx + 1][1]
        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


# function to get Flight3 KIT feature dictionary for a given user
def bbmass_get_KIT_features_F3(data):
    """Input: keystroke data, Output: Dictionary of (next_key_release - current_key_release)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][2]
        next_timing = data[row_idx + 1][2]
        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


# function to get Flight3 KIT feature dictionary for a given user
def bbmass_get_KIT_features_F4(data):
    """Input: keystroke data, Output: Dictionary of (next_key_release - current_key_press)"""
    feature_dictionary = {}

    for row_idx in range(0, len(data)):
        curr_key = data[row_idx][0]
        if row_idx + 1 >= len(data):
            break
        next_key = data[row_idx + 1][0]
        curr_timing = data[row_idx][1]
        next_timing = data[row_idx + 1][2]
        if str(curr_key) + str(next_key) in list(feature_dictionary.keys()):
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )
        else:
            feature_dictionary[str(curr_key) + str(next_key)] = []
            feature_dictionary[str(curr_key) + str(next_key)].append(
                int(float(next_timing)) - int(float(curr_timing))
            )

    return feature_dictionary


def bbmass_kit_from_dataframe(df: pd.DataFrame, kit_feature_index: int):
    df = bbmass_get_dataframe_KIT(df.values)
    user_data = df.values
    assert 1 <= kit_feature_index <= 4
    if kit_feature_index == 1:
        return bbmass_get_KIT_features_F1(user_data)
    elif kit_feature_index == 2:
        return bbmass_get_KIT_features_F2(user_data)
    elif kit_feature_index == 3:
        return bbmass_get_KIT_features_F3(user_data)
    elif kit_feature_index == 4:
        return bbmass_get_KIT_features_F4(user_data)


def bbmass_generate_kit_features_df():
    bbmass_demographics = make_bbmass_gender_df()
    # TODO:
    dir = "Desktop/"
    f1 = pd.DataFrame()
    f2 = pd.DataFrame()
    f3 = pd.DataFrame()
    f4 = pd.DataFrame()
    user_id = 1
    directory = dir
    user_files = os.listdir(directory)
    for i in tqdm(range(len(user_files))):
        user_file = user_files[i]
        if ".csv" in user_file and not user_file.startswith("."):
            path = directory + user_file
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                print(user_file, "is empty")
                continue
            user_data_kit_f1 = bbmass_kit_from_dataframe(df, 1)
            user_data_kit_f2 = bbmass_kit_from_dataframe(df, 2)
            user_data_kit_f3 = bbmass_kit_from_dataframe(df, 3)
            user_data_kit_f4 = bbmass_kit_from_dataframe(df, 4)
            for key, timings in user_data_kit_f1.items():
                time_mean = get_mean_timings(timings)
                if math.isnan(time_mean):
                    print("NAN FOUND: f1 time_mean, bbmass_generate_kit_features_df")
                time_median = get_median_timing(timings)
                if math.isnan(time_median):
                    print("NAN FOUND: f1 time_median, bbmass_generate_kit_features_df")
                fquartile = first_quartile(timings)
                if math.isnan(fquartile):
                    print("NAN FOUND: f1 fquartile, bbmass_generate_kit_features_df")
                tquartile = third_quartile(timings)
                if math.isnan(tquartile):
                    print("NAN FOUND: f1 tquartile, bbmass_generate_kit_features_df")
                try:
                    time_stdev = get_standard_deviation_timing(timings)
                except statistics.StatisticsError:
                    # print("ERROR: Less than 2 points to make a standard deviation variance")
                    time_stdev = 0
                user_id = bbmass_path_to_id(user_file)
                gender = bbmass_demographics.loc[
                    bbmass_demographics["User ID"] == int(user_id)
                ].iloc[0]["Gender"]
                temp = {
                    "ID": [user_id],
                    "Gender": [gender],
                    "Key(s)": [key],
                    "F1_Mean": [float(time_mean)],
                    "F1_Median": [float(time_median)],
                    "F1_Standard Deviation": [float(time_stdev)],
                    "F1_First Quartile": [float(fquartile)],
                    "F1_Third Quartile": [float(tquartile)],
                }
                # print(temp)
                partial = pd.DataFrame.from_dict(temp)
                # display(partial)
                f1 = f1.append(partial, ignore_index=True)
            for key, timings in user_data_kit_f2.items():
                time_mean = get_mean_timings(timings)
                time_median = get_median_timing(timings)
                fquartile = first_quartile(timings)
                tquartile = third_quartile(timings)
                if (
                    math.isnan(time_mean)
                    or math.isnan(time_median)
                    or math.isnan(fquartile)
                    or math.isnan(tquartile)
                ):
                    print("NAN FOUND")
                    print(timings)
                    input()
                try:
                    time_stdev = get_standard_deviation_timing(timings)
                except statistics.StatisticsError:
                    # print("ERROR: Less than 2 points to make a standard deviation variance")
                    time_stdev = 0
                user_id = bbmass_path_to_id(user_file)
                gender = bbmass_demographics.loc[
                    bbmass_demographics["User ID"] == int(user_id)
                ].iloc[0]["Gender"]
                temp = {
                    # "ID":[user_id],
                    # "Gender": [gender],
                    # "Key(s)":[key],
                    "F2_Mean": [float(time_mean)],
                    "F2_Median": [float(time_median)],
                    "F2_Standard Deviation": [float(time_stdev)],
                    "F2_First Quartile": [float(fquartile)],
                    "F2_Third Quartile": [float(tquartile)],
                }
                # print(temp)
                partial = pd.DataFrame.from_dict(temp)
                # display(partial)
                f2 = f2.append(partial, ignore_index=True)
            for key, timings in user_data_kit_f3.items():
                time_mean = get_mean_timings(timings)
                time_median = get_median_timing(timings)
                fquartile = first_quartile(timings)
                tquartile = third_quartile(timings)
                if (
                    math.isnan(time_mean)
                    or math.isnan(time_median)
                    or math.isnan(fquartile)
                    or math.isnan(tquartile)
                ):
                    print("NAN FOUND")
                    print(timings)
                    input()
                try:
                    time_stdev = get_standard_deviation_timing(timings)
                except statistics.StatisticsError:
                    # print("ERROR: Less than 2 points to make a standard deviation variance")
                    time_stdev = 0
                user_id = bbmass_path_to_id(user_file)
                gender = bbmass_demographics.loc[
                    bbmass_demographics["User ID"] == int(user_id)
                ].iloc[0]["Gender"]
                temp = {
                    # "ID":[user_id],
                    # "Gender": [gender],
                    # "Key(s)":[key],
                    "F3_Mean": [float(time_mean)],
                    "F3_Median": [float(time_median)],
                    "F3_Standard Deviation": [float(time_stdev)],
                    "F3_First Quartile": [float(fquartile)],
                    "F3_Third Quartile": [float(tquartile)],
                }
                # print(temp)

                partial = pd.DataFrame.from_dict(temp)
                # display(partial)
                f3 = f3.append(partial, ignore_index=True)

            for key, timings in user_data_kit_f4.items():
                time_mean = get_mean_timings(timings)
                time_median = get_median_timing(timings)
                fquartile = first_quartile(timings)
                tquartile = third_quartile(timings)
                if (
                    math.isnan(time_mean)
                    or math.isnan(time_median)
                    or math.isnan(fquartile)
                    or math.isnan(tquartile)
                ):
                    print("NAN FOUND")
                    print(timings)
                    input()
                try:
                    time_stdev = get_standard_deviation_timing(timings)
                except statistics.StatisticsError:
                    # print("ERROR: Less than 2 points to make a standard deviation variance")
                    time_stdev = 0
                user_id = bbmass_path_to_id(user_file)
                gender = bbmass_demographics.loc[
                    bbmass_demographics["User ID"] == int(user_id)
                ].iloc[0]["Gender"]
                temp = {
                    # "ID":[user_id],
                    # "Gender": [gender],
                    # "Key(s)":[key],
                    "F4_Mean": [float(time_mean)],
                    "F4_Median": [float(time_median)],
                    "F4_Standard Deviation": [float(time_stdev)],
                    "F4_First Quartile": [float(fquartile)],
                    "F4_Third Quartile": [float(tquartile)],
                }
                # print(temp)
                partial = pd.DataFrame.from_dict(temp)
                # display(partial)
                f4 = f4.append(partial, ignore_index=True)
    # f1 = f1.drop(f1[f1["F1_Standard Deviation"]==0].index)
    # f2 = f2.drop(f2[f2["F2_Standard Deviation"]==0].index)
    # f3 = f3.drop(f3[f3["F3_Standard Deviation"]==0].index)
    # f4 = f4.drop(f4[f4["F4_Standard Deviation"]==0].index)
    final_df = pd.concat([f1, f2, f3, f4], axis=1)
    print("NANs Found:", final_df.isnull().sum().sum())
    # final_df[['ID']] = final_df[['ID']].fillna(99)
    # final_df[['Gender']] = final_df[['Gender']].fillna("OTHER")
    # final_df[['Key(s)']] = final_df[['Key(s)']].fillna("FILL")
    # final_df[['F1_Mean', 'F1_Median', 'F1_Standard Deviation', 'F1_First Quartile', 'F1_Third Quartile', 'F2_Mean', 'F2_Median', 'F2_Standard Deviation', 'F2_First Quartile', 'F2_Third Quartile', 'F3_Mean', 'F3_Median', 'F3_Standard Deviation', 'F3_First Quartile', 'F3_Third Quartile', 'F4_Mean', 'F4_Median', 'F4_Standard Deviation', 'F4_First Quartile', 'F4_Third Quartile']] = final_df[['F1_Mean', 'F1_Median', 'F1_Standard Deviation', 'F1_First Quartile', 'F1_Third Quartile', 'F2_Mean', 'F2_Median', 'F2_Standard Deviation', 'F2_First Quartile', 'F2_Third Quartile', 'F3_Mean', 'F3_Median', 'F3_Standard Deviation', 'F3_First Quartile', 'F3_Third Quartile', 'F4_Mean', 'F4_Median', 'F4_Standard Deviation', 'F4_First Quartile', 'F4_Third Quartile']].fillna(99.9999)
    return final_df


# df = bbmass_generate_kit_features_df()


def get_unique_keys(feature_type: FeatureType):
    if feature_type == FeatureType.KHT:
        # os.chdir("/content/drive/My Drive/Fake_Profile/")
        if os.path.exists("bbmass_kht_df.pkl"):
            with open("bbmass_kht_df.pkl", "rb") as f:
                df = pickle.load(f)
        else:
            df = bbmass_generate_kht_features_df()
            # os.chdir("/content/drive/My Drive/Fake_Profile/")
            with open("bbmass_kht_df.pkl", "wb") as f:
                pickle.dump(df, f)
        # os.chdir("/content/drive/My Drive/Fake_Profile/")
        if os.path.exists("ident_kht_df.pkl"):
            # with open("ident_kht_df.pkl", "rb") as f:
            #   fp_df = pickle.load(f)
            pass
        else:
            fp_df = generate_kht_features_df()
            # os.chdir('/content/drive/My Drive/Fake_Profile/')
            # with open("ident_kht_df.pkl", "wb") as f:
            #   pickle.dump(fp_df, f)

    elif feature_type == FeatureType.KIT:
        # os.chdir("/content/drive/My Drive/Fake_Profile/")
        if os.path.exists("bbmass_kit_df.pkl"):
            with open("bbmass_kit_df.pkl", "rb") as f:
                df = pickle.load(f)
        else:
            df = bbmass_generate_kit_features_df()
            # os.chdir("/content/drive/My Drive/Fake_Profile/")
            with open("bbmass_kit_df.pkl", "wb") as f:
                pickle.dump(df, f)
        # os.chdir("/content/drive/My Drive/Fake_Profile/")
        if os.path.exists("ident_kit_df.pkl"):
            # with open("ident_kit_df.pkl", "rb") as f:
            #   fp_df = pickle.load(f)
            pass
        else:
            fp_df = generate_kit_features_df()
            # os.chdir('/content/drive/My Drive/Fake_Profile/')
            # with open("ident_kit_df.pkl", "wb") as f:
            #   pickle.dump(fp_df, f)

    elif feature_type == FeatureType.WORD_LEVEL:
        return list(get_all_word_level_words())
    else:
        raise ValueError("Unknown enum variant of FeatureType")
    print("Invalid data found in df:", df.isnull().sum().sum())
    print("Invalid data found in fp_df:", fp_df.isnull().sum().sum())

    unique_keys = df["Key(s)"].unique()
    fp_unique = fp_df["Key(s)"].unique()
    final_unique_keys = set()
    for k in unique_keys:
        final_unique_keys.add(k)
    # for k in fp_unique:
    #   final_unique_keys.add(k)
    return list(final_unique_keys)


def fake_profile_kht_feature_vector():
    kht_df = generate_kht_features_df()
    kht_unique = get_unique_keys(FeatureType.KHT)
    unique_ids = list(kht_df["ID"].unique())
    print(len(kht_unique))
    features_vector = defaultdict(list)
    for id in unique_ids:
        rows = kht_df.loc[kht_df["ID"] == id]
        # print(rows)
        # input()
        for key in kht_unique:
            rows_for_key = rows.loc[rows["Key(s)"] == key]
            if rows_for_key.empty:
                # print("No data for key", key, "for id:", id)
                features_vector[key].append([0, 0, 0, 0, 0, 0])
            else:
                for idx, row in rows_for_key.iterrows():
                    features_vector[key].append(
                        [
                            row["Mean"],
                            row["Median"],
                            row["Standard Deviation"],
                            row["First Quartile"],
                            row["Third Quartile"],
                            row["Gender"],
                        ]
                    )
                # print(features_vector)
    return features_vector


def fake_profile_kit_feature_vector():
    kit_df = generate_kit_features_df()
    kit_unique = get_unique_keys(FeatureType.KIT)
    unique_ids = list(kit_df["ID"].unique())
    features_vector = defaultdict(list)
    for id in unique_ids:
        rows = kit_df.loc[kit_df["ID"] == id]
        # print(rows)
        # input()
        for key in kit_unique:
            rows_for_key = rows.loc[rows["Key(s)"] == key]
            if rows_for_key.empty:
                # print("No data for key", key, "for id:", id)
                features_vector[key].append(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                )
            else:
                for idx, row in rows_for_key.iterrows():
                    # print(row)
                    features_vector[key].append(
                        [
                            row["F1_Mean"],
                            row["F1_Median"],
                            row["F1_Standard Deviation"],
                            row["F1_First Quartile"],
                            row["F1_Third Quartile"],
                            row["F2_Mean"],
                            row["F2_Median"],
                            row["F2_Standard Deviation"],
                            row["F2_First Quartile"],
                            row["F2_Third Quartile"],
                            row["F3_Mean"],
                            row["F3_Median"],
                            row["F3_Standard Deviation"],
                            row["F3_First Quartile"],
                            row["F3_Third Quartile"],
                            row["F4_Mean"],
                            row["F4_Median"],
                            row["F4_Standard Deviation"],
                            row["F4_First Quartile"],
                            row["F4_Third Quartile"],
                        ]
                    )
                # print(features_vector)
    return features_vector


# fp_kit_feature_vector = fake_profile_kit_feature_vector()


def find_max_feature_set_length(value_set):
    current_max_length = 0
    for feature_set in value_set:
        if len(feature_set) > current_max_length:
            current_max_length = len(feature_set)
    current_max_length += 1
    return current_max_length


def pad_list(l, target_len):
    return l[:target_len] + [0] * (target_len - len(l))


def pad_feature_vector(fp_kht_feature_dict):
    value_set = list(fp_kht_feature_dict.values())
    max_feature_length = find_max_feature_set_length(value_set)
    print("Max length is:", max_feature_length)
    kht_keys = list(fp_kht_feature_dict.keys())
    for key in kht_keys:
        nested_value_list = fp_kht_feature_dict.get(key)
        if len(nested_value_list) < max_feature_length:
            nested_value_list += [0] * (max_feature_length - len(nested_value_list))


def pad_kit_feature_sets(fp_kit_dict):
    kit_keys = list(fp_kit_dict.keys())
    print(len(kit_keys))
    holder = defaultdict(list)
    max_feature_vector_length = 0
    for key in kit_keys:
        feature_set = fp_kit_dict[key]
        # print(len(feature_set))
        if len(feature_set) > max_feature_vector_length:
            max_feature_vector_length = len(feature_set)
    for key in kit_keys:
        feature_set = fp_kit_dict[key]
        if len(feature_set) < max_feature_vector_length:
            pad_list(feature_set, max_feature_vector_length)
    return fp_kit_dict


def fake_profile_kht_into_df(fp_kht_dict):
    kht_keys = list(fp_kht_dict.keys())
    holder = defaultdict(list)
    for key in kht_keys:
        feature_set = fp_kht_dict.get(key)
        # print(feature_set)
        for individual_features in feature_set:
            # print("Length of vector:", individual_features)
            # input()
            holder[key + "mean"].append(individual_features[0])
            holder[key + "median"].append(individual_features[1])
            holder[key + "stdev"].append(individual_features[2])
            holder[key + "first_quartile"].append(individual_features[3])
            holder[key + "third_quartile"].append(individual_features[4])
        # print(individual_features)
    pad_feature_vector(holder)
    return pd.DataFrame(holder)


def fake_profile_kit_into_df(fp_kit_dict):
    holder = defaultdict(list)
    fp_kit_dict = pad_kit_feature_sets(fp_kit_dict)
    kit_keys = list(fp_kit_dict.keys())
    for key in kit_keys:
        feature_set = fp_kit_dict[key]
        # assert len(feature_set) == 130
        # print(feature_set)
        # TODO: The assertion here fails because not all of the feature vectors are
        # the same length. I think to solve this we need to create a seperate
        # function that precomputes the maximum feature_set length, before we loop,
        # and pads any non-matching sets
        # Is there similar behavior in the bbmas version that the padding at the
        # very end before returning solved there but not here?
        # TODO: The resultant df is 130 rows when it should be 132 rows since
        # that's the number of files processed. Maybe it has something to do with
        # how our dataset files are distributed ? After finalizing the dataset
        # and recreating the folders, see if this still happens
        for individual_features in feature_set:
            # print(len(individual_features))
            assert len(individual_features) == 20
            try:
                holder[str(key) + "F1_mean"].append(float(individual_features[0]))
                holder[str(key) + "F1_median"].append(float(individual_features[1]))
                holder[str(key) + "F1_stdev"].append(float(individual_features[2]))
                holder[str(key) + "F1_first_quartile"].append(
                    float(individual_features[3])
                )
                holder[str(key) + "F1_third_quartile"].append(
                    float(individual_features[4])
                )

                holder[str(key) + "F2_mean"].append(float(individual_features[5]))
                holder[str(key) + "F2_median"].append(float(individual_features[6]))
                holder[str(key) + "F2_stdev"].append(float(individual_features[7]))
                holder[str(key) + "F2_first_quartile"].append(
                    float(individual_features[8])
                )
                holder[str(key) + "F2_third_quartile"].append(
                    float(individual_features[9])
                )

                holder[str(key) + "F3_mean"].append(float(individual_features[10]))
                holder[str(key) + "F3_median"].append(float(individual_features[11]))
                holder[str(key) + "F3_stdev"].append(float(individual_features[12]))
                holder[str(key) + "F3_first_quartile"].append(
                    float(individual_features[13])
                )
                holder[str(key) + "F3_third_quartile"].append(
                    float(individual_features[14])
                )

                holder[str(key) + "F4_mean"].append(float(individual_features[15]))
                holder[str(key) + "F4_median"].append(float(individual_features[16]))
                holder[str(key) + "F4_stdev"].append(float(individual_features[17]))
                holder[str(key) + "F4_first_quartile"].append(
                    float(individual_features[18])
                )
                holder[str(key) + "F4_third_quartile"].append(
                    float(individual_features[19])
                )
            except TypeError:
                print("Bad key: ", key)
    max_length = max(map(len, list(holder.values())))
    # print(max_length)
    for key, sub in holder.items():
        if max_length - len(sub) != 0:
            for i in range(max_length - len(sub)):
                sub.append(0)
    return pd.DataFrame.from_dict(holder)


# df = fake_profile_kit_into_df(fp_kit_feature_vector)
# print(df)
# print("Null Count:", df.isnull().sum().sum())


################################################################################################
# Feature Selection Utilities


def select_k_best_features(X, feature_count=50):
    y = pd.read_csv("Filtered_Demographics.csv")
    id_df = y[["ID"]]
    Y_vector = id_df

    X_matrix = X.to_numpy()
    Y_vector = Y_vector.to_numpy()
    print(Y_vector.ravel().shape)
    Y_vector = Y_vector.ravel()
    # Temporary Fix to address an off by one error when selecting k_best features
    if not X_matrix.shape[0] == Y_vector.shape[0]:
        Y_vector = np.delete(Y_vector, -1, 0)
    le = LabelEncoder()
    sb = SelectKBest(f_classif, k=feature_count)
    Y_vector = le.fit_transform(Y_vector)
    X_matrix_new = sb.fit_transform(X_matrix, Y_vector)
    print("SHAPE:", X_matrix_new.shape)
    cols = X.columns[sb.get_support(indices=True)].tolist()
    print(len(cols))
    data = X[X.columns.intersection(cols)]
    print("Top", data.shape)
    return data


def select_features(X, using_test_set=False):
    y = pd.read_csv("/content/drive/My Drive/Fake_Profile/Filtered_Demographics.csv")
    id_df = y[["ID"]]
    Y_vector = id_df
    features = X.columns
    final_features = []
    X_matrix = X.to_numpy()
    min_max_scaler = preprocessing.MinMaxScaler()
    X_matrix = min_max_scaler.fit_transform(X_matrix)
    # print(X_matrix)
    if using_test_set == False:
        Y_vector = Y_vector.to_numpy()
        if X_matrix.shape[0] < Y_vector.shape[0]:
            Y_vector = np.delete(Y_vector, [Y_vector.shape[0] - 1])
    else:
        Y_vector = np.tile(Y_vector.to_numpy(), 2)
    Y_vector = Y_vector.ravel()
    kept_features = {}
    mi_score = MIC(X_matrix, Y_vector, discrete_features=True)
    max_mi = max(mi_score)
    mi_score /= max_mi
    # Iterate through the mi list and keep all the features with more 0.7 or 0.8 and keep the position (roughly 40-50 features)
    for idx, score in enumerate(mi_score):
        if score >= 0.90:
            kept_features[idx] = score
    for idx, score in kept_features.items():
        # print("Feature:", features[idx], "Score:", score)
        final_features.append(features[idx])
    print(len(list(kept_features.keys())))
    data = X[X.columns.intersection(final_features)]
    return data


simplefilter(action="ignore", category=UserWarning)


class PlatformCombinations(Enum):
    FF = 0
    FI = 1
    FT = 2
    II = 3
    IT = 4
    TT = 5


def scale_together(df1, df2):
    # When normalizing we should consider the entire numpy array colum-wise
    # and compute the min-max as a whole of the column using the min-max
    # normalization algorithm but look at the distance measures paper to see what
    # other methods we can use

    first_column_names = list(df1.columns)
    second_column_names = list(df2.columns)
    if first_column_names.sort() == second_column_names.sort():
        # This is a special case: If we are dealing with 2 dataframes representing
        # identical datasets (such as when heatmaping the BBMAS dataset), we can
        # just scale one of the dataframes and just return it for split1, and
        # split2
        scaler = preprocessing.MinMaxScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(df1.values), columns=df1.columns, index=df1.index
        )
        split1 = scaled_data[scaled_data.columns.intersection(first_column_names)]
        split2 = scaled_data[scaled_data.columns.intersection(second_column_names)]

    else:
        # This is the general casee where we are dealing with 2 distinct sets of data
        # where column names do not repeat
        concatenated_df = pd.concat([df1, df2], axis=1)
        scaler = preprocessing.MinMaxScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(concatenated_df.values),
            columns=concatenated_df.columns,
            index=concatenated_df.index,
        )
        split1 = scaled_data[scaled_data.columns.intersection(first_column_names)]
        split2 = scaled_data[scaled_data.columns.intersection(second_column_names)]

    return (split1, split2)


def to_matrix(l, n):
    return [l[i : i + n] for i in range(0, len(l), n)]


def get_manhattan_distance(A, B):
    return sum(abs(val1 - val2) for val1, val2 in zip(A, B))


def euclidean_distance(a, b):
    return math.sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(a, b)))


def make_MIC_heatmap(X, top_feature_count=None):
    if top_feature_count is not None:
        data = select_k_best_features(X, top_feature_count)
    else:
        data = select_features(X)
    row_count = data.shape[0]
    print("Row Count is:", row_count)
    rows, cols = (top_feature_count, top_feature_count)
    arr = [[0] * cols] * rows
    distances = []
    for i in range(0, row_count):
        vec1 = data.iloc[[i]].to_numpy().tolist()[0]
        for j in range(0, row_count):
            vec2 = data.iloc[[j]].to_numpy().tolist()[0]
            distance = get_manhattan_distance(vec1, vec2)
            distances.append(distance)

    matrix = to_matrix(distances, row_count)
    print(matrix)
    max_dist = max(distances)
    distances = [x / max_dist for x in distances]
    ax = plt.axes()

    sns.heatmap(matrix, fmt=".1f", linewidth=0.5, vmin=0, vmax=max(distances), ax=ax)
    if top_feature_count is not None:
        ax.set_title(
            "Instagram Manhattan Distance Heatmap of Top "
            + str(top_feature_count)
            + " MIC Features"
        )
    else:
        ax.set_title("Instagram Manhattan Distance Heatmap of Top MIC Features")
    plt.show()


def plot_interclass_MIC_heatmap(X, X2, title, top_feature_count=None):
    if top_feature_count is not None:
        data = select_k_best_features(X, top_feature_count)
        data2 = select_k_best_features(X2, top_feature_count)
    else:
        data = select_features(X)
        data2 = select_features(X2)
    print("First shape:", data.shape)
    print("Second shape:", data2.shape)
    # print("Row Count is:", row_count)
    if top_feature_count is not None:
        print("Using top features")
        rows, cols = (top_feature_count, top_feature_count)
    else:
        rows, cols = (data.shape[0], data2.shape[0])
    arr = [[0] * cols] * rows
    distances = []

    scaled_data, scaled_data2 = scale_together(data, data2)
    if top_feature_count is not None:
        row_count = scaled_data.shape[1]
        other_row_count = scaled_data2.shape[1]
    else:
        row_count = scaled_data.shape[0]
        other_row_count = scaled_data2.shape[0]
    # For the identification pipeline we should consider using scan based
    # evaluation
    for i in range(0, row_count):
        vec1 = scaled_data.iloc[[i]].to_numpy().tolist()[0]
        # print(vec1)
        for j in range(0, other_row_count):
            vec2 = scaled_data2.iloc[[j]].to_numpy().tolist()[0]
            distance = get_manhattan_distance(vec1, vec2)
            distances.append(distance)
            # print(distance)
    # TODO: Try applying a min_max scaler to the matrix prior to graphing right here
    matrix = to_matrix(distances, row_count)
    print(matrix)
    max_dist = max(distances)
    scaled_dists = []
    for x in distances:
        try:
            scaled_dists.append(x / max_dist)
        except ZeroDivisionError:
            print("Encountered ZeroDivisionError")
            scaled_dists.append(0)
    # distances = [x / max_dist for x in distances]
    ax = plt.axes()

    sns.heatmap(matrix, fmt=".1f", linewidth=0.5, vmin=0, vmax=max(scaled_dists), ax=ax)
    if top_feature_count is not None:
        ax.set_title(
            "Instagram Manhattan Distance Heatmap of Top "
            + str(top_feature_count)
            + " MIC Features"
        )
    else:
        ax.set_title(title)
        plt.savefig(title + ".png")
    plt.show()


def make_heatmap(combination: PlatformCombinations, title, top_feature_count=None):
    # os.chdir("/content/drive/My Drive/Fake_Profile/")
    if combination == PlatformCombinations.FF:
        with open("facebook_result_df.pkl", "rb") as f:
            test_df = pickle.load(f)
        with open("facebook_result_df.pkl", "rb") as f:
            test_df2 = pickle.load(f)
    elif combination == PlatformCombinations.FI:
        with open("facebook_result_df.pkl", "rb") as f:
            test_df = pickle.load(f)
        with open("instagram_result_df.pkl", "rb") as f:
            test_df2 = pickle.load(f)
    elif combination == PlatformCombinations.FT:
        with open("facebook_result_df.pkl", "rb") as f:
            test_df = pickle.load(f)
        with open("twitter_result_df.pkl", "rb") as f:
            test_df2 = pickle.load(f)
    elif combination == PlatformCombinations.II:
        with open("instagram_result_df.pkl", "rb") as f:
            test_df = pickle.load(f)
        with open("instagram_result_df.pkl", "rb") as f:
            test_df2 = pickle.load(f)
    elif combination == PlatformCombinations.IT:
        with open("instagram_result_df.pkl", "rb") as f:
            test_df = pickle.load(f)
        with open("twitter_result_df.pkl", "rb") as f:
            test_df2 = pickle.load(f)
    elif combination == PlatformCombinations.TT:
        with open("twitter_result_df.pkl", "rb") as f:
            test_df = pickle.load(f)
        with open("twitter_result_df.pkl", "rb") as f:
            test_df2 = pickle.load(f)
    X = test_df
    X2 = test_df2
    if top_feature_count == None:
        plot_interclass_MIC_heatmap(X, X2, title)
    else:
        plot_interclass_MIC_heatmap(X, X2, title, top_feature_count=top_feature_count)

df = generate_kht_features_df()
print(df)
# if os.path.exists("facebook_result_df.pkl"):
#   with open("facebook_result_df.pkl", "rb") as f:
#     fp_result = pickle.load(f)
# else:
#   data = fake_profile_kht_feature_vector()
#   fp_kht_df = fake_profile_kht_into_df(data)
#   with open("fake_profile_kht_df.pkl", 'wb') as f:
#       pickle.dump(fp_kht_df, f)

# #   os.chdir('/content/drive/My Drive/Fake_Profile/')
#   # with open("kht.pkl", "rb") as f:
#   #   df1 = pickle.load(f)
#   fp_kht_df.reset_index(inplace=True, drop=True)
#   # fp_kht_df = fp_kht_df.drop(columns=fp_kht_df.columns[fp_kht_df.eq(0).mean()>0.5])
#   data2 = fake_profile_kit_feature_vector()
# #   os.chdir('/content/drive/My Drive/Fake_Profile/')
#   with open("fake_profile_pickled_kit_data.pkl", 'wb') as f:
#       pickle.dump(data2, f)
#   # with open("fake_profile_pickled_kit_data.pkl", "rb") as f:
#   #     data2 = pickle.load(f)

#   fp_kit_df = fake_profile_kit_into_df(data2)
#   fp_kit_df.reset_index(inplace=True, drop=True)
#   # fp_kit_df = fp_kit_df.drop(columns=fp_kit_df.columns[fp_kit_df.eq(0).mean()>0.5])
#   fp_word_level_df = generate_word_level_features_df(False)
#   with open("fake_profile_word_level_df.pkl", 'wb') as f:
#       pickle.dump(fp_word_level_df, f)

#   # fp_word_level_df = fp_word_level_df.drop(columns=fp_word_level_df.columns[fp_word_level_df.eq(0).mean()>0.5])
#   fp_result = pd.concat([fp_kht_df,fp_kit_df,fp_word_level_df], axis=1)
#   # fp_result = fp_result.drop(columns=fp_result.columns[fp_result.eq(0).mean()>0.5])
# #   fp_result.dropna(inplace=True)
# #   os.chdir('/content/drive/My Drive/Fake_Profile/')
#   with open("facebook_result_df.pkl", "wb") as f:
#         pickle.dump(fp_result, f)

# make_heatmap(PlatformCombinations.FF, "FF Manhattan All MIC Features", 10)
# make_heatmap(PlatformCombinations.FI, "FI Manhattan All MIC Features")
# make_heatmap(PlatformCombinations.FT, "FT Manhattan All MIC Features")
# make_heatmap(PlatformCombinations.II, "II Manhattan All MIC Features")
# make_heatmap(PlatformCombinations.IT, "IT Manhattan All MIC Features")
# make_heatmap(PlatformCombinations.TT, "TT Manhattan All MIC Features", 10)

