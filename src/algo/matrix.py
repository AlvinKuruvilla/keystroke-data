import json
import statistics
import string


def get_json_values_for_key(path: str, key: str):
    with open(path) as json_file:
        data = json.load(json_file)
    return data.get("'" + key + "'")


def get_kit_json_values_for_key(path: str, key: str):
    with open(path) as json_file:
        return json.load(json_file)


# The JSON stores all the KHT values for a particular key, but since this is going to be
# used with a 1d diagonal matrix we can just take the average of the values to get a
# single value
def parse_kht_average_dict_from_values(path: str):
    data = {}
    lower_alpha = list(string.ascii_lowercase)
    upper_alpha = list(string.ascii_uppercase)
    nums = list(range(10))
    for letter in lower_alpha:
        if get_json_values_for_key(path, letter) is None:
            data[letter] = 0
            continue
        data[letter] = statistics.mean(get_json_values_for_key(path, letter))
    for letter in upper_alpha:
        if get_json_values_for_key(path, letter) is None:
            data[letter] = 0
            continue
        data[letter] = statistics.mean(get_json_values_for_key(path, letter))
    for num in nums:
        if get_json_values_for_key(path, str(num)) is None:
            data[str(num)] = 0
            continue
        data[str(num)] = statistics.mean(get_json_values_for_key(path, str(num)))
    return data


def parse_kit_average_dict_from_values(path: str):
    with open(path) as json_file:
        return json.load(json_file)


def kht_matrix(dict):
    vals = list(dict.values())
    row_size = len(list(dict.keys()))
    matrix = []

    for index, value in enumerate(vals):
        row = [0] * row_size
        row[index] = value
        matrix.append(row)
    return matrix


res = parse_kit_average_dict_from_values(
    "/Users/alvinkuruvilla/Dev/keystroke-research/keystroke-data/features/kit/KIT1_for_1.json"
)
k = list(res.keys())
for s in k:
    print((s))
