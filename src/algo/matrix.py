import json
import statistics
import string


def get_json_values_for_key(path: str, key: str):
    with open(path) as json_file:
        data = json.load(json_file)
    return data.get("'" + key + "'")

def get_kit_keys(path):
    with open(path) as json_file:
        res =json.load(json_file)
        k = list(res.keys())
        store = set()
        for s in k:
            try:
                store.add(eval(s))
            except SyntaxError:
                store.add(find_enclosed_string(s))
            except NameError:
                store.add(handle_concatenated_special_keys(s))
    return list(store)
def tuple_to_string(t):
    return ''.join(t)

def get_clean_kit_keys(path):
    keys = get_kit_keys(path)
    clean = set()
    for key in keys:
        if type(key) == str:
            clean.add(key)
        elif type(key) == tuple:
            clean.add(tuple_to_string(key))
    return clean
    

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
def find_enclosed_string(text):
    # Find the index of the first single quote
    start_index = text.find("'")

    # Find the index of the second single quote
    end_index = text.find("'", start_index + 1)

    # Extract the substring between the two single quotes
    enclosed_string = text[start_index + 1:end_index]
    
    # Remove the enclosed string from the original string
    new_string = text[:start_index] + text[end_index + 1:]

    return (new_string,enclosed_string)

def handle_concatenated_special_keys(text):
    substring = "Key"
    substring_count = text.count(substring)
    # Key.xKey.y
    if substring_count == 2:
        first_index = text.find(substring)
        second_index = text.find(substring, first_index + 1)
        return (text[:first_index], text[second_index:])

res = get_clean_kit_keys("/Users/alvinkuruvilla/Dev/keystroke-data/features/kit/KIT1_for_1.json")
# for key in res:
#     print(get_kit_json_values_for_key("/Users/alvinkuruvilla/Dev/keystroke-data/features/kit/KIT1_for_1.json", key))
for key in res:
    print(get_kit_json_values_for_key("/Users/alvinkuruvilla/Dev/keystroke-data/features/kit/KIT1_for_1.json", key))

