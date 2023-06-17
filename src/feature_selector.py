import statistics


def quickly_typed_selector(keystroke_data_dict):
    features = []
    for key in list(keystroke_data_dict.keys()):
        value = keystroke_data_dict[key]
        if isinstance(value, int):
            features.append(value)
        elif isinstance(value, list) and len(value) == 1:
            features.append(value[0])
        elif isinstance(value, list) and len(value) > 1:
            features.append(statistics.mean(value))
    return features.sort()


def most_frequent_features(keystroke_data_dict, cutoff_length: int):
    features = {}
    for key in list(keystroke_data_dict.keys()):
        value = keystroke_data_dict[key]
        if isinstance(value, int):
            features[key] = 1
        elif isinstance(value, list) and len(value) == 1:
            features[key] = 1
        elif isinstance(value, list) and len(value) > 1:
            features[key] = len(value)
    # This should sort the keys by frequency of occurrence from greatest to smallest
    sorted_data = dict(sorted(features.items(), key=lambda x: x[1], reverse=True))
    # Filter out all the keys where the value is less than the cutoff frequency
    filtered_dict = {k: v for (k, v) in sorted_data.items() if v > cutoff_length}
    final = {}
    for key in filtered_dict.keys():
        final[key] = keystroke_data_dict[key]
    return final
