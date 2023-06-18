import statistics
from src.verifiers.ecdf import ECDF
import numpy as np

class Verifiers:
    def __init__(self, p1, p2):
        # p1 and p2 are dictionaries of features
        # keys in the dictionaries would be the feature names
        # feature names mean individual letters for KHT
        # feature names could also mean pair of letters for KIT or diagraphs
        # feature names could also mean pair of sequence of three letters for trigraphs
        # feature names can be extedned to any features that we can extract from keystrokes

        self.pattern1 = p1
        self.pattern2 = p2
        self.common_features = set(self.pattern1.keys()).intersection(
            set(self.pattern2.keys())
        )
        # print('self.common_features:', self.common_features)

    def get_abs_match_score(self):  # A verifier
        if len(self.common_features) == 0:  # if there exist no common features,
            raise ValueError("Error: no common features to compare!")
        matches = 0
        for feature in self.common_features:  # checking for every common feature for match
            print(f'feature:{feature}')
            print(f'self.pattern1[feature]:{self.pattern1[feature]}')
            print(f'self.pattern2[feature]:{self.pattern2[feature]}')

            pattern1_mean = statistics.mean(self.pattern1[feature])
            pattern2_mean = statistics.mean(self.pattern2[feature])
            if min(pattern1_mean, pattern2_mean) == 0:
                raise ValueError('min of means is zero, should not happen!')
            else:
                ratio = max(pattern1_mean, pattern2_mean) / min(
                    pattern1_mean, pattern2_mean
                )
            # the following threshold is what we thought would be good
            # we have not analyzed it yet!
            # try:
            #     threshold = max(self.pattern1[feature]) / min(self.pattern1[feature])
            # except ZeroDivisionError:
            #     threshold = 0
            threshold = 1.5 # hardcoding the threshold
            if ratio <= threshold: # basically the current feature matches
                matches += 1
        return matches / len(self.common_features)

    def get_similarity_score(self):  # S verifier, each key same weight
        if len(self.common_features) == 0:  # if there exist no common features,
            return 0
            # raise ValueError("No common features to compare!")
        key_matches, total_features = 0, 0
        for feature in self.common_features:
            pattern1_mean = statistics.mean(list(self.pattern1[feature]))
            try:
                pattern1_stdev = statistics.stdev(self.pattern1[feature])
            except statistics.StatisticsError:
                print("In error: ", self.pattern1[feature])
                if len(self.pattern1[feature]) == 1:
                    pattern1_stdev = self.pattern1[feature][0] / 4
                else:
                    pattern1_stdev = (
                        self.pattern1[feature] / 4
                    )  # this will always be one value that is when exception would occur

            value_matches, total_values = 0, 0
            for time in self.pattern2[feature]:
                if (pattern1_mean - pattern1_stdev) < time and time < (pattern1_mean + pattern1_stdev):
                    value_matches += 1
                total_values += 1
            if value_matches / total_values <= 0.5:
                key_matches += 1
            total_features += 1

        return key_matches / total_features

    def get_weighted_similarity_score(
        self,
    ):  # S verifier, each feature different weights
        if len(self.common_features) == 0:  # if there exist no common features,
            return 0
            # raise ValueError("No common features to compare!")
        matches, total = 0, 0
        for feature in self.common_features:
            enroll_mean = statistics.mean(list(self.pattern1[feature]))
            try:
                template_stdev = statistics.stdev(self.pattern1[feature])
            except statistics.StatisticsError:
                print("In error: ", self.pattern1[feature])
                if len(self.pattern1[feature]) == 1:
                    template_stdev = self.pattern1[feature][0] / 4
                else:
                    template_stdev = self.pattern1[feature] / 4

            for time in self.pattern2[feature]:
                if (enroll_mean - template_stdev) < time and time < (enroll_mean + template_stdev):
                    matches += 1
                total += 1
        return matches / total

    def get_median_of_common_key(self, key):
        if not key in self.common_features:
            raise ValueError(str(key) + " is not in common keys")
        return statistics.median(self.pattern1[key])

    def x_i(self, key):
        return statistics.mean(self.pattern2[key])

    def ecdf_of_x(self, key, x_i):
        # The ITAD algorithm requires that we get CDF(x_i)
        # To do that we first get all the timing values of the key
        # We then append the (at the point of calling this function) calculated x_i value

        # After getting the y values for the data by calling the ecdf function, we get the specific y_value for x_i by assuming
        # that each y_value in the list is a 1-1 match to sorted x_values. So by finding the index of x_i in the sorted x_values
        # we can find the corresponding y value by y_vals[x_i_index]
        data = list(self.pattern2[key])
        data.append(x_i)
        x_vals, y_vals = self.compute_ecdf(data)
        data = list(np.sort(data))
        x_i_index = data.index(x_i)
        return y_vals[x_i_index]

    def itad_metric(self, key, sample_duration):
        # In the context of this function, the sample_duration is x_i in the algorithm (in our case, it will be the probe mean)
        m_x = self.get_median_of_common_key(key)
        ecdf = ECDF(self.pattern2[key])
        if sample_duration <= m_x:
            # return self.ecdf_of_x(key, sample_duration)
            return ecdf(sample_duration)
        # return 1 - self.ecdf_of_x(key, sample_duration)
        return 1 - ecdf(sample_duration)

    def itad_similarity(self, p=0.5):
        total = 0
        matching_keys = self.common_features
        for key in matching_keys:
            x = self.x_i(key)
            itad_value = self.itad_metric(key, x) * p
            total += itad_value
        try:
            return (1 / len(self.common_features)) * total
        except ZeroDivisionError:
            raise ValueError('Zero division occured: no common key found!')

    def compute_ecdf(data):
        """Compute ECDF"""
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n + 1) / n
        return (x, y)

# local testing

# Same pattern | complete overlap
# completely different patterns
# moderate overlapping patterns
#
pattern1 = {
    "W": [210, 220, 200, 230, 210, 220, 200, 230, 210, 220, 200, 230],
    "E": [110, 115, 107, 110, 115, 107, 110, 115, 107],
    "L": [150, 130, 190, 120, 150, 130, 190, 120],
    "C": [25, 30, 35, 70, 25, 30, 35, 70, 25, 30, 35, 70, 25, 30, 35, 70],
    "O": [90, 40, 49, 90, 40, 49, 90, 40, 49, 90, 40, 49],
}


pattern2 = {
    "W": [11, 12, 13, 14, 15, 16, 11, 12, 13, 14, 15, 16, 11, 12, 13, 14, 15, 16],
    "E": [25, 30, 35, 70, 25, 30, 35, 70, 25, 30, 35, 70],
    "L": [1, 23, 21, 23, 43, 45, 64, 23, 43],
    "N": [9 , 4, 12, 23, 21, 11, 9, 9 , 4, 12, 23, 21, 11, 9],
    "S": [512, 621, 234, 257, 289, 512, 621, 234, 257, 289],
}



ExampleVerifier = Verifiers(pattern1, pattern2)
print('get_abs_match_score():', ExampleVerifier.get_abs_match_score())
print('get_similarity_score():', ExampleVerifier.get_similarity_score())
print('get_weighted_similarity_score():', ExampleVerifier.get_weighted_similarity_score())
print('itad_similarity():', ExampleVerifier.itad_similarity())
