import statistics


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

    def get_abs_match_score(self):  # A verifier
        if len(self.common_features) == 0:  # if there exist no common features,
            return 0
            # raise ValueError("No common features to compare!")
        matches = 0
        for (
            feature
        ) in self.common_features:  # checking for every common feature for match
            pattern1_mean = statistics.mean(self.pattern1[feature])
            pattern2_mean = statistics.mean(self.pattern2[feature])
            if min(pattern1_mean, pattern2_mean) == 0:
                ratio = 1
            else:
                ratio = max(pattern1_mean, pattern2_mean) / min(
                    pattern1_mean, pattern2_mean
                )
            # the following threshold is what we thought would be good
            # we have not analyzed it yet!
            try:
                threshold = max(self.pattern1[feature]) / min(self.pattern1[feature])
            except ZeroDivisionError:
                threshold = 0
            if ratio <= threshold:
                matches += 1
        return matches / len(self.common_features)

    def get_similarity_score(self):  # S verifier, each key same weight
        if len(self.common_features) == 0:  # if there exist no common features,
            raise ValueError("No common features to compare!")
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
                if (
                    (pattern1_mean - pattern1_stdev)
                    < time
                    < (pattern1_mean + pattern1_stdev)
                ):
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
            raise ValueError("No common features to compare!")
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
                if (
                    (enroll_mean - template_stdev)
                    < time
                    < (enroll_mean + template_stdev)
                ):
                    matches += 1
                total += 1

        return matches / total


# local testing
pattern1 = {
    "W": [210, 220, 200, 230],
    "E": [110, 115, 107],
    "L": [150, 130, 190, 120],
    "C": [25, 30, 35, 70],
    "O": [90, 40, 49],
}
pattern2 = {
    "W": [245, 190],
    "E": [25, 30, 35, 70],
    "L": [150, 130, 190, 120],
    "N": [25, 30, 35, 70],
    "S": [90, 40, 49],
}

ExampleVerifier = Verifiers(pattern1, pattern2)
print(ExampleVerifier.get_abs_match_score())
print(ExampleVerifier.get_similarity_score())
print(ExampleVerifier.get_weighted_similarity_score())
