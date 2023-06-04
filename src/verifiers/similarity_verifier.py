import statistics
from verifier import Verifier

class SimilarityVerifier(Verifier):
    def __init__(self, raw_enroll, raw_verification):
        super().__init__(raw_enroll, raw_verification)
    # Compute the match score for all the matching keys by seeing if the verification mean
    # falls in the range of the template mean + or - the standard deviation.
    # I wasn't sure how to handle the case where there was not enough points to compute the standard deviation
    def get_match_score(self):
        common_keys = self.get_common_keys()
        matches, total = 0, 0
        if len(common_keys) == 0:
            return 0
        for key in common_keys:
            enroll_mean = statistics.mean(list(self.enroll[key]))
            try:
                template_stdev = statistics.stdev(self.enroll[key])
            except statistics.StatisticsError:
                template_stdev = self.enroll[key]/6  # this will always be one value that is when exception would occur

            for time in self.verification[key]:
                if (enroll_mean - template_stdev) < time < (enroll_mean + template_stdev):
                    matches += 1
                total+=1

        return matches / total


    def get_match_score_unweighted(self):
        common_keys = self.get_common_keys()
        if len(common_keys) == 0:
            return 0
        key_matches, total_keys = 0, 0
        for key in common_keys:
            enroll_mean = statistics.mean(list(self.enroll[key]))
            try:
                template_stdev = statistics.stdev(self.enroll[key])
            except statistics.StatisticsError:
                template_stdev = self.enroll[key]/6  # this will always be one value that is when exception would occur

            value_matches, total_values = 0, 0
            for time in self.verification[key]:
                if (enroll_mean - template_stdev) < time < (enroll_mean + template_stdev):
                    value_matches += 1
                total_values+=1
            if value_matches/total_values <= 0.5:
                key_matches+=1

            total_keys+=1

        return key_matches / total_keys

# local testing
enroll = {"W":[210, 220, 200, 230], "E":[110, 115, 107], "L":[150, 130, 190, 120], "C":[25, 30, 35, 70], "O":[90, 40, 49]}
verification = {"W":[200, 205, 203, 225, 245, 190], "E":[25, 30, 35, 70], "L":[150, 130, 190, 120], "N":[25, 30, 35, 70], "S":[90, 40, 49]}

SimVer = SimilarityVerifier(enroll,verification)
print(SimVer.get_common_keys())
print(SimVer.get_match_score_unweighted())
