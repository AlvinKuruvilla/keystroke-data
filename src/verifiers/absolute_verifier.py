import statistics
from verifier import Verifier

class AbsoluteVerifier(Verifier):
    def __init__(self, raw_enroll, raw_verification):
        super().__init__(raw_enroll, raw_verification)

    def get_match_score(self):
        common_keys = self.get_common_keys()
        if len(common_keys) == 0:
            return 0
        matches = 0
        for key in common_keys:
            print(self.verification[key])
            template_mean = statistics.mean(self.enroll[key])
            verification_mean = statistics.mean(self.verification[key])
            ratio = max(template_mean, verification_mean) / min(template_mean, verification_mean)

            threshold = max(self.enroll[key]) / min(self.enroll[key])

            if ratio <= threshold:
                matches += 1

        return  (matches / len(common_keys))


# # local testing
# enroll = {"W":[210, 220, 200, 230], "E":[110, 115, 107], "L":[150, 130, 190, 120], "C":[25, 30, 35, 70], "O":[90, 40, 49]}
# verification = {"W":[200, 205, 203, 225, 245, 190], "E":[25, 30, 35, 70], "L":[150, 130, 190, 120], "N":[25, 30, 35, 70], "S":[90, 40, 49]}
#
# AbsVer = AbsoluteVerifier(enroll,verification)
# print(AbsVer.get_common_keys())
# print(AbsVer.get_match_score())
