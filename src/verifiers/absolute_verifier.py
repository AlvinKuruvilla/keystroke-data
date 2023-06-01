import statistics

from verifiers.verifier import Verifier


class AbsoluteVerifier(Verifier):
    def __init__(self, raw_template, raw_verification):
        super().__init__(raw_template, raw_verification)

    def find_match_percent(self):
        matching_keys = self.get_all_matching_keys()
        matches = 0
        for key in matching_keys:
            template_mean = statistics.mean(self.template[key])
            verification_mean = statistics.mean(self.verification[key])
            ratio = max(template_mean, verification_mean) / min(
                template_mean, verification_mean
            )
            threshold = max(self.template[key]) / min(self.verification[key])
            if ratio < threshold:
                matches += 1
        return 1 - (matches / len(matching_keys))
