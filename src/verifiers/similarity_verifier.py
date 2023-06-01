import statistics

from verifiers.verifier import Verifier


class SimilarityVerifier(Verifier):
    def __init__(self, raw_template, raw_verification):
        super().__init__(raw_template, raw_verification)

    # Compute the match score for all the matching keys by seeing if the verification mean
    # falls in the range of the template mean + or - the standard deviation.
    # I wasn't sure how to handle the case where there was not enough points to compute the standard deviation
    def find_match_percent(self):
        matching_keys = self.get_all_matching_keys()
        matches = 0
        for key in matching_keys:
            template_mean = statistics.mean(self.template[key])
            try:
                template_stdev = statistics.stdev(self.template[key])
            except statistics.StatisticsError:
                template_stdev = 10
            for times in self.verification[key]:
                if (
                    template_mean - template_stdev
                    < times
                    < template_mean + template_stdev
                ):
                    matches += 1
        return matches / len(matching_keys)
