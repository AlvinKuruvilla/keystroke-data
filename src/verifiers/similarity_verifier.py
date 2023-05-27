import statistics

class SimilarityVerifier:
    def __init__(self, raw_template, raw_verification):
        self.template = raw_template
        self.verification = raw_verification
    def get_all_matching_keys(self):
        return list(set(self.template.keys()).intersection(set (self.verification.keys())))
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
            # TODO: Take each timing of the verification attempt for a key as a seperate match
            verification_mean = statistics.mean(self.verification[key])
            if template_mean - template_stdev <verification_mean <template_mean+ template_stdev:
                matches +=1
        return matches/len(matching_keys)