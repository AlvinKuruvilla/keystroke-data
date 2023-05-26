def max_unigraph_disorder(n):
    return (n*n)-1
class RelativeVerifier:
    def __init__(self, raw_template, raw_verification):
        self.template = raw_template
        self.verification = raw_verification
    def get_all_matching_keys(self):
        return list(set(self.template.keys()).intersection(set (self.verification.keys())))
    def keys_for_template_and_verification(self):
        return (list(self.template.keys()), list(self.verification.keys()))
    def calculate_disorder(self):
        matching_keys = self.get_all_matching_keys()
        disorder = 0
        template_keys, verification_keys = self.keys_for_template_and_verification()
        for key in matching_keys:
            disorder = disorder + abs(template_keys.index(key) - verification_keys.index(key))
        return disorder/max_unigraph_disorder(len(matching_keys))
