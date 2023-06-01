class Verifier:
    def __init__(self, raw_template, raw_verification):
        self.template = raw_template
        self.verification = raw_verification

    def get_all_matching_keys(self):
        return list(
            set(self.template.keys()).intersection(set(self.verification.keys()))
        )
