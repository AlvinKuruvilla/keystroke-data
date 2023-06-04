class Verifier:
    def __init__(self, raw_template, raw_verification):
        self.template = raw_template
        self.verification = raw_verification

    def remove_zero_time_matches(self):
        matching_keys = list(
            set(self.template.keys()).intersection(set(self.verification.keys()))
        )
        final_keys = []
        for key in matching_keys:
            if all(v == 0 for v in self.template[key]) or all(
                v == 0 for v in self.verification[key]
            ):
                continue
            final_keys.append(key)
        return final_keys

    def get_all_matching_keys(self):
        return self.remove_zero_time_matches()
