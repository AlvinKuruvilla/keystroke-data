class Verifier:
    def __init__(self, raw_enroll, raw_verification):
        self.enroll = raw_enroll   # dictionary of the enrollment features
        self.verification = raw_verification # dictionary of the verification features

    '''
    enroll = {"W":[210, 220, 200, 230], "E":[110, 115, 107], "L":[150, 130, 190, 120], "C":[25, 30, 35, 70], "O":[90, 40, 49]}
    verification = {"W":[200, 205, 203, 225, 245, 190], "E":[25, 30, 35, 70], "L":[150, 130, 190, 120], "N":[25, 30, 35, 70], "S":[90, 40, 49]}    
   
    Output: 
    
    formula for a verifier | 
                Step1: find and compare common_keys
                Step2: total_matches, valid_matches = 0, 0
                Step2: for en_key, ver_key common_keys in zip(enroll, verification):
                            mean_en = 
                            std_en = 
                            mean_ver = 
                            std_ver = 
                            th = max(en_values)/min(en_values)
                            if max(mean_en, mean_ver)/min(mean_en, mean_ver) < th : 
                                valid_matches+=1
                            total_matches +=1
                Step 3: 
                        return valid_matches/total_matches
    
    # the output should be 2/3
    
    # formula for s verifier | equal weightage for each feature
    
         Step1: find and compare common_keys
                Step2: valid, total = 0, 0 
                Step3: for en_key, ver_key common_keys in zip(enroll, verification):
                            mean_en = 
                            std_en = 
                            valid_val_matches, total_val_matches = 0, 0 
                            for value in ver_keys:
                                if value < mean_en+std_en or value > mean_en-std_en:
                                    valid_val_matches+=1
                                    
                            if valid_val_matches/total_val_matches < 0.5: # if 50% of the values match for current key then we say the key matches
                                valid+=1
                            total+=1
                            
                Step4:  return  valid/total
        
         # the output should be 2/3  
          
    # formula for s verifier | UNEqual weightage for each feature
         Step1: find and compare common_keys
                Step2: valid, total = 0, 0 
                Step3: for en_key, ver_key common_keys in zip(enroll, verification):
                            mean_en = 
                            std_en = 
                            for value in ver_keys:
                                if value < mean_en+std_en or value > mean_en-std_en:
                                    valid+=1
                            total+=1
                            
                Step4:  return  valid/total
    #  output: we dont know 
    
    # formula for R verifier |         
    a_match_score = 
    s_match_score = 
    r_match_score = 
    '''

    def get_common_keys(self):
        common_keys = set(self.enroll.keys()).intersection(set(self.verification.keys()))
        return common_keys

    def remove_zero_time_matches(self):
        matching_keys = list(
            set(self.enroll.keys()).intersection(set(self.verification.keys()))
        )
        final_keys = []
        for key in matching_keys:
            if all(v == 0 for v in self.enroll[key]) or all(
                v == 0 for v in self.verification[key]
            ):
                continue
            final_keys.append(key)
        return final_keys

    def get_all_matching_keys(self):
        return self.remove_zero_time_matches()


# local testing
# enroll = {"W":[210, 220, 200, 230], "E":[110, 115, 107], "L":[150, 130, 190, 120], "C":[25, 30, 35, 70], "O":[90, 40, 49]}
# verification = {"W":[200, 205, 203, 225, 245, 190], "E":[25, 30, 35, 70], "L":[150, 130, 190, 120], "N":[25, 30, 35, 70], "S":[90, 40, 49]}
#
# Ver = Verifier(enroll,verification)
# print(Ver.get_common_keys())
