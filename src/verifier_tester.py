import random
from verifiers.similarity_verifier import SimilarityVerifier
from verifiers.relative_verifier import RelativeVerifier
from verifiers.template_generator import create_kht_template_for_user_and_platform_id, create_kht_verification_attempt_for_user_and_platform_id, create_kit_flight_template_for_user_and_platform, create_kit_flight_verification_attempt_for_user_and_platform
def pick_random_user():
    return random.randint(1,27)
def cross_user_s_kht_test():
    chosen_user = pick_random_user()
    ids= list(range(1,28))
    ids.remove(chosen_user)
    same_user_kht_template = create_kht_template_for_user_and_platform_id(chosen_user, 1)
    same_user_kht_verification = create_kht_verification_attempt_for_user_and_platform_id(chosen_user)
    s_verifier = SimilarityVerifier(same_user_kht_template, same_user_kht_verification)
    print(f"Same user KHT match score for id {str(chosen_user)} is {str(s_verifier.find_match_percent())}")
    for remaining_id in ids:
        other_user_verification = create_kht_verification_attempt_for_user_and_platform_id(remaining_id)
        s_verifier = SimilarityVerifier(same_user_kht_template, other_user_verification)
        print(f"Cross user KHT match score for id {str(chosen_user)} and {str(remaining_id)} is {str(s_verifier.find_match_percent())}")
def cross_user_r_kht_test():
    chosen_user = pick_random_user()
    ids= list(range(1,28))
    ids.remove(chosen_user)
    same_user_kht_template = create_kht_template_for_user_and_platform_id(chosen_user, 1)
    same_user_kht_verification = create_kht_verification_attempt_for_user_and_platform_id(chosen_user)
    r_verifier = RelativeVerifier(same_user_kht_template, same_user_kht_verification)
    print(f"Same user KHT disorder score for id {str(chosen_user)} is {str(r_verifier.calculate_disorder(False))}")
    for remaining_id in ids:
        other_user_verification = create_kht_verification_attempt_for_user_and_platform_id(remaining_id)
        s_verifier = SimilarityVerifier(same_user_kht_template, other_user_verification)
        print(f"Cross user KHT fidorder score for id {str(chosen_user)} and {str(remaining_id)} is {str(s_verifier.find_match_percent())}")



print("================================================================ R Verifier (KIT F4 and KHHT) =================================================")
for i in range(1,28):
    kht_template = create_kht_template_for_user_and_platform_id(i, 1)
    kht_verification = create_kht_verification_attempt_for_user_and_platform_id(i)
    kit_template = create_kit_flight_template_for_user_and_platform(i,4)
    kit_verification = create_kit_flight_verification_attempt_for_user_and_platform(i, 4)

    s_verifier = SimilarityVerifier(kit_template, kit_verification)
    # print(s_verifier.find_match_percent())
    kht_r_verifier = RelativeVerifier(kht_template, kht_verification)
    kit_r_verifier = RelativeVerifier(kit_template, kit_verification)
    print("KIT R Verifier: "+ str(kit_r_verifier.calculate_disorder(True)))
    print("KHT Verifier:" + str(kht_r_verifier.calculate_disorder(False)))
print("================================================================ S Verifier KHT =================================================")
for i in range(1,28):
    kht_template = create_kht_template_for_user_and_platform_id(i, 1)
    kht_verification = create_kht_verification_attempt_for_user_and_platform_id(i)
    s_verifier = SimilarityVerifier(kht_template, kht_verification)
    print(s_verifier.find_match_percent())
print("================================================================ S Verifier All KIT =================================================")
for i in range(1,28):
    for j in range(1,5):
        kht_template = create_kit_flight_template_for_user_and_platform(i, j)
        kht_verification = create_kit_flight_verification_attempt_for_user_and_platform(i, j)
        s_verifier = SimilarityVerifier(kht_template, kht_verification)
        print(s_verifier.find_match_percent())
    print()
print("================================================================ R Verifier All KIT =================================================")
for i in range(1,28):
    for j in range(1,5):
        kht_template = create_kit_flight_template_for_user_and_platform(i, j)
        kht_verification = create_kit_flight_verification_attempt_for_user_and_platform(i, j)
        r_verifier = RelativeVerifier(kht_template, kht_verification)
        print(r_verifier.calculate_disorder(True))
    print()
cross_user_r_kht_test()
cross_user_s_kht_test()