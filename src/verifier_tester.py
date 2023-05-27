from verifiers.similarity_verifier import SimilarityVerifier
from verifiers.relative_verifier import RelativeVerifier
from verifiers.template_generator import create_kht_template_for_user_and_platform_id, create_kht_verification_attempt_for_user_and_platform_id, create_kit_flight_template_for_user_and_platform, create_kit_flight_verification_attempt_for_user_and_platform
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
