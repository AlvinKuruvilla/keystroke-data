from verifiers.similarity_verifier import SimilarityVerifier
from verifiers.relative_verifier import RelativeVerifier
from verifiers.template_generator import create_kht_template_for_user_and_platform_id, create_kht_verification_attempt_for_user_and_platform_id, create_kit_flight_template_for_user_and_player, create_kit_flight_verification_attempt_for_user_and_player
for i in range(1,27):
    template = create_kit_flight_template_for_user_and_player(i,4)
    verification = create_kit_flight_verification_attempt_for_user_and_player(i, 4)

    s_verifier = SimilarityVerifier(template, verification)
    # print(s_verifier.find_match_percent())
    r_verifier = RelativeVerifier(template, verification)
    print(r_verifier.calculate_disorder(True))
