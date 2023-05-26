from verifiers.similarity_verifier import SimilarityVerifier
from verifiers.template_generator import create_kht_template_for_user_and_platform_id, create_kht_verification_attempt_for_user_and_platform_id
for i in range(1,27):
    template = create_kht_template_for_user_and_platform_id(i,1)
    verification = create_kht_verification_attempt_for_user_and_platform_id(i)

    s_verifier = SimilarityVerifier(template, verification)
    print(s_verifier.find_match_percent())
