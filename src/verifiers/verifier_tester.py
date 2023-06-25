import random
from verifiers.absolute_verifier import AbsoluteVerifier
from verifiers.similarity_verifier import SimilarityVerifier
from verifiers.relative_verifier import RelativeVerifier
from verifiers.template_generator import (
    create_kht_template_for_user_and_platform_id,
    create_kht_verification_attempt_for_user_and_platform_id,
    create_kit_flight_template_for_user_and_platform,
    create_kit_flight_verification_attempt_for_user_and_platform,
)


def pick_random_user():
    return random.randint(1, 27)


def cross_user_s_kht_test():
    chosen_user = pick_random_user()
    ids = list(range(1, 28))
    ids.remove(chosen_user)
    same_user_kht_template = create_kht_template_for_user_and_platform_id(
        chosen_user, 1
    )
    same_user_kht_verification = (
        create_kht_verification_attempt_for_user_and_platform_id(chosen_user, 1)
    )
    s_verifier = SimilarityVerifier(same_user_kht_template, same_user_kht_verification)
    same_user_match_score = s_verifier.find_match_percent()
    # print(
    #     f"Same user KHT match score for id {str(chosen_user)} is {str(s_verifier.find_match_percent())}"
    # )
    flag = 0
    for remaining_id in ids:
        other_user_verification = (
            create_kht_verification_attempt_for_user_and_platform_id(remaining_id, 1)
        )
        s_verifier = SimilarityVerifier(same_user_kht_template, other_user_verification)
        # print(
        #     f"Cross user KHT match score for id {str(chosen_user)} and {str(remaining_id)} is {str(s_verifier.find_match_percent())}"
        # )
        if s_verifier.find_match_percent() > same_user_match_score:
            flag += 1
    print(f"{flag}/ {len(ids)} found")


def cross_user_r_kht_test():
    chosen_user = pick_random_user()
    ids = list(range(1, 28))
    ids.remove(chosen_user)
    same_user_kht_template = create_kht_template_for_user_and_platform_id(
        chosen_user, 1
    )
    same_user_kht_verification = (
        create_kht_verification_attempt_for_user_and_platform_id(chosen_user, 1)
    )
    r_verifier = RelativeVerifier(same_user_kht_template, same_user_kht_verification)
    # print(
    #     f"Same user KHT disorder for id {str(chosen_user)} is {str(r_verifier.calculate_disorder())}"
    # )
    same_user_match_score = r_verifier.calculate_disorder()
    flag = 0
    for remaining_id in ids:
        other_user_verification = (
            create_kht_verification_attempt_for_user_and_platform_id(remaining_id, 1)
        )
        r_verifier = RelativeVerifier(same_user_kht_template, other_user_verification)
        # print(
        #     f"Cross user KHT disorder for id {str(chosen_user)} and {str(remaining_id)} is {str(s_verifier.find_match_percent())}"
        # )
        if r_verifier.calculate_disorder() > same_user_match_score:
            flag += 1

    print(f"{flag}/ {len(ids)} found")


def math_checks():
    template = {"tic": [370], "ica": [430], "the": [450]}
    verification = {"the": [340], "ica": [420], "tic": [540]}
    rv = RelativeVerifier(template, verification)
    print(rv.calculate_disorder())
    assert rv.calculate_disorder() == 1

    template = {"ic": [150], "he": [220], "th": [230], "ti": [265], "ca": [280]}
    verification = {"th": [150], "he": [190], "ca": [200], "ic": [220], "ti": [320]}
    rv = RelativeVerifier(template, verification)
    print(rv.calculate_disorder())
    assert rv.calculate_disorder() == 8 / 12

    template = {"tic": [370], "ica": [430], "the": [450]}
    verification = {"the": [340], "ica": [420], "tic": [540]}
    rv = AbsoluteVerifier(template, verification)
    print(rv.find_match_percent())
    assert rv.find_match_percent() == (1 - (2 / 3))

    template = {"ic": [150], "he": [220], "th": [230], "ti": [265], "ca": [280]}
    verification = {"th": [150], "he": [190], "ca": [200], "ic": [220], "ti": [320]}
    rv = AbsoluteVerifier(template, verification)
    print(rv.find_match_percent())
    assert rv.find_match_percent() == 2 / 5


def all_users_tests():
    print(
        "================================================================ R Verifier (KIT F4 and KHHT) ================================================="
    )
    for i in range(1, 28):
        kht_template = create_kht_template_for_user_and_platform_id(i, 1)
        kht_verification = create_kht_verification_attempt_for_user_and_platform_id(
            i, 1
        )
        kit_template = create_kit_flight_template_for_user_and_platform(i, 4, 1)
        kit_verification = create_kit_flight_verification_attempt_for_user_and_platform(
            i, 4, 1
        )

        s_verifier = SimilarityVerifier(kit_template, kit_verification)
        # print(s_verifier.find_match_percent())
        kht_r_verifier = RelativeVerifier(kht_template, kht_verification)
        kit_r_verifier = RelativeVerifier(kit_template, kit_verification)
        print("KIT R Verifier: " + str(kit_r_verifier.calculate_disorder()))
        print("KHT Verifier:" + str(kht_r_verifier.calculate_disorder()))
    print(
        "================================================================ S Verifier KHT ================================================="
    )
    for i in range(1, 28):
        kht_template = create_kht_template_for_user_and_platform_id(i, 1)
        kht_verification = create_kht_verification_attempt_for_user_and_platform_id(
            i, 1
        )
        s_verifier = SimilarityVerifier(kht_template, kht_verification)
        print(s_verifier.find_match_percent())
    print(
        "================================================================ S Verifier All KIT ================================================="
    )
    for i in range(1, 28):
        for j in range(1, 5):
            kht_template = create_kit_flight_template_for_user_and_platform(i, j, 1)
            kht_verification = (
                create_kit_flight_verification_attempt_for_user_and_platform(i, j, 1)
            )
            s_verifier = SimilarityVerifier(kht_template, kht_verification)
            print(s_verifier.find_match_percent())
        print()
    print(
        "================================================================ R Verifier All KIT ================================================="
    )
    for i in range(1, 28):
        for j in range(1, 5):
            kht_template = create_kit_flight_template_for_user_and_platform(i, j, 1)
            kht_verification = (
                create_kit_flight_verification_attempt_for_user_and_platform(i, j, 1)
            )
            r_verifier = RelativeVerifier(kht_template, kht_verification)
            print(r_verifier.calculate_disorder())
        print()


def cross_platform_r_test():
    flag = 0
    ids = list(range(1, 28))
    for id in ids:
        same_user_kht_template = create_kht_template_for_user_and_platform_id(id, 3)
        same_user_kht_verification = (
            create_kht_verification_attempt_for_user_and_platform_id(id, 3)
        )
        r_verifier = SimilarityVerifier(
            same_user_kht_template, same_user_kht_verification
        )
        same_user_match_score = r_verifier.find_match_percent()
        for remaining_id in ids:
            other_user_verification = (
                create_kht_verification_attempt_for_user_and_platform_id(
                    remaining_id, 3
                )
            )
            r_verifier = SimilarityVerifier(
                same_user_kht_template, other_user_verification
            )
            # print(
            #     f"Cross user KHT disorder for id {str(chosen_user)} and {str(remaining_id)} is {str(s_verifier.find_match_percent())}"
            # )
            if r_verifier.find_match_percent() > same_user_match_score:
                flag += 1

    print(f"{flag}/ {len(ids)**2} found")


def cross_platform_s_test(start_id, end_id):
    flag = 0
    ids = list(range(1, 28))
    for id in ids:
        same_user_kht_template = create_kht_template_for_user_and_platform_id(
            id, start_id
        )
        same_user_kht_verification = (
            create_kht_verification_attempt_for_user_and_platform_id(id, start_id)
        )
        r_verifier = RelativeVerifier(
            same_user_kht_template, same_user_kht_verification
        )
        same_user_match_score = r_verifier.calculate_disorder()
        for remaining_id in ids:
            other_user_verification = create_kht_template_for_user_and_platform_id(
                remaining_id, end_id
            )
            r_verifier = RelativeVerifier(
                same_user_kht_template, other_user_verification
            )
            # print(
            #     f"Cross user KHT disorder for id {str(chosen_user)} and {str(remaining_id)} is {str(s_verifier.find_match_percent())}"
            # )
            if r_verifier.calculate_disorder() > same_user_match_score:
                flag += 1

    print(f"{flag}/ {len(ids)**2} found")


# cross_user_r_kht_test()
# cross_user_s_kht_test()
# math_checks()
cross_platform_s_test(1, 3)

# TODO: Same platform: same user vs all the other users  (flag any values in cross user scores that are higher than the same user match score)
# Same as step 2 but with across platforms (f vs i)