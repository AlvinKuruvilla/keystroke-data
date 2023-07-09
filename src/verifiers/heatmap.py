import enum
import matplotlib.pyplot as plt
import seaborn as sns
from features.keystroke_features import create_kht_data_from_df, create_kit_data_from_df
from verifiers.template_generator import read_compact_format
from verifiers.verifiers import Verifiers
from tqdm import tqdm


class VerifierType(enum.Enum):
    Relative = 1
    Similarity = 2
    SimilarityUnweighted = 3
    Absolute = 4
    Itad = 5


def get_user_by_platform(user_id, platform_id, session_id=None):
    print(f"user_id:{user_id}")
    df = read_compact_format()
    if session_id is None:
        return df[(df["user_ids"] == user_id) & (df["platform_id"] == platform_id)]
    return df[
        (df["user_ids"] == user_id)
        & (df["platform_id"] == platform_id)
        & (df["session_id"] == session_id)
    ]


class HeatMap:
    def __init__(self, verifier_type):
        self.verifier_type = verifier_type  # The verifier class to be used

    def make_kht_matrix(
        self, enroll_platform_id, probe_platform_id, enroll_session_id, probe_session_id
    ):
        if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
            raise ValueError("Platform ID must be between 1 and 3")

        matrix = []
        # TODO: We have to do a better job of figuring out how many users there
        # are automatically so we don't need to keep changing it manually
        ids = [num for num in range(1, 26) if num != 22]
        for i in ids:
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            enrollment = create_kht_data_from_df(df)
            row = []
            # TODO: We have to do a better job of figuring out how many users there
            # are automatically so we don't need to keep changing it manually
            for j in ids:
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                probe = create_kht_data_from_df(df)
                v = Verifiers(enrollment, probe)
                if self.verifier_type == VerifierType.Absolute:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.Similarity:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SimilarityUnweighted:
                    row.append(v.get_similarity_score())
                elif self.verifier_type == VerifierType.Itad:
                    row.append(v.itad_similarity())
                else:
                    raise ValueError(
                        "Unknown VerifierType {}".format(self.verifier_type)
                    )
            matrix.append(row)
        return matrix

    def make_kit_matrix(
        self,
        enroll_platform_id,
        probe_platform_id,
        enroll_session_id,
        probe_session_id,
        kit_feature_type,
    ):
        if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
            raise ValueError("Platform ID must be between 1 and 3")
        if not 1 <= kit_feature_type <= 4:
            raise ValueError("KIT feature type must be between 1 and 4")
        matrix = []
        ids = [num for num in range(1, 26) if num != 22]
        for i in tqdm(ids):
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            enrollment = create_kit_data_from_df(df, kit_feature_type)
            row = []
            for j in ids:
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                probe = create_kit_data_from_df(df, kit_feature_type)
                v = Verifiers(enrollment, probe)
                if self.verifier_type == VerifierType.Absolute:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.Similarity:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SimilarityUnweighted:
                    row.append(v.get_similarity_score())
                elif self.verifier_type == VerifierType.Itad:
                    row.append(v.itad_similarity())
                else:
                    raise ValueError(
                        "Unknown VerifierType {}".format(self.verifier_type)
                    )
            matrix.append(row)
        return matrix

    def combined_keystroke_matrix(
        self,
        enroll_platform_id,
        probe_platform_id,
        enroll_session_id,
        probe_session_id,
        kit_feature_type,
    ):
        if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
            raise ValueError("Platform ID must be between 1 and 3")
        if not 1 <= kit_feature_type <= 4:
            raise ValueError("KIT feature type must be between 1 and 4")
        matrix = []
        ids = [num for num in range(1, 26) if num != 22]
        for i in ids:
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            kht_enrollment = create_kht_data_from_df(df)
            kit_enrollment = create_kit_data_from_df(df, kit_feature_type)
            combined_enrollment = kht_enrollment | kit_enrollment
            row = []
            for j in ids:
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                kht_probe = create_kht_data_from_df(df)
                kit_probe = create_kit_data_from_df(df, kit_feature_type)
                combined_probe = kht_probe | kit_probe
                v = Verifiers(combined_enrollment, combined_probe)
                if self.verifier_type == VerifierType.Absolute:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.Similarity:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SimilarityUnweighted:
                    row.append(v.get_similarity_score())
                elif self.verifier_type == VerifierType.Itad:
                    row.append(v.itad_similarity())
                else:
                    raise ValueError(
                        "Unknown VerifierType {}".format(self.verifier_type)
                    )
            matrix.append(row)
        return matrix

    def plot_heatmap(self, matrix, title=None):
        ax = sns.heatmap(matrix, linewidth=0.5).set_title(title)
        plt.savefig(title)
