import enum
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from verifiers.template_generator import read_compact_format
from verifiers.verifiers import Verifiers


class VerifierType(enum.Enum):
    Relative = 1
    Similarity = 2
    SimilarityUnweighted = 3
    Absolute = 4


def get_user_by_platform(user_id, platform_id, session_id=None):
    df = read_compact_format()
    if session_id is None:
        return df[(df["user_ids"] == user_id) & (df["platform_id"] == platform_id)]
    return df[
        (df["user_ids"] == user_id)
        & (df["platform_id"] == platform_id)
        & (df["session_id"] == session_id)
    ]


def create_kht_data_from_df(df):
    kht_dict = defaultdict(list)
    for i, row in df.iterrows():
        kht_dict[row["key"]].append(row["release_time"] - row["press_time"])
    return kht_dict


def create_kit_data_from_df(df, kit_feature_type):
    kit_dict = defaultdict(list)
    for i, g in df.groupby(df.index // 2):
        if g.shape[0] == 1:
            continue
        key = g.iloc[0]["key"] + g.iloc[1]["key"]
        initial_press = g.iloc[0]["press_time"]
        second_press = g.iloc[1]["press_time"]
        initial_release = g.iloc[0]["release_time"]

        second_release = g.iloc[1]["release_time"]
        if kit_feature_type == 1:
            kit_dict[key].append(float(second_press) - float(initial_release))
        elif kit_feature_type == 2:
            kit_dict[key].append(float(second_release) - float(initial_release))
        elif kit_feature_type == 3:
            kit_dict[key].append(float(second_press) - float(initial_press))
        elif kit_feature_type == 4:
            kit_dict[key].append(float(second_release) - float(initial_press))
    return kit_dict


class HeatMap:
    def __init__(self, verifier_type):
        self.verifier_type = verifier_type  # The verifier class to be used

    def make_kht_matrix(
        self, enroll_platform_id, probe_platform_id, enroll_session_id, probe_session_id
    ):
        if not 1 <= enroll_session_id <= 6 or not 1 <= probe_session_id <= 6:
            raise ValueError("Session ID must be between 1 and 6")
        if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
            raise ValueError("Platform ID must be between 1 and 3")

        matrix = []
        for i in range(1, 28):
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            enrollment = create_kht_data_from_df(df)
            row = []
            for j in range(1, 28):
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                probe = create_kht_data_from_df(df)
                v = Verifiers(enrollment, probe)
                if self.verifier_type == VerifierType.Absolute:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.Similarity:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SimilarityUnweighted:
                    row.append(v.get_similarity_score())
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
        if not 1 <= enroll_session_id <= 6 or not 1 <= probe_session_id <= 6:
            raise ValueError("Session ID must be between 1 and 6")
        if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
            raise ValueError("Platform ID must be between 1 and 3")
        if not 1 <= kit_feature_type <= 4:
            raise ValueError("KIT feature type must be between 1 and 4")
        matrix = []
        for i in range(1, 28):
            df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
            enrollment = create_kit_data_from_df(df, kit_feature_type)
            row = []
            for j in range(1, 28):
                df = get_user_by_platform(j, probe_platform_id, probe_session_id)
                probe = create_kit_data_from_df(df, kit_feature_type)
                v = Verifiers(enrollment, probe)
                if self.verifier_type == VerifierType.Absolute:
                    row.append(v.get_abs_match_score())
                elif self.verifier_type == VerifierType.Similarity:
                    row.append(v.get_weighted_similarity_score())
                elif self.verifier_type == VerifierType.SimilarityUnweighted:
                    row.append(v.get_similarity_score())
            matrix.append(row)
        return matrix

    def plot_heatmap(self, matrix):
        ax = sns.heatmap(matrix, linewidth=0.5)
        plt.show()
