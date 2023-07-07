import os
import enum
import seaborn as sns
import collections
from taaled import ld
from scipy.stats import entropy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import wordnet as wn
from features.word_parser import SentenceParser
from scipy.spatial import distance

from verifiers.heatmap import get_user_by_platform


class LinguisticFeature(enum.Enum):
    MATTR = 1
    MTLDO = 2
    Shannon = 3


def plot_multi_bar_graph(fb_data, insta_data, twitter_data):
    x = list(range(1, 28))
    df = pd.DataFrame(np.c_[fb_data, insta_data, twitter_data], index=x)
    df.plot.bar()

    plt.savefig("Shannon.png")


def mattr_for_words(words):
    if len(words) == 0:
        return 0
    ldvals = ld.lexdiv(words)
    return ldvals.mattr


def word_counts(tokens):
    """Tokenize the List of str `tokens` and accumulate counts in a dictionary for each
    word that appears.

    Parameters
    ----------
    tokens : str
        The input List of string tokens.

    Returns
    -------
    dict of { str : int }
        Word counts in the string, with words as keys and their corresponding
        counts as values.
    """
    counts = collections.Counter()
    for token in tokens:
        counts[token] += 1
    return counts


def type_token_ratio(tokens):
    """Calculate the type-token ratio on the tokens from a string. The type-token
    ratio is defined as the number of unique word types divided by the number
    of total words.

    Parameters
    ----------
    tokens : List
        The input set of tokens to process.

    Returns
    -------
    float
        A decimal value for the type-token ratio of the words in the string.
    """
    counts = word_counts(tokens)
    type_count = len(counts.keys())
    token_count = sum(counts.values())
    return type_count / token_count


# Refrence implementation: https://lcr-ads-lab.github.io/TAALED/ld_indices/1.%20Revised_LD_indices.html#mtld

# MTLD measures the average number of tokens it takes for the TTR value to reach
# a point of stabilization (which McCarthy & Jarvis, 2010 defined as TTR = .720).
# Texts that are less lexically diverse will have lower MTLD values than more
# lexically diverse texts.


# MTLD is calculated by determining the number of factors in a text (the number
# of non-overlapping portions of text that reach TTR = .720) and the length of
# each factor. In most texts, a partial factor will occur at the end of the
# text. MTLD runs both forwards and backwards.
def mtldo_for_words(words):
    if len(words) == 0:
        return 0
    ldvals = ld.lexdiv(words)
    return ldvals.mtldo


def estimate_shannon_entropy(sequence):
    bases = collections.Counter([tmp_base for tmp_base in sequence])
    # define distribution
    dist = [x / sum(bases.values()) for x in bases.values()]

    # use scipy to calculate entropy
    entropy_value = entropy(dist, base=2)

    return entropy_value


# NOTE: These sets of functions do not work very well because the words we
# generate are often partial and incorrect, so they are not registered as nouns,
# adjectives, verbs, and adverbs


def is_noun(token: str) -> bool:
    nouns = {x.name().split(".", 1)[0] for x in wn.all_synsets("n")}
    return token in nouns


def is_verb(token: str) -> bool:
    classification = wn.synsets(token)
    print(classification)
    # Sometimes the classidication will be an empty list so we have to explicitly check for that case and just return False
    if len(classification) > 0:
        return classification[0].pos() == "v"
    return False


def is_adjective(token: str):
    classification = wn.synsets(token)
    # Sometimes the classidication will be an empty list so we have to explicitly check for that case and just return False
    if len(classification) > 0:
        return classification[0].pos() == "a"
    return False


def is_adverb(token: str):
    classification = wn.synsets(token)
    # Sometimes the classidication will be an empty list so we have to explicitly check for that case and just return False
    if len(classification) > 0:
        return classification[0].pos() == "r"
    return False


def is_lexical_word(token: str):
    "A lexical word is defined such as nouns, adjectives, verbs, and adverbs that convey meaning in a text"
    if is_verb(token) or is_adjective(token) or is_adverb(token) or is_noun(token):
        return True
    return False


def lexical_diversity(tokens):
    "Take the number of lexical words divided by the total word count"
    word_count = len(tokens)
    lexical_word_count = 0
    for token in tokens:
        print("token:", token)
        if is_lexical_word(token):
            lexical_word_count += 1
        return lexical_word_count / word_count


def create_linguistic_feature_matrix(
    enroll_platform_id,
    probe_platform_id,
    enroll_session_id,
    probe_session_id,
    linguistic_feature,
):
    if not 1 <= enroll_platform_id <= 3 or not 1 <= probe_platform_id <= 3:
        raise ValueError("Platform ID must be between 1 and 3")
    matrix = []
    # TODO: We have to do a better job of figuring out how many users there
    # are automatically so we don't need to keep changing it manually
    for i in tqdm(range(1, 26)):
        df = get_user_by_platform(i, enroll_platform_id, enroll_session_id)
        sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
        enrollment_words = sp.get_words(df)
        row = []
        for j in range(1, 26):
            df = get_user_by_platform(j, probe_platform_id, probe_session_id)
            sp = SentenceParser(os.path.join(os.getcwd(), "cleaned2.csv"))
            probe_words = sp.get_words(df)
            if linguistic_feature == LinguisticFeature.MATTR:
                row.append(
                    distance.cosine(
                        [mattr_for_words(enrollment_words)],
                        [mattr_for_words(probe_words)],
                    )
                )
            elif linguistic_feature == LinguisticFeature.MTLDO:
                row.append(
                    distance.cosine(
                        [mtldo_for_words(enrollment_words)],
                        [mtldo_for_words(probe_words)],
                    )
                )
            elif linguistic_feature == LinguisticFeature.Shannon:
                row.append(
                    distance.cosine(
                        [estimate_shannon_entropy(enrollment_words)],
                        [estimate_shannon_entropy(probe_words)],
                    )
                )
        matrix.append(row)
    return matrix


def plot_heatmap(matrix, title=None):
    sns.heatmap(matrix, linewidth=0.5).set_title(title)
    plt.savefig(title)
