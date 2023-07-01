from taaled import ld
import collections
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import wordnet as wn


def plot_multi_bar_graph(fb_data, insta_data, twitter_data):
    x = list(range(1, 28))
    df = pd.DataFrame(np.c_[fb_data, insta_data, twitter_data], index=x)
    df.plot.bar()

    plt.savefig("Shannon Entropy.png")


def mattr_for_words(words):
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


# Linguistic Features from: https://par.nsf.gov/servlets/purl/10282263
