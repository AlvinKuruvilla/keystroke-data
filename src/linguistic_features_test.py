import os
from tqdm import tqdm
from features.linguistic_features import (
    estimate_shannon_entropy,
    plot_multi_bar_graph,
)
from features.ndtw import CSM

from features.word_parser import get_user_by_platform, plot_heatmap, SentenceParser


def shannon_entropy_similarity():
    fb = []
    insta = []
    twitter = []
    for i in range(1, 28):
        df = get_user_by_platform(i, 1)
        sp = SentenceParser(os.path.join(os.getcwd(), "cleaned.csv"))
        words = sp.get_words(df)
        fb.append(estimate_shannon_entropy(words))

        df = get_user_by_platform(i, 2)
        sp = SentenceParser(os.path.join(os.getcwd(), "cleaned.csv"))
        words = sp.get_words(df)
        insta.append(estimate_shannon_entropy(words))

        df = get_user_by_platform(i, 3)
        sp = SentenceParser(os.path.join(os.getcwd(), "cleaned.csv"))
        words = sp.get_words(df)
        twitter.append(estimate_shannon_entropy(words))
    plot_multi_bar_graph(fb, insta, twitter)


# matrix = []
# for i in tqdm(range(1, 28)):
#     df = get_user_by_platform(i, 1)
#     sp = SentenceParser(os.path.join(os.getcwd(), "cleaned.csv"))
#     document_words = sp.get_words(df)
#     row = []
#     for j in range(1, 28):
#         df = get_user_by_platform(j, 1)
#         sp = SentenceParser(os.path.join(os.getcwd(), "cleaned.csv"))
#         query_words = sp.get_words(df)
#         csm = CSM(query_words, document_words)
#         row.append(csm.calculate_csm())
#     matrix.append(row)
# plot_heatmap(matrix, "CSM Facebook-Facebook")
shannon_entropy_similarity()
