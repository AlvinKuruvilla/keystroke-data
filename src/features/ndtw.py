# NDTW (Nonuniform Distributed Term Weight) Measure
import math
from collections import Counter


class NDTWMetric:
    def __init__(self, words_list):
        self.words_list = words_list

    def total_occurrence_count(self, word):
        return Counter(self.words_list)[word]

    def calculate_ndtw(self, term):
        # I think it is okay to simply subtract by 1 here because ff(term, document) is
        # the same as total_occurrence_count(term), due to us only having 1 document
        # by each author
        # It doesn't abide exactly by the formula though as presented in the paper
        return math.log(self.total_occurrence_count(term)) - 1


class CSM:
    def __init__(self, query_words, document_words):
        # For our use case:
        # the query words are the probe
        # the document words are the enrollment
        self.query_words = query_words
        self.document_words = document_words

    def find_common_terms(self):
        return list(set(self.query_words).intersection(set(self.document_words)))

    def calculate_csm(self):
        terms = self.find_common_terms()
        if len(terms) == 0:
            return 0
        query_ndtw = NDTWMetric(self.query_words)
        doc_ndtw = NDTWMetric(self.document_words)
        numerator = 0
        denominator = 0
        for term in terms:
            print(query_ndtw.calculate_ndtw(term))
            numerator += query_ndtw.calculate_ndtw(term) * doc_ndtw.calculate_ndtw(term)
            denominator += math.sqrt(
                math.pow(query_ndtw.calculate_ndtw(term), 2)
            ) * math.sqrt(math.pow(doc_ndtw.calculate_ndtw(term), 2))
        return numerator / denominator
