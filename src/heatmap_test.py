from verifiers.heatmap import HeatMap, VerifierType


heatmap = HeatMap(VerifierType.SimilarityUnweighted)
matrix = heatmap.make_kht_matrix(1, 3, 1, 3)
print(matrix)
heatmap.plot_heatmap(matrix)
