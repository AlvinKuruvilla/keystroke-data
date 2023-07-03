from verifiers.heatmap import HeatMap, VerifierType

heatmap = HeatMap(VerifierType.Similarity)
matrix = heatmap.make_kht_matrix(1, 1, None, None)
print(matrix)
heatmap.plot_heatmap(matrix, "FF Similarity (new file)")
# matrix = heatmap.make_kht_matrix(2, 2, None, None)
# heatmap.plot_heatmap(matrix, "II")

# matrix = heatmap.make_kht_matrix(3, 3, None, None)
# heatmap.plot_heatmap(matrix, "TT")
