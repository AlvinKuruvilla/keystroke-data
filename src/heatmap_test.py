from verifiers.heatmap import HeatMap, VerifierType


heatmap = HeatMap(VerifierType.Absolute)
matrix = heatmap.make_matrix(1, 3, 1, 3)
print(matrix)
heatmap.plot_heatmap(matrix)
