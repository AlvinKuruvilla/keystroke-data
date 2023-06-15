from verifiers.heatmap import HeatMap, VerifierType, create_kit_data_from_df
from verifiers.template_generator import read_compact_format

df = read_compact_format()

res = create_kit_data_from_df(df, 1)
print(res)
input()
heatmap = HeatMap(VerifierType.Itad)
matrix = heatmap.make_kht_matrix(1, 1, 1, 1)
print(matrix)
# input()
heatmap.plot_heatmap(matrix, "FF KHT ITAD")
# matrix = heatmap.make_kht_matrix(2, 2, None, None)
# heatmap.plot_heatmap(matrix, "II")

# matrix = heatmap.make_kht_matrix(3, 3, None, None)
# heatmap.plot_heatmap(matrix, "TT")
