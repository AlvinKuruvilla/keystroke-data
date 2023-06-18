from verifiers.heatmap import HeatMap, VerifierType, create_kit_data_from_df
from verifiers.template_generator import read_compact_format

# i = 0
# for key, value_set in res.items():
#     i += 1
#     print(key, len(value_set))
#     print(i)
heatmap = HeatMap(VerifierType.Itad)
# matrix = heatmap.make_kht_matrix(1, 1, None, None)
# print(matrix)
# # input()
# heatmap.plot_heatmap(matrix, "FF KHT ITAD FS")
# matrix = heatmap.make_kht_matrix(2, 2, None, None)
# heatmap.plot_heatmap(matrix, "II KHT ITAD FS")

# matrix = heatmap.make_kht_matrix(3, 3, None, None)
# heatmap.plot_heatmap(matrix, "TT KHT ITAD FS")

# matrix = heatmap.make_kit_matrix(1, 1, None, None, 1)
# print(matrix)
# heatmap.plot_heatmap(matrix, "FF KIT1 ITAD")

combined_matrix = heatmap.make_combined_kht_kit_matrix(1, 1, None, None, 1)
print(combined_matrix)
heatmap.plot_heatmap(combined_matrix, "FF Combined ITAD with FS")
