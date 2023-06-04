import pandas as pd

# Use the following function to plot the heatmap for
# each session, platform, and verifier.
def plot_heatmap(reference, probe, verifier):

    # reference and probe here contain a tuple (platform, session)
    ref_platform = reference[0]
    ref_session = reference[1]
    prob_platform = probe[0]
    prob_session = probe[1]
    ref_data = pd.read_csv()
    prob_data = pd.read_csv()
    heat_matrix = []

    for ref_user in ref_data:
        row = []
        for prob_user in prob_data:
            ref_features = get_features(ref_user)
            prob_feature = get_features(prob_user)
            row.append(verifier.get_match_score(ref_features, prob_feature)) # any
        heat_matrix.append(row)

    plot(heat_matrix)



