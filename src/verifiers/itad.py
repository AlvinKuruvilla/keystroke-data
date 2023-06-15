import statistics
import numpy as np


def ecdf(data):
    """Compute ECDF"""
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    return (x, y)


class ITADVerifier:
    def __init__(self, enrollment, probe) -> None:
        self.enrollment = enrollment
        self.probe = probe

    def get_common_keys(self):
        common_keys = set(self.enrollment.keys()).intersection(set(self.probe.keys()))
        return common_keys

    def get_common_key_count(self):
        return len(self.get_common_keys())

    def get_median_of_common_key(self, key):
        if not key in self.get_common_keys():
            raise ValueError(str(key) + " is not in common keys")
        return statistics.median(self.enrollment[key])

    def x_i(self, key):
        return statistics.mean(self.probe[key])

    def ecdf_of_x(self, key, x_i):
        # The ITAD algorithm requires that we get CDF(x_i)
        # To do that we first get all the timing values of the key
        # We then append the (at the point of calling this function) calculated x_i value

        # After getting the y values for the data by calling the ecdf function, we get the specific y_value for x_i by assuming
        # that each y_value in the list is a 1-1 match to sorted x_values. So by finding the index of x_i in the sorted x_values
        # we can find the corresponding y value by y_vals[x_i_index]
        data = list(self.probe[key])
        data.append(x_i)
        x_vals, y_vals = ecdf(data)
        data = list(np.sort(data))
        x_i_index = data.index(x_i)
        return y_vals[x_i_index]

    def itad_metric(self, key, sample_duration):
        # In the context of this function, the sample_duration is x_i in the algorithm (in our case, it will be the probe mean)
        m_x = self.get_median_of_common_key(key)
        if sample_duration <= m_x:
            return self.ecdf_of_x(key, sample_duration)
        return 1 - self.ecdf_of_x(key, sample_duration)

    def itad_similarity(self, p=0.5):
        S = 0
        total = 0
        matching_keys = self.get_common_keys()
        for key in matching_keys:
            x = self.x_i(key)
            itad_value = self.itad_metric(key, x) * p
            total += itad_value
        try:
            return (1 / self.get_common_key_count()) * total
        except ZeroDivisionError:
            return 0
