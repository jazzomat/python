__author__ = 'Jakob Abesser'


class SonicVisualiserWrapper:
    """ Class provides wrapper functions to import / export results from Sonic Visualiser annotation layers """

    def __init__(self):
        pass

    @staticmethod
    def export_values_layer(fn_csv,
                            times,
                            values):
        """ Export values layers (timestamps & values
        :param fn_csv: (string) CSV file name
        :param times: (ndarray / list) Time stamps in seconds
        :param values: (ndarray / list) Values
        """
        assert len(times) == len(values)
        with open(fn_csv, 'w+') as f:
            for _ in range(len(times)):
                f.write("{},{}\n".format(times[_], values[_]))
