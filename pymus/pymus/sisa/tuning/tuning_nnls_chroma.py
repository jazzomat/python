import os

__author__ = 'Jakob Abesser'


class TuningEstimatorMauch:
    """ Class to call the Sonic Annotator commandline tool to estimate the tuning frequency using the NNLS chroma
        VAMP plugin, make sure you have sonic-annoator and the NNLS Vamp plugin installed
    """

    @staticmethod
    def process(fn_wav):
        """ Wrapper to call Sonic Annotator with NNLS-Chroma Tuning VAMP plugin to extract tuning frequency for WAV file
        :param fn_wav (string) : WAV file name
        :return: tuning_freq (float) : Tuning frequency in Hz
        """
        # generate sonic-annotator command and excecute it
        fn_csv = os.path.join(os.path.dirname(fn_wav),
                              'tuning.csv')
        # add sonic annotator to PATH if necessary
        # os.system('export "PATH=/Applications/SonicAnnotator/:$PATH"')
        command = "sonic-annotator -d vamp:nnls-chroma:tuning:tuning \"{}\" -w csv --csv-one-file {} --csv-separator ';'".format(fn_wav,
                                                                                                                                 fn_csv)
        os.system(command)

        # read result (tuning frequency)
        with open(fn_csv, 'r') as f:
            content = f.readline().split(';')
        if os.path.isfile(fn_csv):
            os.remove(fn_csv)

        return float(content[3])
