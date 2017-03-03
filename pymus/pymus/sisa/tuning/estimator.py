from .tuning_nnls_chroma import TuningEstimatorMauch

__author__ = 'Jakob Abesser'


class TuningEstimator:
    """ Wrapper class for tuning estimation algorithms
    """

    @staticmethod
    def process(**options):
        """ Call tuning estimator
        """
        if options['tuning_estimation_method'] == 'mauch_nnls':
            return TuningEstimatorMauch.process(options['fn_wav'])
        else:
            raise Exception("Tuning estimation method not implemented!")


