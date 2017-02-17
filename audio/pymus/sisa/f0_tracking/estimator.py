from .f0_tracking_peak_tracking import F0TrackerPeakTrackingAbesserDAFX2014

__author__ = 'Jakob Abesser'


class F0Tracker:
    """ Wrapper function for fundamental frequency tracker
    """

    def __init__(self):
        pass

    @staticmethod
    def process(samples,
                sample_rate,
                pitch,
                onset,
                duration,
                **options):

        if options['f0_tracking_method'] == 'abesser_dafx_2014':
            return F0TrackerPeakTrackingAbesserDAFX2014().process(samples,
                                                                  sample_rate,
                                                                  pitch,
                                                                  onset,
                                                                  duration,
                                                                  tuning_frequency=options['tuning_frequency'])
        else:
            raise Exception("F0 tracking method not yet implemented!")


