import os

from audio_experiments import AudioAnalysisExperiments
from symbolic_experiments import SymbolicAnalysisExperiments

__author__ = 'Jakob Abesser'

""" This script allows to reproduce all experiment reported in the Jazzomat Research Project final publication 2017 """


if __name__ == '__main__':
    dir_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    dir_results = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results')
    for extractor in [SymbolicAnalysisExperiments(dir_data, dir_results),]:
                      # AudioAnalysisExperiments(dir_data, dir_results)]:
        extractor.run()
