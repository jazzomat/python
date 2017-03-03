import os

from experiments import AudioAnalysisExperiments

__author__ = 'Jakob Abesser'

""" This script allows to reproduce all experiment reported in the Jazzomat Research Project final publication 2017 """


if __name__ == '__main__':
    dir_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    dir_results = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results')
    extractor = AudioAnalysisExperiments(dir_data, dir_results)
    extractor.run()
