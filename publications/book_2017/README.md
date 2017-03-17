# python
These python scripts allow to reproduce all experiments that will be presented in the final book publication (2017) of the Jazzomat Research Project.

The experiments are split into
* **symbolic_analysis.py** - Experiments on symbolic features
* **audio_analysis** - Experiments on audio-based analysis features

In order to have all required python packages available, we create a conda environment which you can use.
We recommend to install [miniconda](https://conda.io/miniconda.html) to be able to use it.

    conda env create -f environment.yml
    source activate jazzomat_experiments
    python main.py
