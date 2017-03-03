from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='pymus',
      version='0.2.5',
      long_description=readme(),
      description='Tools for audio analysis, special focus on score-informed audio analysis of instrumental / vocal solo recordings',
      url='',
      keywords='score-informed audio analysis solo recordings tuning loudness intonation music information retrieval',
      author='Jakob Abesser',
      author_email='nudelsalat@posteo.de',
      license='MIT',
      packages=['pymus'],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)