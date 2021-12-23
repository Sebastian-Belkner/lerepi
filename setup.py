import setuptools
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=''):
    config = Configuration('', parent_package, top_path)
    return config

setup(
    name='lerepi',
    version='0.0.0',
    packages=['lerepi'],
    url='https://github.com/carronj/cmbs4',
    author=['Julien Carron', 'Sebastian Belkner'],
    author_email=['to.jcarron@gmail.com', 'to.sebastian.belkner@unige.ch'],
    description='various iterative lensing estimation pipelines',
    install_requires=['healpy', 'numpy', 'plancklens'],
    long_description=long_description,
    configuration=configuration)

