from setuptools import setup, find_packages
from os.path import abspath, join, dirname


# read the contents of your README file
this_directory = abspath(dirname(__file__))
with open(join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='multi_view_da_ssl',
    version='0.0.1',
    description='A lightweight configer builder by given the config file or config str',
    author='Rick',
    #author_email='xxx@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license=None,
    url='https://github.com/TranNhiem/multi_augmentation_strategies_self_supervised_learning',
    packages=["Augment_Data_utils", "Augmentation_Strategies", "config", 
                "losses_optimizers", "Neural_Net_Architecture", "self_supervised_learning_frameworks"],
    keywords=["self-supervised", "mixing data augmentation", "multi view"],
    classifiers=[
        'Programming Language :: Python :: 3.8',
	    'Programming Language :: Python :: 3.9',
    ]
)