#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup
from os import path

# read from README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_descript = f.read()

setup(
  name='deepcure',
  version='0.1.0',
  description="Cure survival models with Tensorflow/Keras",
  long_description=long_descript,
  long_description_content_type='text/markdown',
  packages=find_packages(),
  author="Victor Medina-Olivares",    
  author_email='vitomedina@gmail.com',
  keywords=['survival analysis', 'deep learning', 'cure model', 'promotion time cure model'],
  url='https://github.com/vhmedina/deepcure',
  license='MIT License',
)