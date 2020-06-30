#!/usr/bin/python
import os
from setuptools import setup, find_packages
from setuptools.command.install import install
import site
    
setup(
    name = "relight",
    version = "0.5",
    author = "Morris Franken",
    author_email = "morrisfranken@gmail.com",
    description = ("Relight faces based on normal map"),
    packages=find_packages(),
    package_data={'': ['datagen.blend']},
    include_package_data=True,
)
