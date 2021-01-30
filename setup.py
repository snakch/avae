# from distutils.file_util import copy_file
from setuptools import setup, find_packages


setup(
    name="avae",
    tests_require=["pytest"],
    version="0.1",
    packages=find_packages(include=["src/avae"]),
    package_dir={"": "src"},
)
