from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


ext_modules = [
    Pybind11Extension(
        "_kzhutil",
        ['src/math.cpp'],  # Sort source files for reproducibility
    ),
]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='kzhutil',
    version='0.0.2',
    author='kuangzh',
    author_email='kuangzh@smail.nju.edu.cn',
    url='https://github.com/kuangkzh/kzhutil',
    description='some python utils',
    long_description=long_description,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    zip_safe=False,
)
