
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python_fractal_scaling",
    version="0.0.1",
    author="Andrew Stier",
    author_email="andrewstier@uchicago.edu",
    description="python simple dfa",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enlberman/python_fractal_scaling",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)