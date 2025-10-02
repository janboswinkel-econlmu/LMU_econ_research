from setuptools import setup, find_packages

setup(
    name="superlmu", 
    version="0.1.0",
    description="A python package for LMU Econ Research on historical documents",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "Pillow",
        "openai",
        "requests"
    ],
)