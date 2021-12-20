import pathlib
from setuptools import setup, find_packages

root_dirpath = pathlib.Path(__file__).parent

readme_filepath = (root_dirpath / "README.md").read_text()

setup(
    name="simpleicp",
    version="1.0.2",  # should match version in __init__.py
    description="Simple version of the Iterative Closest Point (ICP) algorithm",
    long_description=readme_filepath,
    long_description_content_type="text/markdown",
    url="https://github.com/pglira/simpleICP/tree/master/python",
    author="Philipp Glira",
    author_email="philipp.glira@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=("tests",)),
    install_requires=["numpy", "scipy"],
)
