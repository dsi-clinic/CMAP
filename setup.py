"""Setup script for 2024-winter-cmap package."""

from setuptools import find_packages, setup

setup(
    name="2024-winter-cmap",
    version="0.1.0",
    packages=find_packages(
        include=[
            "utils",
            "utils.*",
        ]
    ),
    install_requires=[],
)
