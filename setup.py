import os
import pathlib

from setuptools import setup

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

REQUIREMENTS = [
    "attrs==21.2",
    "numpy==1.21.2",
    "pyscf==2.1.1",
]

with open(PATH / "pyppm" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break

with open("README.md", "r") as readme:
    LONG_DESCRIPTION = readme.read()


# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="pyppm",
    version="0.1.1",
    author="""
    Daniel Bajac,
    """,
    author_email="""
    danielbajac94@gmail.com
    """,
    packages=["pyppm"],
    install_requires=REQUIREMENTS,
    license="The GPLv3 License",
    description="Quantum entanglement between virtual excitations",
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    url="https://github.com/DanielBajac/pyppm",
    keywords=[
        "CLOPPA",
        "Spin Spin Contact",
        "Entanglement",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
    include_package_data=True,
)
