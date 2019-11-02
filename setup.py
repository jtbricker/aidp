from setuptools import setup, find_packages
import sys

if sys.version_info < (3,0):
    sys.exit('Sorry, Python 2 is not supported')

with open("README.md", "r") as readme_file:
    README = readme_file.read()

with open("requirements.txt", 'r') as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name="aidp",
    version="1.0.0",
    author="Justin Bricker",
    author_email="jt.bricker@gmail.com",
    description="A tool for interacting with AIDP model",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jtbricker/aidp/",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    entry_points={
        'console_scripts': [
            'aidp=main:main'
        ]
    }
)