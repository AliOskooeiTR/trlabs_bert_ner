
from setuptools import setup, find_packages, Command
import os


PYPI_REQUIREMENTS = []
if os.path.exists('requirements.txt'):
    for line in open('requirements.txt'):
        PYPI_REQUIREMENTS.append(line.strip())

setup(
    name='bert_ner',
    version='0.10',
    description=(
        'Named Entity Recognition with Bert'
    ),
    long_description=open('README.md').read(),
    author='Ali Oskooei',
    author_email='ali.oskooei@thomsonreuters.com',
    packages=find_packages('.'),
    install_requires=PYPI_REQUIREMENTS,
    zip_safe=False
)
