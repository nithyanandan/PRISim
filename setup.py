import setuptools
from setuptools import setup, find_packages
import re

metafile = open('./prisim/__init__.py').read()
metadata = dict(re.findall("__([a-z]+)__\s*=\s*'([^']+)'", metafile))

setup(name='PRISim',
    version=metadata['version'],
    description=metadata['description'],
    long_description=open("README.txt").read(),
    url=metadata['url'],
    author=metadata['author'],
    author_email=metadata['authoremail'],
    license='MIT',
    packages=find_packages(),
    package_data={'prisim': ['data/*.yaml']},
    include_package_data=True,
    scripts=['scripts/run_prisim.py'],
    install_requires=['astroutils>=0.1.0', 'psutil>=2.2.1'],
    setup_requires=['astroutils>=0.1.0', 'psutil>=2.2.1', 'pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False)

