import setuptools, re, glob
from setuptools import setup, find_packages

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
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Utilities'],
    packages=find_packages(),
    package_data={'prisim': ['examples/simparms/*.yaml',
                             'examples/schedulers/*.txt',
                             'data/catalogs/*.txt', 'data/catalogs/*.csv',
                             'data/catalogs/*.fits', 'data/beams/*.hmap',
                             'data/beams/*.hmap',
                             'data/phasedarray_layouts/*.txt',
                             'data/array_layouts/*.txt',
                             'data/bandpass/*.fits']},
    include_package_data=True,
    scripts=glob.glob('scripts/*'),
    install_requires=['astropy>=1.0', 'astroutils>=0.1.0', 'healpy>=1.5.3',
                      'ipdb>=0.6.1', 'matplotlib>=1.4.3', 'mpi4py>=1.2.2',
                      'numpy>=1.8.1', 'progressbar>=2.3', 'psutil>=2.2.1',
                      'pyephem>=3.7.5.3', 'pyyaml>=3.11', 'scipy>=0.15.1'],
    setup_requires=['astropy>=1.0', 'astroutils>=0.1.0', 'healpy>=1.5.3',
                    'ipdb>=0.6.1', 'matplotlib>=1.4.3', 'mpi4py>=1.2.2',
                    'numpy>=1.8.1', 'progressbar>=2.3', 'psutil>=2.2.1',
                    'pyephem>=3.7.5.3', 'pytest-runner', 'pyyaml>=3.11',
                    'scipy>=0.15.1'],
    tests_require=['pytest'],
    zip_safe=False)

