import setuptools, re, glob, os
from setuptools import setup, find_packages
from subprocess import Popen, PIPE

githash = 'unknown'
if os.path.isdir(os.path.dirname(os.path.abspath(__file__))+'/.git'):
    try:
        gitproc = Popen(['git', 'rev-parse', 'HEAD'], stdout = PIPE)
        githash = gitproc.communicate()[0]
        if gitproc.returncode != 0:
            print "unable to run git, assuming githash to be unknown"
            githash = 'unknown'
    except EnvironmentError:
        print "unable to run git, assuming githash to be unknown"
githash = githash.replace('\n', '')

with open(os.path.dirname(os.path.abspath(__file__))+'/prisim/githash.txt', 'w+') as githash_file:
    githash_file.write(githash)

metafile = open(os.path.dirname(os.path.abspath(__file__))+'/prisim/__init__.py').read()
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
    package_data={'prisim': ['*.txt', 'examples/simparms/*.yaml',
                             'examples/schedulers/*.txt',
                             'examples/dbparms/*.yaml',
                             'examples/ioparms/*.yaml',
                             'data/catalogs/*.txt', 'data/catalogs/*.csv',
                             'data/catalogs/*.fits', 'data/beams/*.hmap',
                             'data/beams/*.txt', 'data/beams/*.hdf5',
                             'data/beams/*.FITS', 'data/array_layouts/*.txt',
                             'data/phasedarray_layouts/*.txt',
                             'data/bandpass/*.fits', 'data/bandpass/*.txt']},
    include_package_data=True,
    scripts=glob.glob('scripts/*.py'),
    install_requires=['astropy>=1.0', 'astroutils>=0.1.0', 'healpy>=1.5.3',
                      'ipdb>=0.6.1', 'matplotlib>=1.4.3', 'mpi4py>=1.2.2',
                      'numpy>=1.8.1', 'progressbar>=2.3', 'psutil>=2.2.1',
                      'pyephem>=3.7.5.3', 'pyyaml>=3.11', 'scipy>=0.15.1',
                      'pyuvdata==1.1'],
    setup_requires=['astropy>=1.0', 'astroutils>=0.1.0', 'healpy>=1.5.3',
                    'ipdb>=0.6.1', 'matplotlib>=1.4.3', 'mpi4py>=1.2.2',
                    'numpy>=1.8.1', 'progressbar>=2.3', 'psutil>=2.2.1',
                    'pyephem>=3.7.5.3', 'pytest-runner', 'pyyaml>=3.11',
                    'scipy>=0.15.1', 'pyuvdata==1.1'],
    tests_require=['pytest'],
    zip_safe=False)
