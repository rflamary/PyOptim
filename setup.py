#!/usr/bin/env python

# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License

from setuptools import setup, find_packages
from codecs import open
from os import path
import re
import os

here = path.abspath(path.dirname(__file__))

# dirty but working
__version__ = '0.1'
# The beautiful part is, I don't even need to check exceptions here.
# If something messes up, let the build process fail noisy, BEFORE my release!

ROOT = os.path.abspath(os.path.dirname(__file__))


# convert markdown readme to rst in pypandoc installed
try:
   import pypandoc
   README = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   README = open(os.path.join(ROOT, 'README.md')).read()


setup(name='PyOptim',
      version=__version__,
      description='Python numerical optimization library',
      long_description=README,
      author=u'Remi Flamary',
      author_email='remi.flamary@gmail.com',
      url='https://github.com/rflamary/PyOptim',
      packages=find_packages(),
      platforms=['linux','macosx','windows'],
      download_url='https://github.com/rflamary/PyOptim/archive/v{}.tar.gz'.format(__version__),
      license = 'MIT',
      scripts=[],
      data_files=[],
      requires=["numpy","scipy"],
      install_requires=["numpy","scipy"],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Utilities'
    ]
     )
