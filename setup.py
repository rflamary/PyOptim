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
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    open('optim/__init__.py').read()).group(1)
# The beautiful part is, I don't even need to check exceptions here.
# If something messes up, let the build process fail noisy, BEFORE my release!

ROOT = os.path.abspath(os.path.dirname(__file__))


# convert markdown readme to rst in pypandoc installed
with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()


setup(name='PyOptim',
      version=__version__,
      description='Python numerical optimization library',
      long_description=README,
      long_description_content_type='text/markdown', 
      author=u'Remi Flamary',
      author_email='remi.flamary@gmail.com',
      url='https://github.com/rflamary/PyOptim',
      packages=find_packages(),
      platforms=['linux','macosx','windows'],
      download_url='https://github.com/rflamary/PyOptim/archive/v{}.tar.gz'.format(__version__),
      license = 'MIT',
      scripts=[],
      data_files=[],
      requires=["numpy","scipy",'cvxopt'],
      install_requires=["numpy","scipy",'cvxopt'],
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
