# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:22:58 2023

@author: ZHANG Jun
"""

from distutils.core import setup
from setuptools import find_packages

with open("README.md", "r") as f:
  long_description = f.read()

# print(long_description)
# with open('requirements.txt', "r") as f:
#     requirements = f.readlines()
#     requirements = [x.strip() for x in requirements]

# long_description = (this_directory / "README.md").read_text()

setup(name='zjpf',
      version='0.0.2',
      python_requires='>=3.10',
      description='ZHANG Jun Python functions.',
      long_description=long_description,
      include_package_data=True,
      long_description_content_type='text/markdown',
      author='ZHANG Jun',
      author_email='j.zhang@my.cityu.edu.hk',
      url='https://github.com/jzhang-github/PyFunction',
      install_requires=['numpy',
                        'scipy'],
      license='MIT',
      packages=find_packages('zjpf'),
      platforms=["all"],
      classifiers=[
                # How mature is this project? Common values are
                #   3 - Alpha
                #   4 - Beta
                #   5 - Production/Stable
                'Development Status :: 3 - Alpha',
                'License :: OSI Approved :: GNU General Public License (GPL)',
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.10',
          'Topic :: Software Development :: Libraries'
      ],
      )
