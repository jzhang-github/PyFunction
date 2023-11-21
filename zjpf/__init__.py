# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:35:39 2023

@author: ZHANG Jun
"""

__version__ = '0.0.3'

import np, math, RGI, os
from .f import *

# remove redundant variables from the global memory.
del np, math, RGI, os