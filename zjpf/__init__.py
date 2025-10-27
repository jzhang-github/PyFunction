# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:35:39 2023

@author: ZHANG Jun
"""

__version__ = '0.0.5'

from .f import read_chg, flatten_new, fractional2cartesian, cartesian2fractional, file_exit, get_con_dict_from_ase, get_cutoff, get_atom_index_in_layer, shift_fcoords2, get_atomic_diameter, get_centroid, modify_INCAR, pad_dict_list, get_density

from .f import get_concentration_from_ase_formula, get_dos

from .f import get_ase_atom_from_formula_template, scale_atoms

from .f import remove_pd_outlier

# remove redundant variables from the global memory.
