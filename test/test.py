# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:28:14 2024

@author: 18326
"""

from zjpf.f import get_ase_atom_from_formula_template
from ase.io import write
from ase.build import sort
import os

atoms = get_ase_atom_from_formula_template(
    'V40Nb15Ta45C100',
    template_file=os.path.join('not_used', 'POSCAR_NbTa.txt'),
    exclude_shuffle_elements=['C'])
atoms = sort(atoms)
write(os.path.join('not_used', 'POSCAR_test'), atoms)
