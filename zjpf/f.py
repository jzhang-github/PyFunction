import numpy as np
import math
import os
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy import stats
import ase
from ase.io import read
from ase.formula import Formula
from ase.data import covalent_radii, atomic_numbers
import re
import pandas as pd

def read_chg(CHG_NAME='CHGCAR'):
    infile=open(CHG_NAME,"r")
    #outfile1=open("mytot","w")
    #outfile2=open("mymag","w")
    #outfile3=open("myCHGt","w")
    #outfile4=open("myslice","w")

    comment=infile.readline()
    A=float(infile.readline().split()[0])
    a=np.array([float(x) for x in infile.readline().split()])
    b=np.array([float(x) for x in infile.readline().split()])
    c=np.array([float(x) for x in infile.readline().split()])
    lx=A*np.sqrt(a[0]**2+a[1]**2+a[2]**2)
    ly=A*np.sqrt(b[0]**2+b[1]**2+b[2]**2)
    lz=A*np.sqrt(c[0]**2+c[1]**2+c[2]**2)
    vol=A*A*A*np.dot(np.cross(a,b),c)
    #print(lx,ly,lz,vol)

    #label=np.array([infile.readline().split()])
    label=np.array([str(x) for x in infile.readline().split()])
    #print(label)
    ntype=np.array([int(x) for x in infile.readline().split()])
    #print(ntype)

    N=np.sum(ntype)
    #print(N)

    dc=infile.readline()

    pos=[]
    for i in range(1,N+1):
      pos.append([float(x) for x in infile.readline().split()])
    #print("Number of positions being read", len(pos))

    tmp=infile.readline()
    (nx,ny,nz)=infile.readline().split()
    nx=int(nx)
    ny=int(ny)
    nz=int(nz)
    #print(nx,ny,nz)

    chgtot = np.fromfile(infile, count=nx*ny*nz, sep=' ')
    chgtot2 = np.reshape(chgtot, (nx,ny,nz), order='F')
    sumt=sum(chgtot)
    #print("Number of total electrons", sumt/nx/ny/nz)
    #############################################
    x_top   = [chgtot2[1,:,:]]                    #
    chgtot2 = np.append(chgtot2, x_top, axis=0) #
    y_top   = [chgtot2[:,1,:]]                    #
    y_top   = np.reshape(y_top, (nx+1,1,nz))      #
    chgtot2 = np.append(chgtot2, y_top, axis=1) #
    z_top   = [chgtot2[:,:,1]]                    #
    z_top   = np.reshape(z_top, (nx+1,ny+1,1))    #
    chgtot2 = np.append(chgtot2, z_top, axis=2) #
    #############################################

    myx=np.linspace(0,lx,nx+1)
    myy=np.linspace(0,ly,ny+1)
    myz=np.linspace(0,lz,nz+1)
    myinterp=RGI((myx,myy,myz),chgtot2)
    #print(len(myx),len(myy),len(myz))
    #print(nx,ny,nz)
    #print(lx-lx/nx)

    return myinterp, lx, ly, lz, vol

def flatten_new(arr:np.array, points=50):
    data_min = np.min(arr[:,0], axis=0)
    data_max = np.max(arr[:,0], axis=0)
    data_span = data_max - data_min
    step = data_span/points
    bins = np.linspace(data_min+step, data_max+step, points)
    new_bins = bins - step/2
    dig = np.digitize(arr[:,0], bins)

    new_arr = [[] for x in range(points)]
    for i, d in enumerate(dig):
        new_arr[d].append(arr[i])

    results = []
    for i, arr_tmp in enumerate(new_arr):
        new_arr_tmp = np.mean(arr_tmp, axis=0)
        results.append(new_arr_tmp)
    return np.array(results)

def fractional2cartesian(vector_tmp, D_coord_tmp):
    C_coord_tmp = np.dot(D_coord_tmp, vector_tmp)
    return C_coord_tmp

def cartesian2fractional(vector_tmp, C_coord_tmp):
    vector_tmp = np.mat(vector_tmp)
    D_coord_tmp = np.dot(C_coord_tmp, vector_tmp.I)
    D_coord_tmp = np.array(D_coord_tmp, dtype=float)
    return D_coord_tmp

class FileExit(Exception):
    pass

def file_exit():
    if os.path.exists('StopPython'):
        os.remove('StopPython')
        raise FileExit('Exit because `StopPython` file is found.')

def get_con_dict_from_ase(atoms):
    symbols = atoms.get_chemical_symbols()
    ele = list(set(symbols))
    counts = [symbols.count(x) for x in ele]
    num = len(atoms)
    con = [x/num for x in counts]
    return dict(zip(ele, con))

def get_concentration_from_ase_formula(formula):
    f_dict = Formula(formula).count()
    tot = np.sum(list(f_dict.values()))
    c_dict = {k: v/tot for k, v in f_dict.items()}
    c_dict = {k:c_dict[k] for k in c_dict if c_dict[k] > 0}
    return c_dict

def get_cutoff(atoms, crystal='FCC'):
    cell_volume    = atoms.get_volume()
    number_of_atoms = len(atoms)
    if crystal == 'FCC':
        atomic_density = 0.7404804896930611
    R = (cell_volume * atomic_density / number_of_atoms * 3 / 4 / np.pi) ** (1 / 3) * 2
    cut_off = (R + R * (2 ** 0.5)) / 2
    return cut_off

def get_atom_index_in_layer(atoms, layer_index, layer_num): # begain from zero.
    average_spacing = 1 / layer_num
    scaled_positions = atoms.get_scaled_positions()
    z = scaled_positions[:,2]
    lo_boundary = layer_index * average_spacing - average_spacing / 2
    hi_boundary = layer_index * average_spacing + average_spacing / 2
    return np.where((z > lo_boundary) & (z < hi_boundary))[0]

def shift_fcoords2(fcoord1, fcoord2, cutoff=0.5):
    """ Relocate fractional coordinate to the image of reference point. ``fcoord1`` is the reference point.

    :param fcoord1: the reference point
    :type fcoord1: numpy.ndarry
    :param fcoord2: coordinate will be shifted
    :type fcoord2: numpy.ndarry
    :param cutoff: cutoff difference to wrap two coordinates to the same periodic image.
    :type cutoff: float
    :return: new ``fcoord2`` in the same periodic image of the reference point.
    :rtype: numpy.ndarry

    .. Important:: Coordinates should be ``numpy.ndarray``.

    """
    shift_status = False
    diff        = fcoord1 - fcoord2
    transition  = np.where(diff >= cutoff, 1.0, 0.0)
    if np.isin(1.0, diff):
        shift_status = True
    fcoord2_new = fcoord2 + transition
    transition  = np.where(diff < -cutoff, 1.0, 0.0)
    if np.isin(1.0, diff):
        shift_status = True
    fcoord2_new = fcoord2_new - transition
    return fcoord2_new, shift_status

def get_atomic_diameter(ase_atoms, crystal_type='fcc'):
    """ Calculate atomic diameter of a bulk structure.

    :param ase_atoms: input structure.
    :type ase_atoms: ase.atoms.Atoms
    :param crystal_type: crystal type, defaults to 'fcc'. Other options: 'bcc', 'hcp', and 'cubic'.
    :type crystal_type: str, optional
    :return: atomic diameter
    :rtype: float

    """
    atomic_density = {'fcc'  : 0.7404804896930611,
                        'bcc'  : 0.6801747615878315,
                        'hcp'  : 0.7404804896930611,
                        'cubic': 0.5235987755982988}

    cell_volume = ase_atoms.get_volume()
    num_sites   = len(ase_atoms)
    diameter    = (cell_volume * atomic_density[crystal_type] / num_sites * 3 / 4 / np.pi) ** (1 / 3) * 2
    return diameter

def get_centroid(fcoords, ref_pos, cutoff=0.5, convergence=0.00001):
    """

    :param fcoords: DESCRIPTION
    :type fcoords: TYPE
    :param ref_pos: DESCRIPTION
    :type ref_pos: TYPE
    :param cutoff: DESCRIPTION, defaults to 0.5
    :type cutoff: TYPE, optional
    :param convergence: DESCRIPTION, defaults to 0.00001
    :type convergence: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    fcoords_tmp = fcoords.copy()
    num_coord = np.shape(fcoords_tmp)[0]

    for i in range(num_coord):
        fcoords_tmp[i], shift_status = shift_fcoords2(ref_pos, fcoords_tmp[i], cutoff=cutoff)

    centroid_tmp = np.sum(fcoords_tmp, axis=0) / num_coord
    return centroid_tmp

orbital_dict = {
                's':  [[1],[2]],
                'p':  [[3,5,7],[4,6,8]],
                'd':  [[9,11,13,15,17],[10,12,14,16,18]],
                'pd': [[9],[11],[13],[15],[17],[10],[12],[14],[16],[18]],
                }

def get_dos(index_list):
    # orbital_dict = {'s': [], 'p': [], 'd':[]}
    dos_up_all, dos_dn_all = [], []
    for i in index_list:
        assert os.path.exists('DOS'+str(i)), 'DOS'+str(i)+" not found."
        dos_tmp = np.loadtxt('DOS'+str(i))
        dos_up_all.append(np.sum(dos_tmp[:,orbital_dict['d'][0]], axis=1))
        dos_dn_all.append(np.sum(dos_tmp[:,orbital_dict['d'][1]], axis=1))
    energy = dos_tmp[:,0]
    dos_up = np.sum(dos_up_all, axis=0)
    dos_dn = np.sum(dos_dn_all, axis=0)
    dos    = np.array([energy, dos_up, dos_dn])
    return dos.T

def remove_pd_outlier(df: pd.core.frame.DataFrame, level=3) -> pd.core.frame.DataFrame:
    return df[(np.abs(stats.zscore(df, nan_policy='omit')) < level)]

INCAR_TAG = '''
SYSTEM
ISTART
ICHARG
INIWAV
ENCUT
ENAUG
PREC
IALGO
NELM
NELMIN
NELMDL
EDIFF
NBANDS
GGA
VOSKOWN
LREAL
WEIMIN
EDIFFG
NSW
IBRION
ISIF
POTIM
IOPT
ISYM
SIGMA
ISMEAR
ISPIN
MAGMOM
LWAVE
LCHARG
RWIGS
NPAR
LORBIT
LDAU
LDAUTYPE
LDAUL
LDAUU
LDAUJ
LDAUPRINT
LMAXMIX
LASPH
IDIPOL
LDIPOL
LAECHG
LADDGRID
NGX
NGY
NGZ
NGXF
NGYF
NGZF
ICHAIN
IMAGES
SPRING
LCLIMB
DdR
DRotMax
DFNMin
DFNMax
NFREE
LUSE_VDW
Zab_vdW
AGGAC
AMIX
AMIX_MAG
BMIX
BMIX_MAG
ALGO
KPAR
NCORE
NEDOS
IVDW
LELF
MDALGO
TEEND
SMASS
'''.split()

def modify_INCAR(working_dir='.', key='NSW', value='300', s=''):
    if not key in INCAR_TAG:
        print('Input key not avaliable, please check.')
        return 1

    new_incar, discover_code = [], False
    with open(os.path.join(working_dir, 'INCAR'), 'r') as f:
        for line in f:
            # str_list = line.split()
            str_list = re.split(' |#|=', line)
            str_list = [x for x in str_list if x != '']
            if len(str_list) == 0:
                new_incar.append('\n')
            elif str_list[0] == key:
                if value == '#':
                    new_incar.append(f'  # {str_list[0]} = {str_list[1]}\n')
                else:
                    new_incar.append(f'  {str_list[0]} = {value}\n')
                discover_code = True
            else:
                new_incar.append(line)

    if s:
        new_incar.append(f'\n{s}\n')

    if not discover_code:
        new_incar.append(f'  {key} = {value}\n')

    with open(os.path.join(working_dir, 'INCAR'), 'w') as f:
        for line in new_incar:
            f.write(line)

def pad_dict_list(dict_list, padel=np.nan):
    ''' https://stackoverflow.com/questions/40442014/pandas-valueerror-arrays-must-be-all-same-length '''
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list

def get_density(fname='POSCAR'):
    from ase.io import read

    atoms = read(fname)
    total_mass = np.sum(atoms.get_masses()) * 1.6605402E-27 # unit kg
    vol = atoms.get_volume() * 1e-30 # unit: m^3
    density = total_mass / vol # unit: kg/m^3
    density1 = density * 0.001 # unit: g/cm^3
    return density, density1

def time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def is_vasp_completed(fname='OUTCAR'):
    if not os.path.exists(fname):
        return False
    with open(fname, 'r') as f:
        f.seek(0, os.SEEK_END)
        f.seek(f.tell() - 13000, os.SEEK_SET)
        lines = f.readlines()
        return ' reached required accuracy - stopping structural energy minimisation\n' in lines

def scale_atoms(atoms, scale_factor=1.0):
    calculator = atoms.get_calculator()
    new_atoms  = atoms.copy()
    new_atoms.set_calculator(calculator)
    cell        = new_atoms.get_cell()
    frac_coords = new_atoms.get_scaled_positions()
    new_cell    = cell * scale_factor
    new_atoms.set_cell(new_cell)
    new_atoms.set_scaled_positions(frac_coords)
    return new_atoms

def get_ase_atom_from_formula_template(chemical_formula, v_per_atom=None,
                                       template='POSCAR',
                                       exclude_shuffle_elements=[]):
    # interpret formula
    # the template file should be a bulk structure
    atomic_fracions    = get_concentration_from_ase_formula(chemical_formula)
    elements           = [x for x in atomic_fracions]
    element_number     = [atomic_numbers[x] for x in elements]
    if isinstance(template, ase.atoms.Atoms):
        atoms = template
    elif isinstance(template, str):
        atoms              = read(template)
    total_atom         = len(atoms)
    num_atom_list      = np.array(list(atomic_fracions.values())) * total_atom
    num_atom_list      = np.around(num_atom_list, decimals=0)
    total_tmp          = np.sum(num_atom_list)
    deviation          = total_atom - total_tmp
    num_atom_list[np.random.randint(len(elements))] += deviation

    excluded_element_numbers = [atomic_numbers[x] for x in exclude_shuffle_elements]

    # shuffle atoms
    # if you do not want to shuffle some elements, put them in `exclude_shuffle_elements`
    # note that the unshuffled atoms are not controled by the template.
    ase_number = []
    shuffle_mask = []
    for i_index, i in enumerate(num_atom_list):
        for j in range(int(i)):
            ase_number.append(element_number[i_index])
            shuffle_mask.append(element_number[i_index] not in excluded_element_numbers)

    ase_number_need_shuffle = [x for i, x in enumerate(ase_number) if shuffle_mask[i]]
    np.random.shuffle(ase_number_need_shuffle)

    ase_number_no_shuffle = [x for i, x in enumerate(ase_number) if not shuffle_mask[i]]

    ase_number = []
    for m in shuffle_mask:
        if m:
            ase_number.append(ase_number_need_shuffle.pop(0))
        else:
            ase_number.append(ase_number_no_shuffle.pop(0))

    atoms.set_atomic_numbers(ase_number)

    # scale atoms
    if isinstance(v_per_atom, (float, int)):
        volume = atoms.cell.volume
        volume_per_atom = volume / len(atoms)
        volume_ratio = v_per_atom / volume_per_atom
        scale_factor = pow(volume_ratio, 1/3)
        atoms = scale_atoms(atoms, scale_factor)
    return atoms

def get_ase_atom_from_formula_template(file_name='POSCAR'):
    atoms = read(file_name)
    return atoms.get_chemical_formula()
