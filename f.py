import numpy as np
import math
import os
from scipy.interpolate import RegularGridInterpolator as RGI

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
    
# def get_dos(index_list):
#     orbital_dict = {'s': [], 'p': [], 'd':[]}
#     dos_up_all, dos_dn_all = [], []
#     for i in index_list:
#         assert os.path.exists('DOS'+str(i)), 'DOS'+str(i)+" not found."
#         dos_tmp = np.loadtxt('DOS'+str(i))
#         dos_up_all.append(np.sum(dos_tmp[:,orbital_dict['d'][0]], axis=1))
#         dos_dn_all.append(np.sum(dos_tmp[:,orbital_dict['d'][1]], axis=1))
#     energy = dos_tmp[:,0]
#     dos_up = np.sum(dos_up_all, axis=0)
#     dos_dn = np.sum(dos_dn_all, axis=0)
#     dos    = np.array([energy, dos_up, dos_dn])
#     return dos
    
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
'''.split()

def modify_INCAR(key='NSW', value='300', s=''):
    if not key in INCAR_TAG:
        print('Input key not avaliable, please check.')
        return 1

    new_incar, discover_code = [], False
    with open('INCAR', 'r') as f:
        for line in f:
            str_list = line.split()
            if len(str_list) == 0:
                new_incar.append('\n')
            elif str_list[0] == key:
                str_list[2] = value
                new_incar.append(f'  {str_list[0]} = {str_list[2]}\n')
                discover_code = True
            else:
                new_incar.append(line)

    if s:
        new_incar.append(f'\n{s}\n')

    if not discover_code:
        new_incar.append(f'  {key} = {value}\n')

    with open('INCAR', 'w') as f:
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
