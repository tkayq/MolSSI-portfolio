from scipy.integrate import dblquad, fixed_quad
from scipy.optimize import minimize
from scipy.spatial import distance
from scipy.linalg import eig
import numpy as np
import sys
import time
from multiprocessing import Pool, cpu_count
import os
import pickle
from numpy import linalg

# comment out when using plt_all_scripts.py # plotting file
from essqc import integrand_at_rtp


#np.set_printoptions(linewidth=np.inf)
#np.set_printoptions(precision=6)

######################################
############# PYSCF calc #############
######################################
def pyscf_result(z_array, all_nuc_xyz, basis_set, file_name=None):
  """
  Input:
    z_array = options['Z']
    all_nuc_xyz = options['nuclei'] in [N,3] array
    basis_set = options['basis_type']

  Output:
    s = AOxAO overlap matrix   
    pyscf_1e_energy = AOxAO 1 electron energy -- TODO - maybe include orth_transform step here?
    mocoeff = occupied mo coeffs ---> C matrix
  """
  from pyscf import gto, scf, lo
  nuc_dict = {
  1: 'H',
  2: 'He',
  3: 'Li',
  4: 'Be',
  5: 'B',
  6: 'C',
  7: 'N',
  8: 'O',
  9: 'F',
  10: 'Ne',
  11: 'Na',
  12: 'Mg',
  13: 'Al',
  14: 'Si',
  15: 'P',
  16: 'S',
  17: 'Cl',
  18: 'Ar',
  }

  geom = []
  for i, val in enumerate(z_array.reshape(-1)):
      atom_key = nuc_dict[int(val)]
      nuc_xyz = tuple(all_nuc_xyz[i])
      geom.append([atom_key, nuc_xyz])
  #print("geom", geom, flush=True) 

  print()
  print("From Pyscf with basis set: ", basis_set)
  mol = gto.Mole()
  mol.build(
  unit = 'Bohr',
  atom = geom,
  basis = basis_set,
  )
  print(mol.ao_labels())
  
  my_hf = scf.RHF(mol)
  e_mol = my_hf.kernel()
  
  # Overlap, kinetic, nuclear attraction
  s = mol.intor('int1e_ovlp')
  t = mol.intor('int1e_kin')
  v = mol.intor('int1e_nuc')
  pyscf_1e_energy = t + v

  mo_en = my_hf.mo_energy
  print("MO energies", mo_en)
  orb_coeff = my_hf.mo_coeff
  #my_hf.mulliken_pop()
  occ_orb_coeff = my_hf.mo_coeff[:,my_hf.mo_occ > 0.]
  virt_orb_coeff = my_hf.mo_coeff[:,my_hf.mo_occ == 0.]

  pm_loc_orb = lo.PM(mol, mo_coeff=occ_orb_coeff) #, pop_method='mulliken') # Pipek-Mezey
  pm_loc_orb.pop_method = 'mulliken'
  pm_loc_orb.init_guess = None
  #print(pm_loc_orb.kernel(occ_orb_coeff))

  #pm_loc_orb = lo.pipek.PipekMezey(mol, mo_coeff=occ_orb_coeff, pop_method='mulliken') # Pipek-Mezey
  #pm_pop = lo.pipek.atomic_pops(mol, mo_coeff=pm_loc_orb, method='mulliken')
#  molden.from_mo(mol, 'pm.molden', pm_loc_orb)
  fb_loc_orb = lo.Boys(mol, mo_coeff=occ_orb_coeff) #, init_guess=None)	 # Foster Boys
  er_loc_orb = lo.ER(mol, occ_orb_coeff)	 # Edmiston Ruedenberg
  chol_loc_orb = lo.cholesky_mos(occ_orb_coeff)	 # Cholesky

  #print("Overlap matrix\n", s, "\nKE\n", t, "\nNuclear Potential\n", v)
  #print("")
  #print("Full C matrix:\n", orb_coeff)
  #print("occupied MO:\n", occ_orb_coeff)
  ##print("virtual MO:\n", virt_orb_coeff)
  ##print("\nPipek-Mezey\n", PM-MO) #, "\npm AO population\n", pm_pop)
  #print("\nFoster-Boys\n", fb_loc_orb.kernel(occ_orb_coeff))
  #print("\nEdmiston Ruedenberg\n", er_loc_orb.kernel())
  #print("\nCholesky\n", chol_loc_orb)

  PM_loc = np.asarray(pm_loc_orb.kernel(occ_orb_coeff), order='C')
#  #print("\nPM atomic orb pop\n", pm_loc_orb.atomic_pops(mol=mol, mo_coeff=occ_orb_coeff))
  #print("\nPipek-Mezey\n", PM_loc) #np.array2string(PM_loc, separator=', ', formatter={'float_kind': lambda x: "%.6E" % x})) # , "\npm AO population\n", pm_pop)
  #PM_loc[np.abs(PM_loc) < 0.2] = 0.0
  #print("\nfiltered Pipek-Mezey\n", PM_loc) #np.array2string(PM_loc, separator=', ', formatter={'float_kind': lambda x: "%.6E" % x})) # , "\npm AO population\n", pm_pop)
  #print("returning vanilla occ_orb_coeff")
  #return s, pyscf_1e_energy, np.asarray(occ_orb_coeff, order='C')
  fock_mat = my_hf.get_fock()
  
  #pyscf_info = {
  #  'S': s,
  #  'pyscf_1e_energy': pyscf_1e_energy,
  #  'canonical_C': occ_orb_coeff,
  #  'fock_mat': fock_mat,
  #  #'PM_loc': PM_loc,
  #  'fb_loc_C': np.asarray(fb_loc_orb.kernel(occ_orb_coeff), order='C'),
  #  #'er_loc_orb': er_loc_orb,
  #  #'chol_loc_orb': chol_loc_orb,
  #}

  #if file_name != None:
  #  pickle.dump(pyscf_info, open(file_name+'_pyscf_dict.pkl', "wb"))

  #return s, pyscf_1e_energy, fock_mat, np.asarray(fb_loc_orb.kernel(occ_orb_coeff), order='C')
  print("returning Pipek Mezey localized orbitals")
  return s, pyscf_1e_energy, fock_mat, PM_loc #np.asarray(pm_loc_orb.kernel(occ_orb_coeff), order='C')

def pyscf_result_loc(z_array, all_nuc_xyz, basis_set, loc=True):
  """
  Input:
    z_array = options['Z']
    all_nuc_xyz = options['nuclei'] in [N,3] array
    basis_set = options['basis_type']
    loc = output C F S in localized basis 

  Output:
    s = AOxAO overlap matrix   
    pyscf_1e_energy = AOxAO 1 electron energy -- TODO - maybe include orth_transform step here?
    mocoeff = occupied mo coeffs ---> C matrix
  """
  from pyscf import gto, scf, lo
  nuc_dict = {
  1: 'H',
  2: 'He',
  3: 'Li',
  4: 'Be',
  5: 'B',
  6: 'C',
  7: 'N',
  8: 'O',
  9: 'F',
  10: 'Ne',
  11: 'Na',
  12: 'Mg',
  13: 'Al',
  14: 'Si',
  15: 'P',
  16: 'S',
  17: 'Cl',
  18: 'Ar',
  }

  geom = []
  for i, val in enumerate(z_array.reshape(-1)):
      atom_key = nuc_dict[int(val)]
      nuc_xyz = tuple(all_nuc_xyz[i])
      geom.append([atom_key, nuc_xyz])

  print()
  print("From Pyscf with basis set: ", basis_set)
  mol = gto.Mole()
  mol.build(
  unit = 'Bohr',
  atom = geom,
  basis = basis_set,
  )
  print(mol.ao_labels())
  
  my_hf = scf.RHF(mol)
  e_mol = my_hf.kernel()

  mo_en = my_hf.mo_energy
  print("MO energies", mo_en)

  orb_coeff = my_hf.mo_coeff
  
  # Overlap, kinetic, nuclear attraction, fock, mocoeff in canonical basis
  s = mol.intor('int1e_ovlp')             # AOxAO overlap matrix
  t = mol.intor('int1e_kin')
  v = mol.intor('int1e_nuc')
  pyscf_1e_energy = t + v

  f = my_hf.get_fock()                    # AOxAO fock matrix
  c = my_hf.mo_coeff[:,my_hf.mo_occ > 0.] # AO x MO matrix
  print("Canconical HF MOs before orthogonalization\n",np.array2string(c, separator=','),flush=True)

  if loc:
    print("Returning C in Pipek Mezey localized basis")

    ### Transform into localized Pipek Mezey basis ###
    pm_loc_orb = lo.PM(mol, mo_coeff=c)   # Pipek-Mezey
    pm_loc_orb.pop_method = 'mulliken'
    pm_loc_orb.init_guess = None

    c_loc = np.asarray(pm_loc_orb.kernel(c), order='C')
    #f_loc = c_loc.T @ f @ c_loc
    #s_loc = c_loc.T @ s @ c_loc
    #en_1e_loc = c_loc.T @ pyscf_1e_energy @ c_loc

    print("Pipek Mezey localized HF MOs before orthogonalization\n",np.array2string(c_loc, separator=','),flush=True)
    return s, pyscf_1e_energy, f, c_loc
    #return s_loc, en_1e_loc, f_loc, c_loc
  else:
    print("Returning S, 1e_E, F, C in canconical basis")
    c = np.asarray(c, order='C')

    return s, pyscf_1e_energy, f, c 

def run_cc(z_array, all_nuc_xyz, basis_set):
  # mf - scf.HF(mol).run()
  from pyscf import gto, scf, lo, cc
  nuc_dict = {
  1: 'H',
  2: 'He',
  3: 'Li',
  4: 'Be',
  5: 'B',
  6: 'C',
  7: 'N',
  8: 'O',
  9: 'F',
  10: 'Ne',
  11: 'Na',
  12: 'Mg',
  13: 'Al',
  14: 'Si',
  15: 'P',
  16: 'S',
  17: 'Cl',
  18: 'Ar',
  }

  geom = []
  for i, val in enumerate(z_array.reshape(-1)):
      atom_key = nuc_dict[int(val)]
      nuc_xyz = tuple(all_nuc_xyz[i])
      geom.append([atom_key, nuc_xyz])

  print()
  print("From Pyscf with basis set: ", basis_set)
  mol = gto.Mole()
  mol.build(
  unit = 'Bohr',
  atom = geom,
  basis = basis_set,
  )
  mf = scf.HF(mol).run()
  mycc = cc.CCSD(mf).run()
  ccsd_en = mycc.e_tot
  print("CCSD total energy", ccsd_en, flush=True)
  et = mycc.ccsd_t()
  ccsdt_en = ccsd_en + et 
  print("CCSD(T) total energy", ccsdt_en, flush=True)

###################################
############ FUNCTIONS ############
###################################

def r_from_xyz(xyz, origin=np.zeros((1,3))):
  """ input xyz must be [m,3] or [3,] 
  return matrix of radial distances [m,N], N=# of nuclei, m = # xyz
	if all_diff is true output the r between each xyz and origin input, must be same length """
  #all_diff = True if (xyz.shape == origin.shape) else False	# either get r's between each ao_nuc input, or one xyz rel to another
  xyz = np.reshape(xyz, [-1,3])         # [m,3]
  origin = np.reshape(origin, [-1,3])[:,:,np.newaxis]    # [N,3,1]

  diff_mat = np.transpose(np.sqrt(np.sum((origin - xyz.T)**2,axis=1)))   # (sum([N,3,1] - [3,m])_3 = [N,m]).T = [m,N]
  #print("diffmat in r_from: \n", diff_mat)
  #print("vs scipy diffmat in r_from: \n", distance.cdist(xyz.reshape(-1,3), origin.reshape(-1,3), 'euclidean'))
  return diff_mat   # [m,N]

  ##print(diff_mat)
  #if all_diff:
  #  #print("all_diff is True in r_from_xyz")
  #  return np.diagonal(diff_mat) #[:,np.newaxis]   # [m,1]
  #else:
  #  return diff_mat   # [m,N]

def spherical_to_cartesian(phi, theta, r, shift=np.zeros((1,3))):
  """ return (N,3) vector ---> [x, y, z] x N of cartesian coordinates """
  r = np.reshape(r, [-1,1])
  theta = np.reshape(theta, [-1,1])
  phi = np.reshape(phi, [-1,1])

  shift = np.reshape(shift, [-1,3])

  #print("spherical_to_cartesian --- r", r)
  #print("theta", theta)
  #print("phi", phi)

  x = r * np.sin(theta) * np.cos(phi)
  y = r * np.sin(theta) * np.sin(phi)
  z = r * np.cos(theta)

  #print("x", x.shape, x)
  #print("y", y.shape, y)
  #print("z", z.shape, z)
  
  return np.concatenate((x, y, z), axis=1) + shift  # [m,3]

def get_mol_xyz(dict_key):
  nuc_lines = dict_key.split('\n')
  nuc_lines = [line.strip() for line in nuc_lines if line.strip()]
  all_nuc_xyz = np.array([list(map(float, line.split())) for line in nuc_lines])
  return all_nuc_xyz

####################################
############ BASIS INFO ############
####################################

def set_basis_orb_info(cusp_input, basis, basis_orb_type, basis_centers, Z_array):
  """
  Determine basis specific parmeters for each AO:
  rc_matrix --- nuc x AO matrix of cusp radii for each AO (nuc x AOs)
  orth_orb_array --- array with non-zero values indicate that orb will be orthogonalized against it's core, value = (core index + 1)
  """
  num_AOs = int(len(basis_orb_type))
  #print("num of AOs ",num_AOs)
  rc_matrix = np.zeros((len(Z_array), num_AOs))
  rc_matrixv2 = np.zeros((len(Z_array), num_AOs))
  orth_orb_array = np.zeros(num_AOs, dtype=int)

  add_count = count = 0 
  #print(basis, i, count, Z, flush=True)

  nonH_nuc_ind = np.where(Z_array > 1.0)[0] # nuc ind of Z>1
  #print("nonH_nuc_ind", nonH_nuc_ind)

  for i, Z in enumerate(Z_array.reshape(-1)):   # loop through nuclei - and each orb for columns
    on_center_ind = np.where(basis_centers == i)[0]
    #p_cen_ind = np.where(basis_centers[on_cen_ind] == i)[0]
    print(i, Z, "on center index", on_center_ind)

    if basis == str('sto-3g') or basis == str('STO-3G'):
      #print('basis = STO-3G')
      if Z == 1.0:                # hydrogen
        add_count = count + 1
        default_1s_cusp = 0.2 # 0.1
        def_1s_cusp_offCenter = 0.1

        rc_matrixv2[i, :] = default_1s_cusp
        for ind in nonH_nuc_ind :
          #print("ind in nonH_nuc_ind", ind)
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          #print("corresponding bf_ind", nonH_bf_ind)
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            #print("bf type : ", bf_ind_type,flush=True)
            if bf_ind_type in {0,1,}:  # s-orbs
              #print("s-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.1
            elif bf_ind_type in {2,3,4}: # p-orbs
              #print("p-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              #print("d-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("STO-3G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
      elif Z in {3., 4., 5., 6., 7., 8., 9., 10.}:              # carbon
        add_count = count + 5
        default_1s_cusp = 0.075   # TODO optimize cusp value
        def_1s_cusp_offCenter = 0.0035

        rc_matrixv2[i, :] = default_1s_cusp
        orth_orb_array[count+1] = count+1      # orthogonalize the 2s orb against 1s core
        for ind in nonH_nuc_ind :
          #print("ind in nonH_nuc_ind", ind)
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          #print("corresponding bf_ind", nonH_bf_ind)
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            #print("bf type : ", bf_ind_type,flush=True)
            if bf_ind_type in {0,1,}:  # s-orbs
              #print("s-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.1
            elif bf_ind_type in {2,3,4}: # p-orbs
              #print("p-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              #print("d-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("STO-3G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
      else:
        raise RuntimeError("STO-3G: Z", Z, "not identified in cusp_orbitals.set_basis_orb_info")

    elif basis == str('6-31G') or basis == str('6-31g'):
      #print('basis = 6-31G, count = ', count, flush=True)
      if Z == 1.0:                # hydrogen    1s(3G), 2s_orth(1G)
        add_count = count + 2
        default_s_cusp = 0.15
        default_p_cusp = 0.075
        core_1s_cusp = 0.20
        
        rc_matrixv2[i, :] = 0.2
        for ind in nonH_nuc_ind :
          #print("ind in nonH_nuc_ind", ind)
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          #print("corresponding bf_ind", nonH_bf_ind)
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            #print("bf type : ", bf_ind_type,flush=True)
            if bf_ind_type in {0,1,}:  # s-orbs
              #print("s-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.1
            elif bf_ind_type in {2,3,4}: # p-orbs
              #print("p-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              #print("d-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("6-31G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
       
        orth_orb_array[count+1] = count+1      # orthogonalize 1s (1G) against 1s (3G)
      elif Z > 1.:              # carbon
        add_count = count + 9
        default_s_cusp = 0.2
        default_p_cusp = 0.0
        core_1s_cusp = 0.1

        bf_centered_node = np.where(basis_orb_type[on_center_ind].flatten() > 1)[0] + on_center_ind[0]    # where oncenter p/d orbs and shift to where in full bf list
        #print("Z",Z, "bf_centered_node indices > 1 orb type", bf_centered_node, flush=True)
        rc_matrixv2[i, :] = 0.2
        for ind in nonH_nuc_ind :
          #print("ind in nonH_nuc_ind", ind)
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          #print("corresponding bf_ind", nonH_bf_ind)
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            #print("bf type : ", bf_ind_type)
            if bf_ind_type in {0,1,}:  # s-orbs
              #print("s-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.2
            elif bf_ind_type in {2,3,4}: # p-orbs
              #print("p-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              #print("d-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("6-31G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
        rc_matrixv2[i, bf_centered_node] = 0.0
       
        orth_orb_array[count+1:count+3] = count+1      # orthogonalize the s orbs against 1s core
      else:
        raise RuntimeError("6-31G: Z", Z, "not identified in cusp_orbitals.set_basis_orb_info")
    elif basis == str('6-31G*') or basis == str('6-31g*'):
      #print('basis = 6-31G, count = ', count, flush=True)

      if Z == 1.0:                # hydrogen
        add_count = count + 2
        default_s_cusp = 0.2
        default_p_cusp = 0.075
        core_1s_cusp = 0.20
        
        rc_matrixv2[i, :] = 0.2
        for ind in nonH_nuc_ind:
          #print("ind in nonH_nuc_ind", ind)
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          #print("corresponding bf_ind", nonH_bf_ind)
          #print("corresponding orb types of each bf_ind", basis_orb_type[nonH_bf_ind])
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            #print("bf type : ", bf_ind_type)
            if bf_ind_type in {0,1,}:  # s-orbs
              #print("s-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.1
            elif bf_ind_type in {2,3,4}: # p-orbs
              #print("p-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              #print("d-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("6-31G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")

        orth_orb_array[count+1] = count+1      # orthogonalize 1s (1G) against 1s (3G)
      elif Z in np.arange(2., 11.):              # carbon
        add_count = count + 14
        default_s_cusp = 0.2
        default_p_cusp = 0.0
        default_d_cusp = 0.0
        core_1s_cusp = 0.1

        bf_centered_node = np.where(basis_orb_type[on_center_ind].flatten() > 1)[0] + on_center_ind[0]    # where oncenter p/d orbs and shift to where in full bf list
        #print("Z",Z, "bf_centered_node indices > 1 orb type", bf_centered_node, flush=True)
        rc_matrixv2[i, :] = 0.2
        for ind in nonH_nuc_ind:
          #print("ind in nonH_nuc_ind", ind)
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          #print("corresponding bf_ind", nonH_bf_ind)
          #print("corresponding orb types of each bf_ind", basis_orb_type[nonH_bf_ind])
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            #print("bf type : ", bf_ind_type)
            if bf_ind_type in {0,1,}:  # s-orbs
              #print("s-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.2
            elif bf_ind_type in {2,3,4}: # p-orbs
              #print("p-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              #print("d-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("6-31G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
        rc_matrixv2[i, bf_centered_node] = 0.0

        orth_orb_array[count+1:count+3] = count+1      # orthogonalize the s orbs against 1s core
      elif Z in np.arange(11., 19.): # third row 
        add_count = count + 18
        default_s_cusp = 0.2
        default_p_cusp = 0.0
        default_d_cusp = 0.0
        core_1s_cusp = 0.1

        bf_centered_node = np.where(basis_orb_type[on_center_ind].flatten() > 1)[0] + on_center_ind[0]    # where oncenter p/d orbs and shift to where in full bf list
        #print("Z",Z, "bf_centered_node indices > 1 orb type", bf_centered_node, flush=True)
        rc_matrixv2[i, :] = 0.05
        for ind in nonH_nuc_ind:
          #print("ind in nonH_nuc_ind", ind)
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          #print("corresponding bf_ind", nonH_bf_ind)
          #print("corresponding orb types of each bf_ind", basis_orb_type[nonH_bf_ind])
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            #print("bf type : ", bf_ind_type)
            if bf_ind_type in {0,1,}:  # s-orbs
              #print("s-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.05
            elif bf_ind_type in {2,3,4}: # p-orbs
              #print("p-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              #print("d-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("6-31G: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
        rc_matrixv2[i, bf_centered_node] = 0.0

        orth_orb_array[count+1:count+4] = count+1      # orthogonalize the s orbs against 1s core
      else:
        raise RuntimeError("6-31G*: Z", Z, "not identified in cusp_orbitals.set_basis_orb_info")
    elif basis == str('cc-pcvdz'):
      #print('basis = 6-31G, count = ', count, flush=True)

      if Z == 1.0:                # hydrogen
        add_count = count + 3
        default_s_cusp = 0.2
        default_p_cusp = 0.075
        core_1s_cusp = 0.20
        
        rc_matrixv2[i, :] = 0.2
        for ind in nonH_nuc_ind:
          #print("ind in nonH_nuc_ind", ind)
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          #print("corresponding bf_ind", nonH_bf_ind)
          #print("corresponding orb types of each bf_ind", basis_orb_type[nonH_bf_ind])
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            #print("bf type : ", bf_ind_type)
            if bf_ind_type in {0,1,}:  # s-orbs
              #print("s-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.1
            elif bf_ind_type in {2,3,4}: # p-orbs
              #print("p-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              #print("d-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("cc-pcvdz: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")

        orth_orb_array[count+1] = count+1      # orthogonalize 1s (1G) against 1s (3G)
      elif Z in np.arange(2., 11.):              # carbon
        add_count = count + 18
        default_s_cusp = 0.2
        default_p_cusp = 0.0
        default_d_cusp = 0.0
        core_1s_cusp = 0.1

        bf_centered_node = np.where(basis_orb_type[on_center_ind].flatten() > 1)[0] + on_center_ind[0]    # where oncenter p/d orbs and shift to where in full bf list
        #print("Z",Z, "bf_centered_node indices > 1 orb type", bf_centered_node, flush=True)
        rc_matrixv2[i, :] = 0.2
        for ind in nonH_nuc_ind:
          #print("ind in nonH_nuc_ind", ind)
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          #print("corresponding bf_ind", nonH_bf_ind)
          #print("corresponding orb types of each bf_ind", basis_orb_type[nonH_bf_ind])
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            #print("bf type : ", bf_ind_type)
            if bf_ind_type in {0,1,}:  # s-orbs
              #print("s-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.2
            elif bf_ind_type in {2,3,4}: # p-orbs
              #print("p-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              #print("d-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("cc-pcvdz: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
        rc_matrixv2[i, bf_centered_node] = 0.0

        orth_orb_array[count+1:count+3] = count+1      # orthogonalize the s orbs against 1s core
      elif Z in np.arange(11., 19.): # third row 
        add_count = count + 18
        default_s_cusp = 0.2
        default_p_cusp = 0.0
        default_d_cusp = 0.0
        core_1s_cusp = 0.1

        bf_centered_node = np.where(basis_orb_type[on_center_ind].flatten() > 1)[0] + on_center_ind[0]    # where oncenter p/d orbs and shift to where in full bf list
        #print("Z",Z, "bf_centered_node indices > 1 orb type", bf_centered_node, flush=True)
        rc_matrixv2[i, :] = 0.05
        for ind in nonH_nuc_ind:
          #print("ind in nonH_nuc_ind", ind)
          nonH_bf_ind = np.where(basis_centers == ind)[0]
          #print("corresponding bf_ind", nonH_bf_ind)
          #print("corresponding orb types of each bf_ind", basis_orb_type[nonH_bf_ind])
          for bf_count, bf_ind_type in enumerate(basis_orb_type[nonH_bf_ind].flatten()):
            #print("bf type : ", bf_ind_type)
            if bf_ind_type in {0,1,}:  # s-orbs
              #print("s-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.05
            elif bf_ind_type in {2,3,4}: # p-orbs
              #print("p-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            elif bf_ind_type in {5,6,7,8,9}: # d-orbs
              #print("d-orb")
              rc_matrixv2[i, nonH_bf_ind[bf_count]] = 0.075
            else:
              raise RuntimeError("cc-pcvdz: bf_ind type", bf_ind_type, "not identified in cusp_orbitals.set_basis_orb_info")
        rc_matrixv2[i, bf_centered_node] = 0.0

        orth_orb_array[count+1:count+4] = count+1      # orthogonalize the s orbs against 1s core
       # hard-code oth_orb_array because I'm tired
      else:
        raise RuntimeError("cc-pcvdz: Z", Z, "not identified in cusp_orbitals.set_basis_orb_info")
      orth_orb_array = np.array([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 24, 0, 0, 0, 0, 29, 0, 0, 0, 0, 34, 0, 0, 0], dtype=int) 
    else:		# no defined basis, all radii default to zero
      break
    count = add_count
  
  if cusp_input != None:		# default all radii to input cusp-radius
    print("replace with global_cusp: ", cusp_input)
    rc_matrix = np.zeros((len(Z_array), num_AOs)) + cusp_input 
  #print("in set info orth_orb_array (shifted): ", orth_orb_array, flush=True) 
  return rc_matrixv2, orth_orb_array


def orth_transform(orth, proj, nnuc, basis_centers, num_bf):
  """
  change of basis matrix for orthogonalized orbitals in cusping scheme
  return matrix to change basis to orth orbs 
  """
  print("in orth_transform - num_nuc", nnuc, " num_bf", num_bf, " orth_orb_array", orth, flush=True)
  B = np.identity(num_bf)    # AOs x new AOs

  for i in range(0, nnuc):    # loop through nuclei
    #print(i, flush=True)
    on_nuc_ind = np.where(basis_centers == i)[0]  # indicies of current nuc centered orb
    orth_nuc_ind = np.where(orth[on_nuc_ind] != 0)[0]  # indices of orbs to orthogonalize against core orbital (in all ao space)
    
    for j in [x + on_nuc_ind[0] for x in orth_nuc_ind]:    # loop through indices orthogonalized AOs centered on given nucleus
      core_ind = int(orth[j] - 1)	# in full AO list
      B[core_ind, j] = - proj[j, core_ind]

  #print("\nB matrix\n", B)
  return B 


def get_proj_mat(basis_centers, orth_orb_array, all_nuc_xyz, z_array, basis_orb_type, basis_exp, basis_coeff):
  """
  AO x AO matrix to calculate overlap between core AO (rows) and AO to orthogonalize (column)
  """
  proj_ao_mat = np.zeros((len(basis_centers), len(basis_centers)))

  for nuc_ind, Z in enumerate(z_array):
    nuc_xyz = np.reshape(all_nuc_xyz[nuc_ind], [1,3])

    for ao_ind, ao_type in enumerate(basis_orb_type):
      basis_center_ind = basis_centers[ao_ind]                   # index of nucleus this AO is centered on
      basis_center_xyz = np.reshape(all_nuc_xyz[basis_center_ind], [1, 3])  # coords of nucleus this AO is centered on
      core_ind = orth_orb_array[ao_ind]
      #print("core_ind, number if orthogonalizing else False: ", core_ind, flush=True)
      alpha = np.reshape(basis_exp[ao_ind], [-1,1])
      d = np.reshape(basis_coeff[ao_ind], [-1,1])

      if core_ind > 0:
        core_ind = core_ind - 1
        #print("orthogonalize this AO against core with AO index: ", core_ind)
        alpha_core = np.reshape(basis_exp[core_ind], [-1,1])
        d_core = np.reshape(basis_coeff[core_ind], [-1,1])
        #print("alpha core and d core: ", alpha_core, d_core)

        if proj_ao_mat[ao_ind, core_ind] == 0:
          proj = get_proj_2_1(basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d)
          #print("calculate projection:  ", proj, flush=True)
          proj_ao_mat[ao_ind, core_ind] = proj

  return proj_ao_mat


def get_proj_2_1(basis_center, integration_center, alpha_1, d_1, alpha_2, d_2):
    """
      orthogonalize AO2 against AO1: 	 proj = < AO2 | AO1 > 
       				        	_____________

       				        	< AO1 | AO1 > 
    """
    proj_2_1 = 0.0
    norm_1 = 0.0

    # because trailing zeros to match man nG length, find max n between basis func 1 and 2
    basis_1_len = np.trim_zeros(alpha_1.reshape(-1)).size
    basis_2_len = np.trim_zeros(alpha_2.reshape(-1)).size
    nG = max([basis_1_len, basis_2_len]) 
    for i in range(nG):
      for j in range(nG):
          A01_g_overlap = ( 2.0 * alpha_1[i] / np.pi )**0.75 * ( 2.0 * alpha_1[j] / np.pi )**0.75 * d_1[i] * d_1[j] * ( np.pi / (alpha_1[i] + alpha_1[j]) )**1.5     # <1s|1s>
          A01_A02_g_overlap = ( 2.0 * alpha_1[i] / np.pi )**0.75 * ( 2.0 * alpha_2[j] / np.pi )**0.75 * d_1[i] * d_2[j] * ( np.pi / (alpha_1[i] + alpha_2[j]) )**1.5  # <1s|2s>
          proj_2_1 += A01_A02_g_overlap
          norm_1 += A01_g_overlap
    #print("analytic proj_2_1", proj_2_1)
    #print("analytic norm", norm_1)

    proj_2_1 = proj_2_1 / norm_1
    return proj_2_1

###########################################
############ GAUSSIAN ORBITALS ############
###########################################
def gaussian_info_and_eval(nuc_ind, ao_ind, stretched, options, get_bf=None): #, add_to_out=None):
  """ orbital info and evaluation (if get_bf != None, should be string indicating where to eval the bf at or array with value to eval at) 
      returns shifted oth_orb_array"""
  ij = (nuc_ind, ao_ind)
  #print(ij,  "ij in gaussian_info_and_eval", flush=True)

  nuc_ind = np.atleast_1d(nuc_ind)
  ao_ind = np.atleast_1d(ao_ind)

  if stretched == False:
    all_nuc_xyz = get_mol_xyz(options['nuclei'])
    #print("all_nuc_xyz", all_nuc_xyz, all_nuc_xyz.shape)

    py_s = (options["pyscf_S"].diagonal())[ao_ind]
    #print("py_s", py_s, py_s.shape)

    py_en = 0 #(options["pyscf_1e_energy"].diagonal())[ao_ind]
    #print("py_en", py_en, py_en.shape)

  else:
    all_nuc_xyz = get_mol_xyz(options['nuclei_s'])
    #print("all_nuc_xyz", all_nuc_xyz, all_nuc_xyz.shape)

    py_s = (options["pyscf_S_s"].diagonal())[ao_ind]
    #print("py_s", py_s, py_s.shape)

    py_en = 0 #(options["pyscf_1e_energy_s"].diagonal())[ao_ind]
    #print("py_en", py_en, py_en.shape)

  nuc_xyz = (all_nuc_xyz[nuc_ind]).reshape(-1,3)
  #print("nuc_xyz", nuc_xyz, nuc_xyz.shape)

  ao_type = (options['basis_orb_type'][ao_ind]).astype(int).reshape(-1)
  #print("ao_type", ao_type, ao_type.shape)

  basis_center_ind = (options["basis_centers"][ao_ind]).astype(int).reshape(-1)
  #print("basis_center_ind", basis_center_ind, basis_center_ind.shape)

  on_center = (basis_center_ind == nuc_ind)
  #print("on_center", on_center, on_center.shape)
  
  basis_center_xyz = (all_nuc_xyz[basis_center_ind]).reshape(-1,3)
  #print("basis_center_xyz", basis_center_xyz, basis_center_xyz.shape)

  alpha = (options["basis_exp"][ao_ind]).reshape(-1, options['ng'])
  #print("alpha", alpha, alpha.shape)

  d = (options["basis_coeff"][ao_ind]).reshape(-1, options['ng'])
  #print("d", d, d.shape)

  r_cusp = options['cusp_radii_mat'][nuc_ind,ao_ind]
  #print("r_cusp", r_cusp, r_cusp.shape)

  orth_orb_shifted = np.atleast_1d(options['orth_orb_array'][ao_ind]).astype(int)
  orth_orb_bool = np.atleast_1d(orth_orb_shifted != 0)
  #print("orth_orb_shifted", orth_orb_shifted, orth_orb_shifted.shape, orth_orb_shifted.dtype)
  #print("orth_orb_bool", orth_orb_bool, orth_orb_bool.shape,  orth_orb_bool.dtype)

  if True in orth_orb_bool:
    core_true = np.where(orth_orb_bool == True)[0]
    #print("core_true", core_true,flush=True)
    core_ind = orth_orb_shifted[core_true] - 1
    #print("core_ind", core_ind, core_ind.shape)

    alpha_core = (options["basis_exp"][core_ind]).reshape(-1, options['ng'])
    #print("alpha_core", alpha_core, alpha_core.shape)

    d_core = (options["basis_coeff"][core_ind]).reshape(-1, options['ng'])
    #print("d_core", d_core, d_core.shape)
    
    proj = (options["proj_mat"][ao_ind[core_true],core_ind]).reshape(-1)
    #print("proj", proj, proj.shape)
  else:
    core_ind = alpha_core = d_core = proj = 0.0

  if get_bf != None:
    xyz_eval = np.zeros((1,3))
    if type(get_bf) == type(np.array([])):
      get_max = False
      xyz_eval = get_bf 
    elif get_bf == 'nuc':    # evaluate orbs over nuclei input
      get_max = False
      xyz_eval = nuc_xyz
    elif get_bf == 'max':    # find max of orbs
      get_max = True 
      xyz_eval = basis_center_xyz
    else:
      raise RuntimeError('get_bf input in gaussian_r_val is not valid')
    # returns bf value over indicated nuclei
    #xyz_eval = basis_center_xyz if get_max == True else nuc_xyz
    bf_vals = gaussian_r_val(xyz_eval, ao_type, alpha_core, d_core, basis_center_xyz, nuc_xyz, alpha, d, on_center, orth_orb_shifted, proj, get_max)
    return ij, all_nuc_xyz, nuc_xyz, ao_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj, py_s, py_en, bf_vals
  else:
    return ij, all_nuc_xyz, nuc_xyz, ao_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj, py_s, py_en


def gaussian_r_val(eval_xyz, orb_type, alpha_core, d_core, basis_center_xyz, nuc_xyz, alpha, d, on_center, orth_orb_shifted, proj_2_1, get_max=False): # evaluate radial portion of gaussian orbital at some r from xyz_eval to basis_center_xyz 
  """ eval gaussian orb at r: from  eval_xyz (relative to basis_center_xyz) [m,3], all else either length 1 or m
    input shifted orth_orb array """
  #np.set_printoptions(precision=12)
  # HERE TKQ
 # if eval_xyz.shape == basis_center_xyz.shape:  # eval at 1 r

 # elif (len(eval_xyz) == len(orb_type) == len(basis_center_xyz) == len(nuc_xyz) == len(alpha) == len(d) == len(on_center) == 1):

  #print("\t eval_xyz to basis_center_xyz", eval_xyz, "\n", basis_center_xyz, flush=True)
  r_mat = r_from_xyz(eval_xyz, basis_center_xyz)      # xyz_Bmol - xyz_Amol --> r_BA, [m,m] 
  r = np.diagonal(r_mat) if r_mat.shape[1] > 1 else r_mat
  ret_val = np.zeros_like(r)
  #r = np.asarray(r, dtype=float).reshape(-1,1)    # radial dist from basis_center_xyz to point of eval 
  theta = phi = 0.0
  total_orth_orb = 0 # index for orth elements
  #print("\t r from eval_xyz to basis_center_xyz", eval_xyz.shape, basis_center_xyz.shape, flush=True)
  #print("r in gaussian_r_val", r.flatten(), r.shape, flush=True)

  # need to adapt incase only one r with same params for everything else, should be able to pipe in array of r

  for ind, r_val in enumerate(r):
    nuc_xyz_now = nuc_xyz[ind] if len(nuc_xyz) > 1 else nuc_xyz

    #print()
    #print()
    #print(ind, " r val in gaussian_r_val", r_val, flush=True) 
    #print("\ton center", on_center[ind], flush=True)
    #print("\torb_type", orb_type[ind],flush=True)
    #print("\torth_orb --- should be shifted", orth_orb_shifted[ind],flush=True)
    #print("\talpha", alpha[ind],flush=True)
    #print("\td", d[ind],flush=True)
    #print("\tbasis_center_xyz",basis_center_xyz[ind],flush=True)
    #print("\tintegration center", nuc_xyz_now,flush=True)

    # s-orbital
    if orb_type[ind] < 2:

      # orthogonalized valence s orbital
      if orth_orb_shifted[ind] != 0:
        orth_against = orth_orb_shifted[ind] - 1
        #print("\t\torthogonalize!")
        #print("\t\torth_orb against",orth_against, total_orth_orb, flush=True)
        #print("\t\talpha_core",alpha_core[total_orth_orb],flush=True)
        #print("\t\td_core",d_core[total_orth_orb],flush=True)
        #print("\t\tproj",proj_2_1[total_orth_orb],flush=True)

        ret_val[ind] = STOnG_orth_s_eval(r_val, alpha_core[total_orth_orb], d_core[total_orth_orb], alpha[ind], d[ind], proj_2_1[total_orth_orb])[0]
        #print("\t\t\torthogonalized s orb gaussian_r_val (ind of orth):", total_orth_orb, "(", ret_val[ind], ")")
        total_orth_orb += 1

      # non-orthogonalized s-orb
      else: 
        ret_val[ind] = STOnG_s_eval(r_val, alpha[ind], d[ind])[0]
        #print("\t\t\ts orb gaussian_r_val:", ret_val[ind])

    # p-orbital
    elif orb_type[ind] in {2, 3, 4}:
      if get_max == True:   # opt to find max val
          func = lambda r_max: - np.abs(STOnG_p_eval(basis_center_xyz[ind], basis_center_xyz[ind], alpha[ind], d[ind], orb_type[ind]-2, r_max))
          opt_val = minimize(func, x0=0.1, method='BFGS')
          ret_val[ind] = opt_val.fun
          #print("p-orb, r and max val", opt_val.x, ret_val, flush=True)
      
      elif on_center[ind] == False: # code taken from plot_orb.py plot_thru_nuc()
          #delta = nuc_xyz_now - basis_center_xyz[ind]
          #print("Should these be equiv?? r_val (from xyz_eval to basis_center_xyz)", r_val, "delta", delta, "r from delta btw nuc_xyz and basis_center_xyz", np.sqrt(np.sum(delta**2)), flush=True)
          #r_xyz = basis_center_xyz[ind] + r_val * delta
          ret_val[ind] = STOnG_p_eval(eval_xyz[ind], basis_center_xyz[ind], alpha[ind], d[ind], orb_type[ind]-2)
          #print("p orb_eval AFTER2 no r--- gaussian_r_val:", ret_val)

      # centered p-orb
      else:
        #print("returning nan for atom-centered p-orb!!!")
        ret_val[ind] = np.nan
        #raise RuntimeError("calculating gaussian_r_val() of atom centered p orbital to cusp, no p-cusps on p orb built in the code")

    # d-orbital
    elif orb_type[ind] in {5, 6, 7, 8, 9}:
      if get_max == True:   # opt to find max val
          func = lambda r_max: - np.abs(STOnG_d_eval(basis_center_xyz[ind], basis_center_xyz[ind], alpha[ind], d[ind], orb_type[ind], r_max))
          opt_val = minimize(func, x0=0.1, method='BFGS')
          ret_val[ind] = opt_val.fun
          #print("p-orb, r and max val", opt_val.x, ret_val, flush=True)
      
      elif on_center[ind] == False: # code taken from plot_orb.py plot_thru_nuc()
          #delta = nuc_xyz_now - basis_center_xyz[ind]
          #print("Should these be equiv?? r_val (from xyz_eval to basis_center_xyz)", r_val, "delta", delta, "r from delta btw nuc_xyz and basis_center_xyz", np.sqrt(np.sum(delta**2)), flush=True)
          #r_xyz = basis_center_xyz[ind] + r_val * delta
          ret_val[ind] = STOnG_d_eval(eval_xyz[ind], basis_center_xyz[ind], alpha[ind], d[ind], orb_type[ind])
          #print("p orb_eval AFTER2 no r--- gaussian_r_val:", ret_val)

      # centered d-orb
      else:
        #print("returning nan for atom-centered p-orb!!!")
        ret_val[ind] = np.nan
        #raise RuntimeError("calculating gaussian_r_val() of atom centered p orbital to cusp, no p-cusps on p orb built in the code")
    else:
      raise RuntimeError("orbital category not classified in gaussian_r_val() --- orb_type = ", orb_type[ind])

  return ret_val #np.abs(ret_val)

#def gaussian_r_val(eval_xyz, orb_type, alpha_core, d_core, basis_center_xyz, nuc_xyz, alpha, d, on_center, orth_orb_shifted, proj_2_1, get_max=False): # evaluate radial portion of gaussian orbital at some r from xyz_eval to basis_center_xyz 
def gaussian_and_cusp_r_val(input_info, cusp_info=None, lc_cusp_info=None, get_max=False): # evaluate radial portion of gaussian orbital at some r from xyz_eval to basis_center_xyz 
  """ eval gaussian orb at r: from  eval_xyz (relative to basis_center_xyz) [m,3], all else either length 1 or m
    input shifted orth_orb array 
    input list [[[set of xyz vals], corresponding set of params], [etc...]] loops through the params 
      add cusp_info = [[a0, r_cusp, zeta, dist]] if want return gaus and slater-cusp
      also add lc_cusp_info = [[cusp_coeffs, order_n_list]] if wan linear combo slater-cusp vals as well
      else, just gauss vals """

  if cusp_info != None:
    if len(input_info) != len(cusp_info):
      raise Exception("length of input_info and cusp_info must be equiv if cusp_info != None in gaussian_and_cusp_r_val()")
    if lc_cusp_info != None:
      if len(input_info) != len(lc_cusp_info):
        raise Exception("length of input_info and lc_cusp_info must be equiv if lc_cusp_info != None in gaussian_and_cusp_r_val()")
  else:
    print("DONT CUSP", flush=True)
    if lc_cusp_info != None:
      raise Exception("lc_cusp_info != None, then cusp_info must also be provided ")


  ret_list = []
  ret_cusp_list = []
  ret_slater_list = []
  ret_lc_slater_cusp_list = []

  for i, info in enumerate(input_info): # single set of params to eval one or many r vals
    r, eval_xyz, orb_type, alpha_core, d_core, basis_center_xyz, nuc_xyz, alpha, d, on_center, orth_orb_shifted, proj_2_1 = info

    #print("\n\t in gaussian_r_eval2 eval_xyz to basis_center_xyz", eval_xyz.shape, basis_center_xyz, flush=True)
    #r_mat = r_from_xyz(eval_xyz, basis_center_xyz[0])      # xyz_Bmol - xyz_Amol --> r_BA, [m,m] 
    #r = np.diagonal(r_mat) if r_mat.shape[1] > 1 else r_mat
    ret_val = np.zeros_like(r) # initialize all possible returns
    #ret_val = ret_cusp_val = ret_slater_val = ret_lc_slater_cusp_val = np.zeros_like(r) # initialize all possible returns
    
    print("\tr from eval_xyz to basis_center_xyz", eval_xyz.shape, basis_center_xyz.shape, flush=True)
    print("\tr in gaussian_r_val --- should match r_to_plot", r[0], r[-1], flush=True)
    print("\tr in gaussian_r_val --- should match r_to_plot", r.shape, flush=True)
    print("\ton center", on_center[0], flush=True)
    print("\torb_type", orb_type[0],flush=True)
    print("\torth_orb --- should be shifted", orth_orb_shifted[0],flush=True)
    print("\talpha", alpha[0],flush=True)
    print("\td", d[0],flush=True)
    print("\tbasis_center_xyz",basis_center_xyz[0],flush=True)
    print("\tintegration center", nuc_xyz[0],flush=True)

    ### EVALUATE GAUSSIAN ###

    # s-orbital
    if orb_type[0] < 2:

      # orthogonalized valence s orbital
      if orth_orb_shifted[0] != 0:
        #orth_against = orth_orb_shifted[0] - 1
        #print("\t\torthogonalize!")
        #print("\t\torth_orb against",orth_against, flush=True)
        #print("\t\talpha_core",alpha_core[0],flush=True)
        #print("\t\td_core",d_core[0],flush=True)
        #print("\t\tproj",proj_2_1[0],flush=True)

        ret_val = STOnG_orth_s_eval(r, alpha_core[0], d_core[0], alpha[0], d[0], proj_2_1[0])
        print("\t\t\torthogonalized s orb gaussian_r_val (ind of orth):", "(", ret_val.shape, ")")

      # non-orthogonalized s-orb
      else: 
        ret_val = STOnG_s_eval(r, alpha[0], d[0])
        print("\t\t\ts orb gaussian_r_val2", ret_val.shape)

    # p-orbital
    elif orb_type[0] in {2, 3, 4}:
      if get_max == True:   # opt to find max val
          func = lambda r_max: - np.abs(STOnG_p_eval(basis_center_xyz[0], basis_center_xyz[0], alpha[0], d[0], orb_type[0]-2, r_max[0]))
          opt_val = minimize(func, x0=0.1, method='BFGS')
          ret_val = opt_val.fun
          print("p-orb, r and max val", opt_val.x, ret_val.shape, flush=True)
      
      elif on_center[0] == False: # code taken from plot_orb.py plot_thru_nuc()
          #delta = nuc_xyz_now - basis_center_xyz
          #print("Should these be equiv?? r_val (from xyz_eval to basis_center_xyz)", r_val, "delta", delta, "r from delta btw nuc_xyz and basis_center_xyz", np.sqrt(np.sum(delta**2)), flush=True)
          #r_xyz = basis_center_xyz + r_val * delta
          ret_val = STOnG_p_eval(eval_xyz, basis_center_xyz[0], alpha[0], d[0], orb_type[0]-2)
          print("p orb_eval AFTER2 no r--- gaussian_r_val:", ret_val.shape, flush=True)

      # centered p-orb
      else:
        #print("returning nan for atom-centered p-orb!!!")
        ret_val = np.nan
        #raise RuntimeError("calculating gaussian_r_val() of atom centered p orbital to cusp, no p-cusps on p orb built in the code")

    # d-orbital
    elif orb_type[0] in {5, 6, 7, 8, 9}:
      if get_max == True:   # opt to find max val
          func = lambda r_max: - np.abs(STOnG_d_eval(basis_center_xyz[0], basis_center_xyz[0], alpha[0], d[0], orb_type[0], r_max[0]))
          opt_val = minimize(func, x0=0.1, method='BFGS')
          ret_val = opt_val.fun
          print("d-orb, r and max val", opt_val.x, ret_val.shape, flush=True)
      
      elif on_center[0] == False: # code taken from plot_orb.py plot_thru_nuc()
          #delta = nuc_xyz_now - basis_center_xyz
          #print("Should these be equiv?? r_val (from xyz_eval to basis_center_xyz)", r_val, "delta", delta, "r from delta btw nuc_xyz and basis_center_xyz", np.sqrt(np.sum(delta**2)), flush=True)
          #r_xyz = basis_center_xyz + r_val * delta
          ret_val = STOnG_d_eval(eval_xyz, basis_center_xyz[0], alpha[0], d[0], orb_type[0])
          print("d orb_eval AFTER2 no r--- gaussian_r_val:", ret_val.shape, flush=True)

      # centered p-orb
      else:
        #print("returning nan for atom-centered p-orb!!!")
        ret_val = np.nan
        #raise RuntimeError("calculating gaussian_r_val() of atom centered p orbital to cusp, no p-cusps on p orb built in the code")
    else:
      raise RuntimeError("orbital category not classified in gaussian_r_val() --- orb_type = ", orb_type)

    ### EVALUATE CUSPS ###
    if cusp_info != None:
      #print("return cusp info") 
      a0, cusp_radius, zeta, dist = cusp_info[i]
      ret_cusp_val = s_cusp((r-dist).reshape(-1), cusp_radius, a0, zeta, ret_val.reshape(-1)) if np.abs(a0) > 0.0 else [] # center r vals on cusp if cusped
      ret_slater_val = slater_func((r-dist).reshape(-1), a0, zeta) if np.abs(a0) > 0.0 else [] # center r vals on cusp if cusped

      ret_cusp_list.append(ret_cusp_val)
      ret_slater_list.append(ret_slater_val)

      if lc_cusp_info != None:
        cusp_coeffs, order_n_list, cusp_type = lc_cusp_info[i]
        print("cusp_coeffs: ", cusp_coeffs[0].size, cusp_coeffs[1].size,  "\n", cusp_coeffs[0], "\n", cusp_coeffs[1])

        if type(cusp_type) == type([]): # both cusp types coeffs provided - slater_polty then _plus_ratio
          print("PLOT BOTH - n gaussian_r_and_cusp", flush=True)
          print("order_n_list", order_n_list, cusp_coeffs[0].shape, flush=True)
          slater_poly_val = np.sum( cusp_coeffs[0].reshape(-1,1) * P_cusp_func((r-dist).reshape(-1), cusp_radius, a0, zeta, ret_val.reshape(-1), order_n_list), axis=0) # sum_k ([k,1] * [k,m], axis=0) --> [m, ]

          gauss_1_minus_b = gaussian_part_of_cusp_func((r-dist).reshape(-1), cusp_radius, ret_val.reshape(-1))
          b_P = slater_only_P_cusp_func((r-dist).reshape(-1), cusp_radius, a0, zeta, ret_val.reshape(-1), order_n_list)
          slater_poly_plus_ratio_val = gauss_1_minus_b + np.sum( cusp_coeffs[1].reshape(-1,1) * b_P, axis=0) # sum_k ([k,1] * [k,m], axis=0) --> [m, ]

          print("\tjust slater_poly", slater_poly_val.shape, flush=True)
          print("\tplus_ratio", slater_poly_plus_ratio_val.shape, flush=True)
          print("\tAre they diff?", np.array_equal(slater_poly_val, slater_poly_plus_ratio_val), flush=True)
          print("\tis slater_poly and just slater cusp diff?", np.array_equal(slater_poly_val,ret_cusp_val), flush=True)

          ret_lc_slater_cusp_list.append([slater_poly_val, slater_poly_plus_ratio_val])
          
        elif cusp_type == 'slater_poly':
          #print("Plot just slater_poly", flush=True)
          ret_lc_slater_cusp_val = np.sum( cusp_coeffs.reshape(-1,1) * P_cusp_func((r-dist).reshape(-1), cusp_radius, a0, zeta, ret_val.reshape(-1), order_n_list), axis=0) # sum_k ([k,1] * [k,m], axis=0) --> [m, ]
          ret_lc_slater_cusp_list.append(ret_lc_slater_cusp_val)

        elif cusp_type == 'slater_poly_plus_ratio':
          #print("Plot just slater_poly_plus_ratio", flush=True)
          gauss_1_minus_b = gaussian_part_of_cusp_func((r-dist).reshape(-1), cusp_radius, ret_val.reshape(-1))
          b_P = slater_only_P_cusp_func((r-dist).reshape(-1), cusp_radius, a0, zeta, ret_val.reshape(-1), order_n_list)

          ret_lc_slater_cusp_val = gauss_1_minus_b + np.sum( cusp_coeffs.reshape(-1,1) * b_P, axis=0) # sum_k ([k,1] * [k,m], axis=0) --> [m, ]
          ret_lc_slater_cusp_list.append(ret_lc_slater_cusp_val)

       #(1-b)gaussian * n order

    #print("ret_val", ret_val.shape)
    #print("ret_cusp_val", ret_cusp_val.shape)
    #print("ret_slater_val", ret_slater_val.shape, np.max(ret_slater_val))

    ret_list.append(ret_val)
    #print("len of slaterlc cusp list", len(ret_lc_slater_cusp_list), len(ret_lc_slater_cusp_list[0]), flush=True)

  if lc_cusp_info != None:
    return ret_list, ret_cusp_list, ret_slater_list, ret_lc_slater_cusp_list
  elif cusp_info != None:
    return ret_list, ret_cusp_list, ret_slater_list, []
  else:
    return ret_list, [], [], [] 

def STOnG_s(phi, theta, r, basis_center, integration_center, alpha, d, fd_shift=np.zeros((1,3))):
  """ convert to correct radial distance and eval s orb
      nuc centered integral (r,theta,phi)_B0 
        m = # of input (r, theta, phi)
        r_BA = distance of sample on B sphere from A 
  """
  xyz_Bmol = spherical_to_cartesian(phi, theta, r, integration_center)  # r_B0 --> xyz_Bmol, [m,3]
  r_BA = r_from_xyz(xyz_Bmol + fd_shift, basis_center)                  # xyz_Bmol - xyz_Amol --> r_BA, [m,1] 
  r = np.diagonal(r_BA) if r_BA.shape[1] > 1 else r_BA
  orb_val = STOnG_s_eval(r, alpha, d)
  #print("STOnG_s: \n", xyz_Bmol)
  #print(basis_center, fd_shift)
  #print(r_BA)
  #N = np.reshape(d * (2 * alpha / np.pi )**(3/4),[-1,1])                # [n,1]
  ##print(N)
  #exp_val = np.outer(alpha, r_BA**2)                                    # [n,m]
  ##print(exp_val)
  #orb_val = np.sum(N * np.exp(-exp_val), axis=0)
  ##print(orb_val)

  return orb_val  # [m, ]


def STOnG_s_eval(r, alpha, d):
  """ evaluate s-orb at given r, 
  r: [m,]/[m,1]/[1,m] and alpha/d: [n,]/[n,1]/[1,n] 
  """
  #print(r.shape)
  N = np.reshape(d * (2 * alpha / np.pi )**(3/4),[-1,1])                # [n,1]
  #print(N.shape)
  exp_val = np.outer(alpha, r**2)                                    # [n,m]
  #print(exp_val.shape)
  orb_val = np.sum(N * np.exp(-exp_val), axis=0).reshape(-1)
  #print(orb_val.shape)

  return orb_val  # [m, ]


def STOnG_p(phi, theta, r, basis_center, integration_center, alpha, d, orb_type):
  """ nuc centered integral (r,theta,phi)_B0 
        r_BA = distance of sample on B sphere from A 
            r: [m,]/[m,1]/[1,m] 
  """
  xyz_Bmol = spherical_to_cartesian(phi, theta, r, integration_center)  # r_B0 --> xyz_Bmol, [m,3]
  orb_val = STOnG_p_eval(xyz_Bmol, basis_center, alpha, d, orb_type - 2)
  return orb_val    # [m, ] 

def STOnG_p_eval(xyz, xyz_o, alpha, d, ax, r_for_plot=None):
  """ eval from xyz to xyz_o (basis_center)
        m = # of input (r, theta, phi)
        n = # contracted gaussians
        r_BA = distance of sample on B sphere from A 
        delta_BA = [k_B0 - k_BA] for (k=x,y,z) 
            r: [m,]/[m,1]/[1,m], xyz_o: [m or 1,3]; xyz: [3,]/[3,m]/[m,3]; alpha/d: [n,]/[n,1]/[1,n] 
  """
  if r_for_plot is not None:
    # replace current xyz
    r_vec = np.zeros(3)
    r_vec[ax] = r_for_plot
    xyz = xyz_o + r_vec
    #print(xyz)

  N = np.reshape(2.0 * np.sqrt(alpha) * d * (2.0 * alpha / np.pi )**(3/4),[-1,1])              # [n,1]
  r = r_from_xyz(xyz, xyz_o)                             # xyz_Bmol - xyz_Amol --> r_BA, [m,1] 
  r = np.diagonal(r) if r.shape[1] > 1 else r
  #print("r in STOnG_p_eval (should match in gaussian_r_eval)", r, r.shape, xyz.shape, xyz_o.shape,flush=True)
  xyz_BA = np.reshape(xyz-xyz_o, [-1,3])			        # xyz_Bmol - xyz_Amol = xyz_BA, [m,3] 
  #print(xyz_BA, flush=True)
  exp_val = np.outer(alpha, r**2)                                    # [n,m]
  #print(exp_val, flush=True)
  delta = xyz_BA[:,ax].reshape(-1) #(xyz - xyz_o).reshape(-1)[ax]   #[m, ]
  #print("delta shape and val", delta.shape, delta, flush=True)
  ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * delta                # [m, ] * [m, ]
  #print(ret_val.shape, flush=True)
  #print("STOnG_p_eval val: ", ret_val, flush=True)
  return ret_val    # [m, ] 

def STOnG_d(phi, theta, r, basis_center, integration_center, alpha, d, orb_type):
  """ nuc centered integral (r,theta,phi)_B0 
        r_BA = distance of sample on B sphere from A 
            r: [m,]/[m,1]/[1,m] 
  """
  xyz_Bmol = spherical_to_cartesian(phi, theta, r, integration_center)  # r_B0 --> xyz_Bmol, [m,3]
  orb_val = STOnG_d_eval(xyz_Bmol, basis_center, alpha, d, orb_type)
  return orb_val    # [m, ]

def STOnG_d_eval(xyz, xyz_o, alpha, d, ax, r_for_plot=None):
  """ eval from xyz to xyz_o (basis_center)
        m = # of input (r, theta, phi)
        n = # contracted gaussians
        r_BA = distance of sample on B sphere from A 
        delta_BA = [k_B0 - k_BA] for (k=x,y,z) 
            r: [m,]/[m,1]/[1,m], xyz_o: [m or 1,3]; xyz: [3,]/[3,m]/[m,3]; alpha/d: [n,]/[n,1]/[1,n] 
  """
  if r_for_plot is not None:
    # replace current xyz
    r_vec = np.zeros(3)
    r_vec[0] = r_for_plot
    xyz = xyz_o + r_vec
    #print(xyz)

  N = np.reshape((2048.0 * alpha**7.0 / np.pi**3.0 )**(1/4) * d,[-1,1])              # [n,1]
  r = r_from_xyz(xyz, xyz_o)                             # xyz_Bmol - xyz_Amol --> r_BA, [m,1] 
  r = np.diagonal(r) if r.shape[1] > 1 else r
  #print("r in STOnG_p_eval (should match in gaussian_r_eval)", r, r.shape, xyz.shape, xyz_o.shape,flush=True)
  xyz_BA = np.reshape(xyz-xyz_o, [-1,3])			        # xyz_Bmol - xyz_Amol = xyz_BA, [m,3] 
  #print(xyz_BA, flush=True)
  exp_val = np.outer(alpha, r**2)                                    # [n,m]
  #print(exp_val, flush=True)
  x = xyz_BA[:,0].reshape(-1) #(xyz - xyz_o).reshape(-1)[ax]   #[m, ]
  y = xyz_BA[:,1].reshape(-1) #(xyz - xyz_o).reshape(-1)[ax]   #[m, ]
  z = xyz_BA[:,2].reshape(-1) #(xyz - xyz_o).reshape(-1)[ax]   #[m, ]
  if ax == 5:
    ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * x * y                # [m, ] * [m, ]
  elif ax == 6:
    ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * y * z                # [m, ] * [m, ]
  elif ax == 7:
    ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * 144.0**-0.25 * (3.0 * z * z - x * x - y * y - z * z)                # [m, ] * [m, ]
  elif ax == 8:
    ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * x * z                # [m, ] * [m, ]
  else:
    ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * (x * x - y * y) * 0.5             # [m, ] * [m, ]
  #print("delta shape and val", delta.shape, delta, flush=True)
  #ret_val = np.reshape(np.sum(N * np.exp(-exp_val), axis=0),[-1]) * delta                # [m, ] * [m, ]
  #print(ret_val.shape, flush=True)
  #print("STOnG_p_eval val: ", ret_val, flush=True)
  return ret_val    # [m, ] 

# orthogonalize second s orb against the frist
def STOnG_orth_s(phi, theta, r, basis_center, integration_center, alpha_1, d_1, alpha_2, d_2, proj_2_1, fd_shift=np.zeros((1,3))):

  # orthogonalize 3s against 2s = 3s - 2s * [<3s * 2s> / <2s * 2s>]   
  orth_2_1 = STOnG_s(phi, theta, r, basis_center, integration_center, alpha_2, d_2, fd_shift) - proj_2_1 * STOnG_s(phi, theta, r, basis_center, integration_center, alpha_1, d_1, fd_shift)

  return orth_2_1  # [m, ]

def STOnG_orth_s_eval(r, alpha_1, d_1, alpha_2, d_2, proj_2_1):
  """   input: _2 = orth_orb, _1 = core 
        return: 2 - proj_1(2) * 1 """
  orth_2_1 = STOnG_s_eval(r, alpha_2, d_2) - proj_2_1 * STOnG_s_eval(r, alpha_1, d_1)
  return orth_2_1  # [m, ]

def grad_STOnG_s(phi, theta, r, basis_center, integration_center, alpha, d):
  """ nuc centered integral (r,theta,phi)_B0 
        m = # of input (r, theta, phi)
        n = # contracted gaussians
        r_BA = distance of sample on B sphere from A 
        delta_BA = [k_B0 - k_BA] for (k=x,y,z) 
  """
  xyz_Bmol = spherical_to_cartesian(phi, theta, r, integration_center)  # r_B0 --> xyz_Bmol, [m,3]
  r_BA = r_from_xyz(xyz_Bmol, basis_center)                             # xyz_Bmol - xyz_Amol --> r_BA, [m,1] 
  r = np.diagonal(r_BA) if r_BA.shape[1] > 1 else r_BA

  xyz_BA = np.reshape(xyz_Bmol - basis_center, [-1,3])			        # xyz_Bmol - xyz_Amol = xyz_BA, [m,3] 

  N = np.reshape(alpha * d * (2 * alpha / np.pi )**(3/4),[-1,1])                           # [n,1]

  exp_val = np.outer(alpha, r**2)                                    # [n,m]
  val = -2 * np.sum(N * np.exp(-exp_val), axis=0)                       # sum_n ([n, ], [n,m]) = [m, ]
  orb_val = xyz_BA * val[:,np.newaxis]                              # [m,3] * [m,1]

  return orb_val     # [m,3] of d/dx, d/dy, d/dz 


def grad_STOnG_orth_s(phi, theta, r, basis_center, integration_center, alpha_1, d_1, alpha_2, d_2, proj_2_1):

  # orthogonalize 3s against 2s = 3s - 2s * [<3s * 2s> / <2s * 2s>]  
  grad_orth_2_1 = grad_STOnG_s(phi, theta, r, basis_center, integration_center, alpha_2, d_2) - proj_2_1 * grad_STOnG_s(phi, theta, r, basis_center, integration_center, alpha_1, d_1)

  return grad_orth_2_1 # [m,3]


def grad_STOnG_p(phi, theta, r, basis_center, integration_center, alpha, d, orb_type):
  """ nuc centered integral (r,theta,phi)_B0 
        m = # of input (r, theta, phi)
        n = # contracted gaussians
        r_BA = distance of sample on B sphere from A 
  """

  xyz_Bmol = spherical_to_cartesian(phi, theta, r, integration_center)  # r_B0 --> xyz_Bmol, [m,3]
  r_BA = r_from_xyz(xyz_Bmol, basis_center)                             # xyz_Bmol - xyz_Amol --> r_BA, [m,1] 
  r = np.diagonal(r_BA) if r_BA.shape[1] > 1 else r_BA
  xyz_BA = np.reshape(xyz_Bmol - basis_center, [-1,3])			        # xyz_Bmol - xyz_Amol = xyz_BA, [m,3] 

  grad = np.zeros((alpha.size,len(r_BA),3))	                            # [n,m,3] 

  N = np.reshape(2.0 * np.sqrt(alpha) * d * (2.0 * alpha / np.pi )**(3/4),[-1,1])        # [n,1] 

  # px
  if orb_type == 2:
    grad[:,:,0] = -2.0 * np.outer(alpha, xyz_BA[:,0]**2) + 1.0                       # [n,m]
    grad[:,:,1] = -2.0 * np.outer(alpha, (xyz_BA[:,0] * xyz_BA[:,1])) 
    grad[:,:,2] = -2.0 * np.outer(alpha, (xyz_BA[:,0] * xyz_BA[:,2])) 
  # py
  elif orb_type == 3:
    grad[:,:,0] = -2.0 * np.outer(alpha, (xyz_BA[:,1] * xyz_BA[:,0]))
    grad[:,:,1] = -2.0 * np.outer(alpha, xyz_BA[:,1]**2) + 1.0
    grad[:,:,2] = -2.0 * np.outer(alpha, (xyz_BA[:,1] * xyz_BA[:,2]))
  # pz
  elif orb_type == 4:
    grad[:,:,0] = -2.0 * np.outer(alpha, (xyz_BA[:,2] * xyz_BA[:,0]))
    grad[:,:,1] = -2.0 * np.outer(alpha, (xyz_BA[:,2] * xyz_BA[:,1]))
    grad[:,:,2] = -2.0 * np.outer(alpha, xyz_BA[:,2]**2) + 1.0

  exp_val = np.outer(alpha, r**2)                                    # [n,m]
  val = N * np.exp(-exp_val)                                            # [n,m]
  orb_val = np.sum(val[:,:,np.newaxis] * grad, axis=0)                     # [n,m,1] * [n,m,3]

  return orb_val  # [m,3] wiht d/dx, d/dy, d/dz contributions


def lap_STOnG_s(phi, theta, r, basis_center, integration_center, alpha, d):
  """ nuc centered integral (r,theta,phi)_B0 
        m = # of input (r, theta, phi)
        n = # contracted gaussians
        r_BA = distance of sample on B sphere from A 
        delta_BA = [k_B0 - k_BA] for (k=x,y,z) 
  """
  xyz_Bmol = spherical_to_cartesian(phi, theta, r, integration_center)  # r_B0 --> xyz_Bmol, [m,3]
  r_BA = r_from_xyz(xyz_Bmol, basis_center)                             # xyz_Bmol - xyz_Amol --> r_BA, [m,1] 
  r = np.diagonal(r_BA) if r_BA.shape[1] > 1 else r_BA
  xyz_BA = np.reshape(xyz_Bmol - basis_center, [-1,3])			        # xyz_Bmol - xyz_Amol = xyz_BA, [m,3] 

  N = np.reshape(alpha * d * (2.0 * alpha / np.pi )**(3/4),[-1,1])        # [n,1] 

  exp_val = np.outer(alpha, r**2)                                    # [n,m]
  val = N * np.exp(-exp_val)                                            # [n,m]
  inner_val = np.sum((2.0 * alpha.reshape(-1,1,1) * xyz_BA**2 - 1.0),axis=-1)  #sum_3( [n,1,1] * [m,3] = [n,m,3]) = [n,m]

  orb_val = 2.0 * np.sum(val * inner_val, axis=0)                     # [n,m] * [n,m], sum over n

  return orb_val # [m, ]


def lap_STOnG_orth_s(phi, theta, r, basis_center, integration_center, alpha_1, d_1, alpha_2, d_2, proj_2_1):

  # orthogonalize 3s against 2s = 3s - 2s * [<3s * 2s> / <2s * 2s>]  
  lap_orth_2_1 = lap_STOnG_s(phi, theta, r, basis_center, integration_center, alpha_2, d_2) - proj_2_1 * lap_STOnG_s(phi, theta, r, basis_center, integration_center, alpha_1, d_1)

  return lap_orth_2_1  #[m, ]

def lap_STOnG_p(phi, theta, r, basis_center, integration_center, alpha, d, orb_type):
  """ nuc centered integral (r,theta,phi)_B0 
        m = # of input (r, theta, phi)
        n = # contracted gaussians
        r_BA = distance of sample on B sphere from A 
  """
  xyz_Bmol = spherical_to_cartesian(phi, theta, r, integration_center)  # r_B0 --> xyz_Bmol, [m,3]
  r_BA = r_from_xyz(xyz_Bmol, basis_center)                             # xyz_Bmol - xyz_Amol --> r_BA, [m,1] 
  r = np.diagonal(r_BA) if r_BA.shape[1] > 1 else r_BA
  xyz_BA = np.reshape(xyz_Bmol - basis_center, [-1,3])			        # xyz_Bmol - xyz_Amol = xyz_BA, [m,3] 

  N = np.reshape(-4.0 * alpha**(3/2) * d * (2.0 * alpha / np.pi )**(3/4),[-1,1])              # [n,1] 

  # px
  if orb_type == 2:
    val = -2.0 * np.outer(alpha, xyz_BA[:,0]**2) * np.sum(xyz_BA,axis=1) + np.reshape(2.0 * xyz_BA[:,0] + xyz_BA[:,1] + xyz_BA[:,2],[-1])     # [n,m] + [m,] = [n,m]
  # py
  elif orb_type == 3:
    val = -2.0 * np.outer(alpha, xyz_BA[:,1]**2) * np.sum(xyz_BA,axis=1) + np.reshape(xyz_BA[:,0] + 2.0 * xyz_BA[:,1] + xyz_BA[:,2],[-1])
  # pz
  elif orb_type == 4:
    val = -2.0 * np.outer(alpha, xyz_BA[:,1]**2) * np.sum(xyz_BA,axis=1) + np.reshape(xyz_BA[:,0] + xyz_BA[:,1] + 2.0 * xyz_BA[:,2],[-1])

  exp_val = np.outer(alpha, r**2)                                    # [n,m]
  inner_val = val * np.exp(-exp_val)                                    # [n,m]
  orb_val = np.sum(inner_val * N,axis=0)                                # [m, ]

  return orb_val


##################################
######## CUSPED ORBITAL ##########
##################################

def b4_coeff(r_cusp): 
  c1 = -3.0 / r_cusp**4 
  c2 =  8.0 / r_cusp**3
  c3 = -6.0 / r_cusp**2
  c4 =  0.0
  c5 =  1.0
  return c1, c2, c3, c4, c5

def b5_coeff(r_cusp):
  c1 = -6.0  / r_cusp**5
  c2 =  15.0 / r_cusp**4
  c3 = -10.0 / r_cusp**3
  c4 =  0.0 
  c5 =  0.0
  c6 =  1.0
  return c1, c2, c3, c4, c5, c6

##### 5th order polynomial function ####
def b(r, r_cusp):
  """ 5th order switching function
    can take values outside of the cusp radius """
  #print("in b", r.shape, r_cusp.shape, flush=True)
  r = np.reshape(np.abs(r),[-1,])
  r5 = r*r*r*r*r
  r4 = r*r*r*r
  r3 = r*r*r
  r2 = r*r
  
  c1, c2, c3, c4, c5, c6 = b5_coeff(r_cusp)

  b_eval = c1 * r5 + c2 * r4 + c3 * r3 + c4 * r2 + c5 * r + c6
  b_eval[np.abs(r) > r_cusp] = 0.0  # set b for r not in cusp region to 0.0

  return  b_eval # [m, ] 

def d1_b(phi, theta, r, r_cusp):
  """ 5th order switching function
    can NOT take values outside of the cusp radius """
  r = np.reshape(r,[-1,1])
  r3 = r*r*r
  r2 = r*r

  c1, c2, c3, c4, c5, c6 = b5_coeff(r_cusp)

  xyz = spherical_to_cartesian(phi, theta, r)   # [m,3]
  d1_b = 5.0 * c1 * r3 + 4.0 * c2 * r2 + 3.0 * c3 * r + 2.0 * c4 + c5 / r 
  grad_vec = d1_b * xyz 

  return grad_vec  # [m,3] vector of d/dx b, d/dy b, d/dz b terms


def d2_b(phi, theta, r, r_cusp):
  """ 5th order switching function
    can NOT take values outside of the cusp radius """
  r = np.reshape(r,[-1,1])
  r2 = r*r
  r3 = r*r*r

  c1, c2, c3, c4, c5, c6 = b5_coeff(r_cusp)

  xyz = spherical_to_cartesian(phi, theta, r)   # [m,3]
  xyz2 = xyz*xyz
  lap = (c1 * (5.0 * r3 + 15.0 * r * xyz2)
      + c2 * (4.0 * r2 + 8.0 * xyz2) 
      + 3.0 * c3 * (r +  xyz2 / r ) 
      + 2.0 * c4 
      + c5 * (1 / r - xyz2 / r3))   # [m,3]

  return np.sum(lap,axis=1)  # [m,] sum of d2/dx2 b, d2/dy2 b, d2/dz2 b terms

##### 4th order switching function ####
#def b(r, r_cusp):
#  """ 4th order switching function
#   can take values outside of the cusp radus, will set to 0.0 """
#  #print("in b", r.shape, r_cusp.shape, flush=True)
#  r = np.reshape(np.abs(r),[-1,])
#  r4 = r*r*r*r
#  r3 = r*r*r
#  r2 = r*r
#  
#  c1, c2, c3, c4, c5 = b4_coeff(r_cusp)
#
#  b_eval = c1 * r4 + c2 * r3 + c3 * r2 + c4 * r + c5
#  b_eval[np.abs(r) > r_cusp] = 0.0  # set b for r not in cusp region to 0.0
#
#  return  b_eval # [m, ] 
#
#def d1_b(phi, theta, r, r_cusp):
#  """ 4th order switching function
#   can NOT take values outside of the cusp radus, will set to 0.0 """
#  r = np.reshape(r,[-1,1])
#  r2 = r*r
#
#  c1, c2, c3, c4, c5 = b4_coeff(r_cusp)
#
#  xyz = spherical_to_cartesian(phi, theta, r)   # [m,3]
#  d1_b = 4.0 * c1 * r2 + 3.0 * c2 * r + 2.0 * c3 + c4 / r 
#  grad_vec = d1_b * xyz 
#
#  return grad_vec  # [m,3] vector of d/dx b, d/dy b, d/dz b terms
#
#
#def d2_b(phi, theta, r, r_cusp):
#  """ 4th order switching function
#   can NOT take values outside of the cusp radus, will set to 0.0 """
#  r = np.reshape(r,[-1,1])
#  r2 = r*r
#  r3 = r*r*r
#
#  c1, c2, c3, c4, c5 = b4_coeff(r_cusp)
#
#  xyz = spherical_to_cartesian(phi, theta, r)   # [m,3]
#  xyz2 = xyz*xyz
#  lap =  c1 * (4.0 * r2 + 8.0 * xyz2) + 3.0 * c2 * (xyz2 / r + r) + 2.0 * c3 + c4 * (1 / r - xyz2 / r3)   # [m,3]
#
#  return np.sum(lap,axis=1)  # [m,] sum of d2/dx2 b, d2/dy2 b, d2/dz2 b terms

def gaussian_part_of_cusp_func(r, r_cusp, gaussian_orb):
   """ return (1 - b) * gaussian """
   b_val = b(r, r_cusp)   # [m,] 
   ret_val = (1 - b_val) * gaussian_orb # [m, ] + [k,m] = [k,m], contribution from overlapping orbital and cusped function
   return ret_val #[k,m]

def P_cusp_func(r, r_cusp, a0, zeta, gaussian_orb, order_n=0):
  """ generalization of s_cusp 
      r/gauss_val must be [m,]
      return gauss_val if outside cusp radius 
      order_n, can be scalar or list
        order_n_list of [k,1], k = # slater funcs"""

  gauss_1_minus_b = gaussian_part_of_cusp_func(r, r_cusp, gaussian_orb)
  b_P = slater_only_P_cusp_func(r, r_cusp, a0, zeta, gaussian_orb, order_n)

  ret_val = gauss_1_minus_b + b_P   # [m, ] + [k,m] = [k,m], contribution from overlapping orbital and cusped function

  return ret_val #[k,m]

def slater_only_P_cusp_func(r, r_cusp, a0, zeta, gaussian_orb, order_n=0):
  order_n_list = np.atleast_2d(order_n).reshape(-1,1)
  if 1 in order_n_list:
    raise Exception("order_n != 1 in P_cusp_func() in order to preserve cusp condition")

  b_val = b(r, r_cusp)   # [m,] 
  r_n = (abs(r.reshape(1,-1))/r_cusp)**order_n_list  # [1,m] * [k,1] ---> [k,m]
  #print("r shape (1,m) : ", (r.reshape(1,-1)).shape, "order_n_list shape (k,1) : ", order_n_list.shape, "r^n shape should be (k x m) : ", r_n.shape)
  b_P = b_val * slater_func(r, a0, zeta)  * np.abs(r_n)   # [k,m]

  return b_P #[k,m]

def lap_P_cusp_func(phi, theta, r, r_cusp, a0, zeta, gauss, d1_gauss, d2_gauss, order_n=0):
  """ d1_gauss input as [m,3] """

  xyz_B0 = spherical_to_cartesian(phi, theta, r).T  # r_B0 --> xyz_B0, [m,3] -- > [3,m]
  grad_r = (xyz_B0) / r    # [3,m]
  lap_r = np.sum(1 - (xyz_B0*xyz_B0) / (r*r), axis=0) / r   # [m, ]

  b_val = b(r, r_cusp)              # [m, ] 
  b1 = d1_b(phi, theta, r, r_cusp).T  # [m,3] --> [3,m]
  b2 = d2_b(phi, theta, r, r_cusp)  # [m, ] 

  Q = a0 * np.exp(-zeta * r)        # [m, ] 1s slater function = cusp 
  d1_Q = d1_slater_func(r, a0, zeta, xyz_B0).T # [m,3] * [m, ] / [m, ] = [m,3] ---> [3,m] 
  d2_Q = d2_slater_func(r, a0, zeta, xyz_B0)   # [m, ]

  # TODO fix the missing dr/dx deriv 
  ret_val = (d2_gauss + b_val * (d2_Q * r**order_n 
                                 + 2.0 * order_n * r**(order_n - 1.0) * np.sum(d1_Q * grad_r, axis=0) 
                                 + Q * order_n * (order_n - 1.0) * r**(order_n - 2.0) * np.sum((grad_r*grad_r),axis=0) 
                                 + order_n * Q * r**(order_n - 1.0) * lap_r 
                                 - d2_gauss) 
            + 2.0 * np.sum( b1 * (d1_Q * r**order_n + Q * order_n * r**(order_n-1.0) * grad_r - d1_gauss.T), axis=0) 
            + b2 * (Q * r**order_n - gauss))

  return ret_val

# replace with above later to not have redundant code
def s_cusp(r, r_cusp, a0, zeta, gaussian_orb):
  b_val = b(r, r_cusp)   # only calc b_val if in cusp_radius, else 0
  #print("r", r)
  #print("b_val", b_val)
  b_Q = b_val * slater_func(r, a0, zeta) #* a0 * np.exp(-zeta * r)       # [m, ], atom specific cusp function * b
  ret_val = (1 - b_val) * gaussian_orb + b_Q  # [m, ], contribution from overlapping orbital and cusped function

  return ret_val    #[m,]

def slater_func(r, a0, zeta):
  """ r/a0 should be [m,] and zeta = scalars """
  #print(a0.shape, r.shape, zeta.shape,flush=True)
  #print("slater func a0 val", a0)

  Q = a0 * np.exp(-zeta * np.abs(r))       # [m, ], atom specific cusp function * b
  return Q #[m,]

def d1_slater_func(r, a0, zeta, delta):
  """ r/a0 should be [m,] and zeta = scalars, delta = [3,m] """

  d1_Q = ((- zeta * delta * slater_func(r,a0,zeta)) / r).T     # [m,3] * [m, ] / [m, ] = [m,3] 

  return d1_Q #[m, 3]

def d2_slater_func(r, a0, zeta, delta):
  """ r/a0 should be [m,] and zeta = scalars, delta = [3,m] """

  d2_Q = zeta * slater_func(r,a0,zeta) * np.sum( zeta * delta**2 / r**2 + delta**2 / r**3 - 1/r, axis=0)  # [m, ]

  return d2_Q #[m, ]

# new generalized
def lap_s_cusp(phi, theta, r, r_cusp, integration_center, a0, zeta, ao, d1_ao, d2_ao):
  """ nuc centered integral (r,theta,phi)_B0 
        m = num input (r,theta,phi)
  """
  #r = np.reshape(r,[-1,1])
  xyz_B0 = spherical_to_cartesian(phi, theta, r).T  # r_B0 --> xyz_B0, [m,3] -- > [3,m]

  b_val = b(r, r_cusp)              # [m, ] 
  b1 = d1_b(phi, theta, r, r_cusp)  # [m,3] 
  b2 = d2_b(phi, theta, r, r_cusp)  # [m, ] 

  Q = a0 * np.exp(-zeta * r)        # [m, ] 1s slater function = cusp 
  d1_Q = d1_slater_func(r, a0, zeta, xyz_B0) #((- a0 * zeta * xyz_B0 * np.exp(-zeta * r)) / r).T     # [m,3] * [m, ] / [m, ] = [m,3] 
  d2_Q = d2_slater_func(r, a0, zeta, xyz_B0) #a0 * zeta * np.exp(-zeta * r) * np.sum( zeta * xyz_B0**2 / r**2 + xyz_B0**2 / r**3 - 1/r, axis=0)  # [m, ]
 
  ret_val = d2_ao + b_val * (d2_Q - d2_ao) + 2.0 * np.sum( b1 * (d1_Q - d1_ao), axis=1) + b2 * (Q - ao)
  #print(xyz_B0.shape, d1_ao.shape, Q.shape, d1_Q.shape, d2_Q.shape, ret_val.shape, ret_val2.shape)

  return ret_val    #[m,]


####################################
########## ENERGY TERMS ############
####################################

def elec_nuc_potential(phi, theta, r, integration_center, nuc_pos_list, Z):
  tic = time.perf_counter()
  #print("IN elec_nuc_potential")
  xyz_Bmol = spherical_to_cartesian(phi, theta, r, integration_center)  # r_B0 --> xyz_Bmol, [m,3]

  r_diff_all = r_from_xyz(xyz_Bmol, nuc_pos_list)                       # xyz_Bmol - xyz_Nucmol --> r_BNuc, [m,N]
  #print("elec_nuc_potential xyz_Bmol shape should be [m,3]", xyz_Bmol.shape)
  #print("elec_nuc_potential r_diff_all shape should be [m,N]", r_diff_all.shape)

  toc = time.perf_counter()
  #print("eN diff", toc-tic, '\t', r_diff_all)
  return np.reshape(np.sum(-Z / r_diff_all, axis=1),[-1])  # [m, ]

# uses updated integrals, that contain the previously missing coordinate shift
def one_elec_energy(a0, ao_ind, nuc_ind, orb_type, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, alpha_core, d_core, proj_2_1, pyscf_norm, pyscf_energy, nuc_xyz, all_nuc_xyz, z_array, info_orb_rc):
  """ one electron energy of the whole cusped AO in the presence of the other nuclei, parametrically dependent upon a0 """
 
  z = z_array[nuc_ind]
  zeta = z  # for s cusps

  #print()
  #print("\t\tone_elec_energy()", nuc_ind, ao_ind, " --- a0", a0, flush=True)
  #print("ao_ind", ao_ind, flush=True)
  #print("nuc_ind", nuc_ind, flush=True)
  #print("cusp_radius", cusp_radius, flush=True)
  #print("basis_center_xyz", basis_center_xyz, basis_center_xyz.shape, flush=True)
  #print("nuc_xyz", nuc_xyz, nuc_xyz.shape, flush=True)
  #print("all_nuc_xyz", all_nuc_xyz, all_nuc_xyz.shape, flush=True)
  #print("alpha", alpha, flush=True)
  #print("d", d, flush=True)
  #print("z_array", z_array, flush=True)
  #print("pyscf_norm", pyscf_norm, flush=True)
  #print("pyscf_energy", pyscf_energy, flush=True)
  #print("orb_type", orb_type, flush=True)
  #print("zeta", zeta, flush=True)
  #print("on_center", on_center, flush=True)
  #print("orth_orb_shifted", orth_orb_shifted, flush=True)
  #print("proj_2_1", proj_2_1, flush=True)
  #print("alpha_core", alpha_core, flush=True)
  #print("d_core", d_core, flush=True)
  #print("\t\t\tinfo_orb_rc", id(info_orb_rc), info_orb_rc, flush=True) 

  n = m = 0 # for OG cusped slater 

  xi_xyz = (nuc_xyz - basis_center_xyz).reshape(-1)
  #print("shape of xi_xyz: ", xi_xyz.shape, flush=True)
  x_from_all_xyz = (nuc_xyz - all_nuc_xyz) #.reshape(-1)
  #print("x_from_all_xyz", x_from_all_xyz.shape, x_from_all_xyz, flush=True)

  # s - orbital
  if orb_type < 2:

    # orthogonalized s orbital
    if orth_orb_shifted > 0: # apply to 3g and 1g valence orbs
      alpha_core = alpha_core.reshape(-1,1)
      d_core = d_core.reshape(-1,1)
      proj_2_1 = proj_2_1[0]

      #print("\torthogonalized s-orbital", flush=True)
      if info_orb_rc[0] == 0.0:
        H_rc_gaussian, H_error_gaussian = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, orthogonalized=True, full_eN_pot=[z_array, x_from_all_xyz], cusped_eval=False)
        S_rc_gaussian, S_error_gaussian = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, orthogonalized=True, H=False, full_eN_pot=[z_array, x_from_all_xyz], cusped_eval=False)
        info_orb_rc[0] = H_rc_gaussian 
        info_orb_rc[1] = S_rc_gaussian

      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, orthogonalized=True, full_eN_pot=[z_array, x_from_all_xyz])
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, orthogonalized=True, H=False, full_eN_pot=[z_array, x_from_all_xyz])

    # s cusp on non-orthogonalized s orb
    else: 

      #print("\t****************** s-orbital", flush=True)
      if info_orb_rc[0] == 0.0:
        H_rc_gaussian, H_error_gaussian = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, full_eN_pot=[z_array, x_from_all_xyz], cusped_eval=False)
        S_rc_gaussian, S_error_gaussian = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, H=False, full_eN_pot=[z_array, x_from_all_xyz], cusped_eval=False)
        info_orb_rc[0] = H_rc_gaussian 
        info_orb_rc[1] = S_rc_gaussian

      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, full_eN_pot=[z_array, x_from_all_xyz])
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, H=False, full_eN_pot=[z_array, x_from_all_xyz])

  # p-orbital
  elif orb_type == 2 or orb_type == 3 or orb_type == 4:

    # s cusp on p orb tail
    if on_center == False:

      #print("\t********************************* p-orbital on_center2:", on_center, flush=True)
      if info_orb_rc[0] == 0.0:
        H_rc_gaussian, H_error_gaussian = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, full_eN_pot=[z_array, x_from_all_xyz], cusped_eval=False)
        S_rc_gaussian, S_error_gaussian = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, H=False, full_eN_pot=[z_array, x_from_all_xyz], cusped_eval=False)
        info_orb_rc[0] = H_rc_gaussian 
        info_orb_rc[1] = S_rc_gaussian

      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, full_eN_pot=[z_array, x_from_all_xyz])
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, H=False, full_eN_pot=[z_array, x_from_all_xyz])

    # atom centered p orbital ---> p cusp on p orb
    else: 
      zeta = z/2
      raise RuntimeError("calculating one_elec_energy() of atom centered p orbital to cusp, no p-cusps on p orb built in the code, a0/ao_ind/nuc_ind/orb_type/on_center: ", a0, ao_ind, nuc_ind, orb_type, on_center)
      
  elif orb_type == 5 or orb_type == 6 or orb_type == 7 or orb_type == 8 or orb_type == 9:

    # s cusp on p orb tail
    if on_center == False:

      #print("\t********************************* p-orbital on_center2:", on_center, flush=True)
      if info_orb_rc[0] == 0.0:
        H_rc_gaussian, H_error_gaussian = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, full_eN_pot=[z_array, x_from_all_xyz], cusped_eval=False)
        S_rc_gaussian, S_error_gaussian = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, H=False, full_eN_pot=[z_array, x_from_all_xyz], cusped_eval=False)
        info_orb_rc[0] = H_rc_gaussian 
        info_orb_rc[1] = S_rc_gaussian

      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, full_eN_pot=[z_array, x_from_all_xyz])
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, 3, H=False, full_eN_pot=[z_array, x_from_all_xyz])
  else:
    raise RuntimeError("orbital category not classified in one_elec_energy() --- orb_type = ", orb_type)

  energy_val = one_e_val(H_rc, S_rc, pyscf_energy, pyscf_norm, info_orb_rc)

  return energy_val

# new integrals
def one_e_val(H_cusp_rc, S_cusp_rc, pyscf_E, pyscf_S, info_orb_rc):
  """ calculate one electron energy of the cusped region (to be minimized by a0)
      in the presence of the orbs being cusped and all nuclei.
      params:
        - H_cusp_rc: energy (with full elec-nuc potential) in the cusp region 
        - S_cusp_rc: norm of the cusp region   
        - pyscf_E: pyscf energy of the full orbital being cusped (orthogonalization acounted for)
        - pyscf_S: pyscf norm of the orbital being cusped 
        - info_orb_rc: len 2 list of [energy, norm] of the gaussian within the cusp radius
  """
  H_gauss_rc, S_gauss_rc = info_orb_rc 

  energy = (pyscf_E - H_gauss_rc + H_cusp_rc)
  norm = (pyscf_S - S_gauss_rc + S_cusp_rc)
 
  return energy / norm 

########################################
############### OPT a0 #################
########################################

def get_a0_matrix(options, stretched=False):
  # reassign if on savio or work computer
  num_cores = cpu_count()
  #num_cores = int(os.getenv("OMP_NUM_THREADS"))
  #num_cores = int(os.getenv("SLURM_CPUS_PER_TASK"))
  #num_cores=32
  #print("NUM of CORES:", num_cores)

  tic_fill_a0_mat = time.perf_counter()
  print()
  print("===================================================================")
  print("Calculate a0 matrix for", options['basis_type'], "basis --- stretched: ", stretched, flush=True)

  # molecular info
  z_array = options['Z'].flatten()
  
  num_nuc = len(z_array) 
  num_ao = options['nbf']
  #print()
  #print("num nuc/AO: ", num_nuc, num_ao)

  final_a0_matrix = np.zeros((num_nuc, num_ao)) 
        
  #print() 
  #print("cusp radius matrix (nuc x AOs)\n", options['cusp_radii_mat'])
  #print() 
  #print("orthogonalized orb array shifted\n", options['orth_orb_array'])
  #print() 
  #print("basis orb type\n", options['basis_orb_type'])
  #print() 
  
  # find unique nuclei (ie, 1 and 6), calc max orb for each in dict 
  unique_nuc = []
  max_orb_info = {}
  run_pairs = []

  #print()
  #print("getting max orbital values for each atom type")
  #print("Nuceli and orbital indices to add to max dict: ")
  for ind, Z in enumerate(z_array):
     if Z not in unique_nuc:    # if new atom - calculate max value corresponding to all it's basis functions, store in dict
        unique_nuc.append(Z)
        # ind of orbitals for 1st appearance of this nuclei
        max_ind = np.where(options["basis_centers"].astype(int) == ind)[0]
        print(Z, max_ind)
        ij, all_nuc_xyz, nuc_xyz, ao_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj, pyscf_s, pyscf_en, max_orb_info[int(Z)] = gaussian_info_and_eval(ind, max_ind, stretched, options, get_bf='max')

  #print("\nDictionary:")
  #print(max_orb_info)
  #print()

  nuc_list = np.arange(num_nuc)     # all nuc
  orb_list = np.arange(num_ao)      # all AOs
  #nuc_list_diff = orb_list_diff = []

  #if stretched == False:
  #  all_nuc_xyz = get_mol_xyz(options['nuclei'])
  #print("\nNuclei coordinates: \n",all_nuc_xyz, flush=True)
  #else:
  #  all_nuc_xyz = get_mol_xyz(options['nuclei_s'])
  #  #nuc_list_diff, orb_list_diff, final_a0_matrix = prep_a0_s_matrix(get_mol_xyz(options['nuclei']), get_mol_xyz(options['nuclei_s']), options["basis_centers"], options["cusp_a0"])
  #  print("\nNuclei_s coordinates: \n",all_nuc_xyz, flush=True)
  #  #print()
  #  #print("Calculate stretched a0 updating nuc/orbs:", nuc_list_diff, orb_list_diff)
  #  #print(final_a0_matrix)
  #  #print()

  # determine cusp elements to opt
  for i in nuc_list:
    for j in orb_list:   
      #print("nuc, orb: ", i, j, flush=True) 
      ij, all_nuc_xyz, nuc_xyz, ao_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj, pyscf_s, pyscf_en, bf_vals = gaussian_info_and_eval(i, j, stretched, options, get_bf='nuc')
      
      for val in bf_vals:
        if np.isnan(val) == False:

          Z = int(z_array[basis_center_ind[0]]) 
          ao_max = max_orb_info[Z][ao_type[0]]

          #print("nuc, orb, val, ao_max: ", i, j, val, ao_max, flush=True)

          ao_on_nuc = np.abs(val/ao_max)

          if (ao_on_nuc > 10e-16) and (r_cusp > 0.0):
            #print("add large enough gaussian value",i,j, '%.10f' % ao_on_nuc, val, ao_max) 
            run_pairs.append((i,j,val))
          #else:
          #  print("too small (< 10e-16) gaussian value or 0.0 cusp radius",i,j, '%.10f' % ao_on_nuc) 

        #else:
        #  print("no cusp on atom centered p orb: ",i,j, val) 

  # setting a0 to 1 instead of optimizing
  print("Hardcoding a0 to 1 if being cusped else 0")
  if len(run_pairs) > 0:
    for val in run_pairs:
      final_a0_matrix[val[0],val[1]] = 1. #val[-1]
  else: 
    print("using default zero matrix for cusp_a0")
  ##print()
  ##print("========================================")
  ##print("pairs to run: ", run_pairs, flush=True)
  ##print("========================================")
  ##print()
  #input_vals = [gaussian_info_and_eval(*ijv[:2], stretched, options) + (z_array, ijv[-1], ) for ijv in run_pairs]
  ##input_vals = [gaussian_info_and_eval(*ijv[:2], stretched, options) + (z_array, ijv[-1]) for ijv in run_pairs]
  ##print("INPUT VALS from gaussian_info_eval", type(input_vals), len(input_vals), len(input_vals[0]))
  ##print(input_vals[0],flush=True)
  ##print()

  ##all_AOs = [ a0_opt_val(*input_val) for input_val in input_vals ]   # not using multiprocessing

  #if len(input_vals) > 0:
  #  with Pool(num_cores) as p:
  #      all_AOs = p.starmap(a0_opt_val, input_vals) 

  #  for val in all_AOs:
  #    final_a0_matrix[val[0],val[1]] = val[-1]
  #else: 
  #  print("using default zero matrix for cusp_a0")

  #print("\n************************\nFINAL a0 MATRIX corrected integrals \n", np.array2string(final_a0_matrix, separator=', ', formatter={'float_kind': lambda x: "%.12f" % x})) #, "\nprojection matrix\n", proj_ao_mat, flush=True)

  return final_a0_matrix


def prep_a0_s_matrix(nuc_xyz, nuc_xyz_s, basis_center, a0_mat):
    """ 0 out rows and colums to be replaced for stretched geom """
    # nuc that have stretched
    diff = np.sum(nuc_xyz_s - nuc_xyz, axis=1)
    nuc_diff_ind = np.flatnonzero(diff)
    print("nuclei indices that are different: ", nuc_diff_ind)

    # corresponidng AOs to move
    ao_diff_ind = [np.where(basis_center == i)[0][0] for i in nuc_diff_ind]
    print("ao_diff_ind", ao_diff_ind)

    # initialize new a0 to calc (nuc rows/AO columns)
    a0_mat_s = a0_mat.copy()
    a0_mat_s[nuc_diff_ind] = 0.0    # nuc rows to 0.0
    a0_mat_s[:, ao_diff_ind] = 0.0    # dif ao columns to 0.0

    #print("ao_mat_s prepped", a0_mat_s)
    
    # nuc/aos to opt, nuc columns, ao rows
    return nuc_diff_ind, ao_diff_ind, a0_mat_s    # return rows and columns to change


# uses updated integrals 
def a0_opt_val(ij, all_nuc_xyz, nuc_xyz, ao_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj, pyscf_s, pyscf_en, z_array, initial_a0):
    nuc_ind = ij[0]
    ao_ind = ij[1]
    orb_info = [0.0,0.0]  # initialize
    opt_a0 = minimize(one_elec_energy, x0=initial_a0, method="BFGS",
            args=(
                    ao_ind,
                    nuc_ind, 
                    ao_type[0],
                    on_center[0],
                    basis_center_xyz,
                    alpha.reshape(-1,1),
                    d.reshape(-1,1),
                    r_cusp[0],
                    orth_orb_shifted[0],
                    alpha_core,
                    d_core,
                    proj,
                    pyscf_s[0],
                    pyscf_en[0],
                    nuc_xyz,
                    all_nuc_xyz,
                    z_array,
                    orb_info    
                    ))
    a0 = opt_a0.x
    #print("nuc/AO optimized a0:", nuc_ind, ao_ind, a0)
    return nuc_ind, ao_ind, a0


########################################
########### OPT q_i coeffs #############
########################################
#def get_cusp_coeff_matrix(options, order_n_list, stretched=False): # or input options which ontains cusp_a0
def get_cusp_coeff_matrix(options, stretched=False): # or input options which ontains cusp_a0
  """ order_n_list = list of bth order terms to add to slater cusp
  return [ n, num_orb, num_nuc, ] of Q_i coeffs for each cusp (ao over nuc)
 
  """
  #print("cusp_a0 in get_cusp_coeff_matrix\n", options["cusp_a0"], flush=True)
  order_n_list = options['order_n_list']
  # reassign if on savio or work computer
  num_cores = cpu_count()
  #num_cores = int(os.getenv("OMP_NUM_THREADS"))
  #num_cores = int(os.getenv("SLURM_CPUS_PER_TASK"))
  #num_cores=32
  #print("NUM of CORES:", num_cores)
  #print()
  a0_ind_to_opt = np.argwhere(np.abs(options["cusp_a0"]) > 0.)
  #a0_ind_to_opt = list(map(tuple,np.stack(np.nonzero(options["cusp_a0"]))))

  #print("ind_to_opt (nuc x orb)", a0_ind_to_opt)
  #print("cusp_radii_mat\n", options["cusp_radii_mat"])
  cusp_coeff_mat = np.zeros((options["cusp_a0"].shape[0], options["cusp_a0"].shape[1], len(order_n_list) ))  # [ num_qi_coeffs, num_orb, num_nuc, ]
  #print("\ncusp_coeff_mat", cusp_coeff_mat.shape, "\n", cusp_coeff_mat)
  #print("\ncusp_coeff_mat first element", cusp_coeff_mat[0,0,:])
  #cusp_coeff_mat = np.zeros((len(order_n_list), options["cusp_a0"].shape[-1], options["cusp_a0"].shape[0] ))  # [ num_qi_coeffs, num_orb, num_nuc, ]
  #for ij in a0_ind_to_opt:
  #  print(ij, flush=True)
  #  nuc = ij[0]
  #  orb = ij[1]
  #  print( options["cusp_a0"][nuc, orb], flush=True)
  #  print(order_n_list, flush=True)
  #  print(options['Z'][nuc],flush=True)
    #print(gaussian_info_and_eval(*ij, stretched, options) + (options['Z'][ij[0]], options["cusp_a0"][ij], order_n_list, ), flush=True)  # each element of list is info coresponding to a single orb/nuc pair
  input_info = [gaussian_info_and_eval(*ij, stretched, options) + (options['Z'][(ij[0])], options["cusp_a0"][(ij[0]),(ij[1])], order_n_list, options["cusp_type"]) for ij in a0_ind_to_opt]  # each element of list is info coresponding to a single orb/nuc pair
  #print("input_info", input_info, flush=True)

  tic_Pn = time.perf_counter()
  with Pool(num_cores) as p:
    all_coeff_vecs = p.starmap(cusp_coeff_vec, input_info) # returns orb_ind, nuc_ind, normed_coeff_vec

  print()
  toc_Pn = time.perf_counter()
  print("time to calc all Pn coeffs (min):", (toc_Pn-tic_Pn)/60.,flush=True)
  print()
  
  #tic2_Pn = time.perf_counter()
  #all_coeff_vecs = [cusp_coeff_vec(*input_val) for input_val in input_info] # returns orb_ind, nuc_ind, normed_coeff_vec
  #toc2_Pn = time.perf_counter()
  #print("time to calc all Pn coeffs WITHOUT Pool:", toc2_Pn-tic2_Pn,flush=True)
  #print()

  for val in all_coeff_vecs:
    cusp_coeff_mat[val[0], val[1], :] = val[-1] # assign coeff vector to corresponding a0 nuc/orb/indice
    #cusp_coeff_mat[:, val[0], val[1]] = val[-1] # assign coeff vector to corresponding a0 nuc/orb/indice

  #print("cusp_coeff_mat", cusp_coeff_mat, flush=True)
  return cusp_coeff_mat

def cusp_coeff_vec(ij, all_nuc_xyz, nuc_xyz, orb_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, core_ind, alpha_core, d_core, proj_2_1, pyscf_s, pyscf_en, Z, a0, order_n_list, cusp_type):
  """ input: (the output of gaussian_info_and_eval for a single {nuc/orb and a0!=0} with (Z, a0, orb_n_list) appended)
  this func: build matrices out of order_n_list = list of i, i=polynomial of slater, diag to get {q_i} per a0 """
  alpha = alpha.reshape(-1,1)
  d = d.reshape(-1,1)
  nuc_ind = ij[0] 
  ao_ind  = ij[1] 
  #print("cusp_coeff_vec nuc/orb", nuc_ind, ao_ind, "\ta0 = ", a0)
  # build eigenvalue prob in P_i basis, diag to get q_i coeffs
  H_mat = np.zeros((len(order_n_list), len(order_n_list)))
  S_mat = np.zeros((len(order_n_list), len(order_n_list)))
  #print(H_mat)

  # < P_i | H | P_j > and < P_i | P_j > - loop through elements for now

  # if using the slater_poly_plus_ratio basis: (1-b)X + sum q_n * bQr^n
  if cusp_type == "slater_poly_plus_ratio":
  #######################################################  
    #print("Using slater_poly_plus_ratio basis")
    # Get main block of elements: < bQr^n | H | bQr^m >
    for count_n, n in enumerate(order_n_list):
      for count_m, m in enumerate(order_n_list):
        H_mat[count_n,count_m], S_mat[count_n,count_m] = cusp_ij_elec_energy(a0, n, m, nuc_ind, orb_type, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, alpha_core, d_core, proj_2_1, nuc_xyz, Z, 0)

    #print("\nHmat in P_n basis", H_mat.shape) 
    #print(H_mat)
    #print("\nSmat in P_n basis") 
    #print(S_mat)
    #print()

    # Get first row and column corresponding to elements < (1-b)X | H | bQr^m >
    H_row_0 = np.zeros((len(order_n_list)+1 , 1))
    S_row_0 = np.zeros((len(order_n_list)+1, 1))

    # Except first entry is < (1-b)X | H | (1-b)X >
    H_row_0[0][0], S_row_0[0][0] = cusp_ij_elec_energy(a0, n, m, nuc_ind, orb_type, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, alpha_core, d_core, proj_2_1, nuc_xyz, Z, 2)
    
    for count_m, m in enumerate(order_n_list):
      H_row_0[count_m+1,0], S_row_0[count_m+1,0] = cusp_ij_elec_energy(a0, n, m, nuc_ind, orb_type, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, alpha_core, d_core, proj_2_1, nuc_xyz, Z, 1)

    #print("H Row 0:\n", H_row_0, flush=True)
    H_mat = np.concatenate((H_row_0[1:][:], H_mat), axis=1)
    H_mat = np.concatenate((H_row_0.reshape(1,-1), H_mat), axis=0)
    #print("\nHmat after adding row", H_mat.shape, flush=True) 
    #print(H_mat,flush=True)
    S_mat = np.concatenate((S_row_0[1:][:], S_mat), axis=1)
    S_mat = np.concatenate((S_row_0.reshape(1,-1), S_mat), axis=0)
    #print("\nSmat after adding row", flush=True) 
    #print(S_mat, flush=True)
    #print()
    # NEXT solve eigenvalue prob
    evals, evec_mat = eig(H_mat, S_mat)

    idx = evals.argsort()	# indices that sort array in acsending order
    evec_best = evec_mat[:,idx[0]]		# resort corrresponding eigenvectors
    
    # test for linear dependence
    evals_S, evec_S = np.linalg.eig(S_mat)
    #print("Evals of slater_poly_plus_ratio mat", evals_S, flush=True) 
 
    #print("All sorted evals: ", evals[idx])
    #print("All sorted evecs: ",evec_mat[:,idx])
    #print("All sorted and normed evecs: ",evec_mat[:,idx]/np.sum(evec_mat[:,idx], axis=0))
    #print()
    #print("Best eval: ", evals[idx[0]])
    #print("Best evec: ", evec_best)

    # normalize st vec sums to 1
    coeff_vec = evec_best / evec_best[0]
    coeff_vec = coeff_vec[1:]
  #######################################################  

  # if using the slater_poly basis: sum q_n[(1-b)X + bQr^n]
  elif cusp_type == "slater_poly":
  #######################################################  
    #print("Using slater_poly basis")
    #print()
    for count_n, n in enumerate(order_n_list):
      for count_m, m in enumerate(order_n_list):
        H_mat[count_n,count_m], S_mat[count_n,count_m] = cusp_ij_elec_energy(a0, n, m, nuc_ind, orb_type, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, alpha_core, d_core, proj_2_1, nuc_xyz, Z, 3)

    #print("\nHmat in P_n basis", H_mat.shape) 
    #print(H_mat)
    #print("\nSmat in P_n basis") 
    #print(S_mat)
    #print()
    evals, evec_mat = eig(H_mat, S_mat)

    evals_S, evec_S = np.linalg.eig(S_mat)
    #print("Evals of slater_poly", evals_S, flush=True) 
    
    idx = evals.argsort()	# indices that sort array in acsending order
    evec_best = evec_mat[:,idx[0]]		# resort corrresponding eigenvectors
    coeff_vec = evec_best / np.sum(evec_best)
  #######################################################  
  else:
    raise RuntimeError('need to specify either "slater_poly" or "slater_poly_plus_ratio" in input "cusp_type"') 
 
  ##print("check = 1: ", np.sum(coeff_vec))
  #print("normalized coeff_vec", coeff_vec,flush=True)
  #print("Shape of evec: ", coeff_vec.shape, flush=True)

  #print("nuc_ind: ", nuc_ind, flush=True)
  #print("ao_ind: ", ao_ind, flush=True)
  #print("a0: ", a0, flush=True)
  #print("alpha:\n", alpha, flush=True)
  #print("d:\n", d, flush=True)

  return nuc_ind, ao_ind, coeff_vec

def cusp_ij_elec_energy(a0, n, m, nuc_ind, orb_type, on_center, basis_center_xyz, alpha, d, cusp_radius, orth_orb_shifted, alpha_core, d_core, proj_2_1, nuc_xyz, z, int_type):
  """ one electron energy element only within the cusp of nth-slater func
      returns H_nm = <Pn|H|Pm> and S_nm = <Pn|Pm>; P_n = nth order slater  """
 
  zeta = z

  xi_xyz = (nuc_xyz - basis_center_xyz).reshape(-1)
  #print("shape of xi_xyz: ", xi_xyz.shape, flush=True)
  #print()
  #print("\t\tcusp_ij_elec_energy(), nuc_ind", nuc_ind, "Pi and Pj ind", n,m, " --- a0", a0, flush=True)
  #print("nuc_ind", nuc_ind, flush=True)
  #print("cusp_radius", cusp_radius, flush=True)
  #print("basis_center_xyz", basis_center_xyz, flush=True)
  #print("nuc_xyz",nuc_xyz, flush=True)
  #print("xi_xyz (vector from bf center to nuc center)",xi_xyz, flush=True)
  #print("alpha", alpha, flush=True)
  #print("d", d, flush=True)
  #print("z", z, flush=True)
  #print("orb_type", orb_type, flush=True)
  #print("zeta", zeta, flush=True)
  #print("on_center", on_center, flush=True)
  #print("orth_orb_shifted", orth_orb_shifted, flush=True)
  #print("proj_2_1", proj_2_1, flush=True)
  #print("alpha_core", alpha_core, flush=True)
  #print("d_core", d_core, flush=True)

  # s - orbital
  if orb_type < 2:

    # orthogonalized s orbital
    if orth_orb_shifted > 0: # apply to 3g and 1g valence orbs
      alpha_core = alpha_core.reshape(-1,1)
      d_core = d_core.reshape(-1,1)
      proj_2_1 = proj_2_1[0]
      #S_rc, S_error = dblquad(lambda phi, theta: 
      #                        fixed_quad(lambda r: 
      #                                   P_cusp_func(r,cusp_radius,a0,zeta,
      #                                                 STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1),
      #                                                 n) *
      #                                   P_cusp_func(r,cusp_radius,a0,zeta,
      #                                                 STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1),
      #                                                 m) *
      #                                   r**2 * np.sin(theta), 0, cusp_radius, n=5)[0],
      #                        0.0, np.pi, 0.0, 2*np.pi)
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type, orthogonalized=True, H=False)

	                  #dblquad(lambda phi, theta: 
                          #    fixed_quad(lambda r: 
                          #               P_cusp_func(r,cusp_radius,a0,zeta,
                          #                           STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1),
                          #                           n) *
                          #               lap_P_cusp_func(phi, theta, r, cusp_radius, a0, zeta, 
                          #                               STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1),
                          #                               grad_STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1),
                          #                               lap_STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1),
                          #                               m)
                          #               * r**2 * np.sin(theta), 0, cusp_radius, n=5)[0],
                          #    0.0, np.pi, 0.0, 2*np.pi)

      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type, orthogonalized=True)
		        #dblquad(lambda phi, theta: 
                        #        fixed_quad(lambda r: 
                        #                   elec_nuc_potential(phi, theta, r, nuc_xyz, nuc_xyz, z) *
                        #                   P_cusp_func(r,cusp_radius,a0,zeta,
                        #                                 STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1),
                        #                                 n) *
                        #                   P_cusp_func(r,cusp_radius,a0,zeta,
                        #                                 STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1),
                        #                                 m) *
                        #                   r**2 * np.sin(theta), 0, cusp_radius, n=5)[0],
                        #        0.0, np.pi, 0.0, 2*np.pi)

    else:
      #S_rc, S_error = dblquad(lambda phi, theta: 
      #                        fixed_quad(lambda r: 
      #                                   P_cusp_func(r,cusp_radius,a0,zeta,
      #                                                STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d),
      #                                                 n) *
      #                                   P_cusp_func(r,cusp_radius,a0,zeta,
      #                                                STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d),
      #                                                 m) *
      #                                   r**2 * np.sin(theta), 0, cusp_radius, n=5)[0],
      #                        0.0, np.pi, 0.0, 2*np.pi)
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type, H=False)

		          #dblquad(lambda phi, theta: 
                          #    fixed_quad(lambda r: 
                          #               P_cusp_func(r,cusp_radius,a0,zeta,
                          #                            STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d),
                          #                             n) *
                          #               lap_P_cusp_func(phi, theta, r, cusp_radius, a0, zeta, 
                          #                               STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d),
                          #                               grad_STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d),
                          #                               lap_STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d),
                          #                               m)
                          #               * r**2 * np.sin(theta), 0, cusp_radius, n=5)[0],
                          #    0.0, np.pi, 0.0, 2*np.pi)

      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type)
                        #dblquad(lambda phi, theta: 
                        #        fixed_quad(lambda r: 
                        #                   elec_nuc_potential(phi, theta, r, nuc_xyz, nuc_xyz, z) *
                        #                   P_cusp_func(r,cusp_radius,a0,zeta,
                        #                                STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d),
                        #                                 n) *
                        #                   P_cusp_func(r,cusp_radius,a0,zeta,
                        #                                STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d),
                        #                                 m) *
                        #                   r**2 * np.sin(theta), 0, cusp_radius, n=5)[0],
                        #        0.0, np.pi, 0.0, 2*np.pi)

  # p-orbital
  elif orb_type == 2 or orb_type == 3 or orb_type == 4:
    # s cusp on p orb tail
    if on_center == False:
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type, H=False)
                      #dblquad(lambda phi, theta: 
                      #        fixed_quad(lambda r: 
                      #                   P_cusp_func(r,cusp_radius,a0,zeta,
                      #                                STOnG_p(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, orb_type),
                      #                                 n) *
                      #                   P_cusp_func(r,cusp_radius,a0,zeta,
                      #                                STOnG_p(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, orb_type),
                      #                                 m) *
                      #                   r**2 * np.sin(theta), 0, cusp_radius, n=5)[0],
                      #        0.0, np.pi, 0.0, 2*np.pi)

                          #dblquad(lambda phi, theta: 
                          #    fixed_quad(lambda r: 
                          #               P_cusp_func(r,cusp_radius,a0,zeta,
                          #                            STOnG_p(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, orb_type),
                          #                             n) *
                          #               lap_P_cusp_func(phi, theta, r, cusp_radius, a0, zeta, 
                          #                               STOnG_p(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, orb_type),
                          #                               grad_STOnG_p(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, orb_type),
                          #                               lap_STOnG_p(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, orb_type),
                          #                               m)
                          #               * r**2 * np.sin(theta), 0, cusp_radius, n=5)[0],
                          #    0.0, np.pi, 0.0, 2*np.pi)

      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type)
      #eN_rc, eN_error = #dblquad(lambda phi, theta: 
                        #        fixed_quad(lambda r: 
                        #                   elec_nuc_potential(phi, theta, r, nuc_xyz, nuc_xyz, z) *
                        #                   P_cusp_func(r,cusp_radius,a0,zeta,
                        #                                STOnG_p(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, orb_type),
                        #                                 n) *
                        #                   P_cusp_func(r,cusp_radius,a0,zeta,
                        #                                STOnG_p(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, orb_type),
                        #                                 m) *
                        #                   r**2 * np.sin(theta), 0, cusp_radius, n=5)[0],
                        #        0.0, np.pi, 0.0, 2*np.pi)
    else:
      zeta = z/2
      raise RuntimeError("calculating one_elec_energy() of atom centered p orbital to cusp, no p-cusps on p orb built in the code, a0/ao_ind/nuc_ind/orb_type/on_center: ", a0, ao_ind, nuc_ind, orb_type, on_center)
  
  # d-orbitals
  elif orb_type == 5 or orb_type == 6 or orb_type == 7 or orb_type == 8 or orb_type == 9:
    if on_center == False:
      S_rc, S_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type, H=False)
      H_rc, H_error = integrand_at_rtp.test_integrand_at_rtp_general(orb_type, cusp_radius, n, m, zeta, a0, d, alpha, xi_xyz, d_core, alpha_core, proj_2_1, int_type)
    else:
      zeta = z/2
      raise RuntimeError("calculating one_elec_energy() of atom centered p orbital to cusp, no p-cusps on p orb built in the code, a0/ao_ind/nuc_ind/orb_type/on_center: ", a0, ao_ind, nuc_ind, orb_type, on_center)
  else:
    raise RuntimeError("orbital category not classified in one_elec_energy() --- orb_type = ", orb_type)

  return H_rc, S_rc 

########################################
################ TESTS #################
########################################

def test_diff_cusp_funcs(ij, options):
  # ij is (nuc_ind, orb_ind) tuple

  ## 1s carbon orb
  #r = np.linspace(-10,10,100)
  #phi = theta =  
  #r_cusp =
  #Z = zeta = 6
  #a0 = 7.8642519984
  #orb_type = 0
  #orth_orb = 0
  #alpha = np.array([3.047524880e+03, 4.573695180e+02, 1.039486850e+02, 2.921015530e+01, 9.286662960e+00, 3.163926960e+00]) 
  #d = np.array([ 0.001834737132,  0.01403732281 ,  0.06884262226 ,  0.2321844432  ,  0.4679413484  ,  0.3623119853  ]) 
  #basis_center_xyz = nuc_xyz = np.array([0., 0., 0.])
  print("======================================================================") 
  a0 = options["cusp_a0"][(ij[0], ij[1])]
  print("ij = ", ij, "; a0 = ", a0)
  zeta = options["Z"][ij[0]]
  ij, all_nuc_xyz, nuc_xyz, orb_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj_2_1, pyscf_s, pyscf_en  = gaussian_info_and_eval(*ij, False, options)

  num_pts = 10
  r = np.linspace(0.0001, 1., num_pts)
  theta = np.random.uniform(low=0., high=2*np.pi, size=num_pts)
  phi = np.random.uniform(low=0., high=np.pi, size=num_pts)

  gaussian_orb = STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d)
  d1_gaussian_orb = grad_STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d)
  d2_gaussian_orb = lap_STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d)

  slater = s_cusp(r, r_cusp, a0, zeta, gaussian_orb)
  lap_slater = lap_s_cusp(phi, theta, r, r_cusp, nuc_xyz, a0, zeta, gaussian_orb, d1_gaussian_orb, d2_gaussian_orb)

  order_n = 0 # should recover original cusp func 
  #cusp_coeff_vec = np.array([1.]) #np.ones(len(order_n_list)) 
  cusp_coeff_val = 1. # one for not to see if we recover the slater func
  slater_linear_combo = cusp_coeff_val * P_cusp_func(r, r_cusp, a0, zeta, gaussian_orb, order_n)
  lap_slater_linear_combo = lap_P_cusp_func(phi, theta, r, r_cusp, a0, zeta, gaussian_orb, d1_gaussian_orb, d2_gaussian_orb, order_n)

  print()
  print()
  print()
  rtol=12
  if np.count_nonzero(np.round(slater - slater_linear_combo, rtol)) == 0:
    print("P_0 recovers slater eval \tPASSED")
  else:
    raise RuntimeError("P_0 DOES NOT recovers slater eval \tFAILED", slater - slater_linear_combo)

  if np.count_nonzero(np.round(lap_slater - lap_slater_linear_combo, rtol)) == 0:
    print("lap_P_0 recovers lap_slater eval \tPASSED")
  else:
    raise RuntimeError("lap_P_0 DOES NOT recovers lap_slater eval \tFAILED", lap_slater - lap_slater_linear_combo)
  
def test_grad(basis_center_xyz, nuc_xyz, alpha, d):
  h = 1e-6

  r = np.random.rand()
  theta = np.random.rand()
  phi = np.random.rand()

  val = STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d)   

  valplus_x = STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, [h, 0.0, 0.0])   
  valminus_x = STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, [-h, 0.0, 0.0])   
  
  valplus_y = STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, [0.0, h, 0.0])   
  valminus_y = STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, [0.0, -h, 0.0])   

  valplus_z = STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, [0.0, 0.0, h])   
  valminus_z = STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d, [0.0, 0.0, -h])   

  dx = (valplus_x - valminus_x)/(h*2)
  dy = (valplus_y - valminus_y)/(h*2)
  dz = (valplus_z - valminus_z)/(h*2)

  d2x = (valplus_x - 2* val + valminus_x)/(h**2)
  d2y = (valplus_y - 2* val + valminus_y)/(h**2)
  d2z = (valplus_z - 2* val + valminus_z)/(h**2)

  grad_val = grad_STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d) 
  grad_diff = grad_val - np.concatenate((dx, dy, dz))  

  lap_val = lap_STOnG_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha, d)     
  lap_diff = lap_val - (d2x+d2y+d2z) 

  print()
  print('\tgrad', grad_val, '\tfd grad: ', dx, dy, dz, '\tdiff: ', grad_diff[0])
  print('\tlap', lap_val, '\tfd lap:', d2x, d2y, d2z, "total =", d2x+d2y+d2z, '\tlap diff:', '%.8f' % lap_diff[0] )
  print()

def test_orth_grad(basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1):
  h = 1e-6

  r = np.random.rand()
  theta = np.random.rand()
  phi = np.random.rand()

  val = STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1)

  valplus_x = STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1, [h, 0.0, 0.0])   
  valminus_x = STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1, [-h, 0.0, 0.0])   
  
  valplus_y = STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1, [0.0, h, 0.0])   
  valminus_y = STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1, [0.0, -h, 0.0])   

  valplus_z = STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1, [0.0, 0.0, h])   
  valminus_z = STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1, [0.0, 0.0, -h])   

  dx = (valplus_x - valminus_x)/(h*2)
  dy = (valplus_y - valminus_y)/(h*2)
  dz = (valplus_z - valminus_z)/(h*2)

  d2x = (valplus_x - 2* val + valminus_x)/(h*h)
  d2y = (valplus_y - 2* val + valminus_y)/(h*h)
  d2z = (valplus_z - 2* val + valminus_z)/(h*h)

  grad_val = grad_STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1)
  grad_diff = grad_val - np.concatenate((dx, dy, dz))  

  lap_val = lap_STOnG_orth_s(phi, theta, r, basis_center_xyz, nuc_xyz, alpha_core, d_core, alpha, d, proj_2_1)
  lap_diff = lap_val - (d2x+d2y+d2z) 

  print()
  print('\tgrad', grad_val, '\tfd grad: ', dx, dy, dz, '\tDiff: ', grad_diff[0])
  print('\tlap', lap_val, '\tfd lap:', d2x, d2y, d2z, "total =", d2x+d2y+d2z, '\tlap Diff:', '%.8f' % lap_diff[0] )
  print()

def check_cpp_xmat(filename, options):
  # moved to a unit test in test_cusp_orbitals.py  --- no longer using this
  #np.set_printoptions(precision=12)
  print()
  print("Compare cusped orbitals to C++ output")
  print("orb_ind, elec_ind, nuc_ind, cusp_val, gauss_val, slater_val, b_val, elec-nuc_dist, nuc x, y, z, elec x, y, z, elec-orb_dist, proj, core_orb_ind, zeta")
  print()
  cpp_dat = np.genfromtxt(filename, delimiter=',')
  cpp_orb_ind = cpp_dat[:,0].astype(int)
  print("cpp_orb_ind: ", cpp_orb_ind)
  cpp_elec_ind = cpp_dat[:,1].astype(int)   # do not use this, arbitrary ind
  cpp_nuc_ind = cpp_dat[:,2].astype(int)
  cpp_cusp_val = cpp_dat[:,3] 
  cpp_gauss_val = cpp_dat[:,4] 
  cpp_slater_val = cpp_dat[:,5] 
  cpp_b = np.round(cpp_dat[:,6],6) 
  cpp_e_nuc_dist = cpp_dat[:,7] 
  cpp_nuc_xyz = cpp_dat[:,8:8+3] 
  cpp_e_xyz = cpp_dat[:,11:11+3] 
  cpp_e_orb_dist = cpp_dat[:,14] 
  cpp_proj = cpp_dat[:,15] 
  print(cpp_proj, flush=True)
  only_orth_ind = np.argwhere(cpp_proj)
  cpp_proj = cpp_proj[only_orth_ind].reshape(-1)
  print(cpp_proj, flush=True)
  cpp_core_ind = cpp_dat[:,16] 
  print(cpp_core_ind, flush=True)
  cpp_core_ind = cpp_core_ind[only_orth_ind].reshape(-1)
  print(cpp_core_ind, flush=True)
  print(cpp_dat.shape, cpp_nuc_xyz.shape, cpp_e_xyz.shape, cpp_orb_ind.shape, flush=True)
  cpp_zeta = cpp_dat[:,17] 

  r_e_nuc = np.diagonal(r_from_xyz(cpp_e_xyz, cpp_nuc_xyz))

  basis_center_ind = (options["basis_centers"][cpp_orb_ind]).astype(int).reshape(-1)
  all_nuc_xyz = get_mol_xyz(options['nuclei']) 
  r_e_orb= np.diagonal(r_from_xyz(cpp_e_xyz, all_nuc_xyz[basis_center_ind]))
  #print("r_e_orb", cpp_r.shape, r.shape, cpp_r.reshape(-1), flush=True)
  r_cusp = options["cusp_radii_mat"][cpp_nuc_ind,cpp_orb_ind]
  b_val = np.round(b(cpp_e_nuc_dist,r_cusp),6)

  a0_cusp = options["cusp_a0"][cpp_nuc_ind,cpp_orb_ind]
  zeta = options["Z"][cpp_nuc_ind].reshape(-1)
  orb_type = (options['basis_orb_type'][cpp_orb_ind]).reshape(-1)
  print("unsorted basis_orb_type: ", options['basis_orb_type'], flush=True)
  print("orb_type: ", orb_type)
  on_center = (basis_center_ind == cpp_nuc_ind)
  alpha = (options["basis_exp"][cpp_orb_ind]).reshape(-1, options['ng'])
  d = (options["basis_coeff"][cpp_orb_ind]).reshape(-1, options['ng'])

  print()
  orth_orb_shifted = np.atleast_1d(options['orth_orb_array'][cpp_orb_ind]).astype(int)
  orth_orb_bool = np.atleast_1d(orth_orb_shifted != 0)
  print("orth_orb_shifted", orth_orb_shifted, orth_orb_shifted.shape, orth_orb_shifted.dtype)
  print("orth_orb_bool", orth_orb_bool, orth_orb_bool.shape,  orth_orb_bool.dtype)
  core_true = np.where(orth_orb_bool == True)[0]
  print("core_true", core_true,flush=True)
  core_ind = orth_orb_shifted[core_true] - 1
  print("core_ind", core_ind, core_ind.shape)
  alpha_core = (options["basis_exp"][core_ind]).reshape(-1, options['ng'])
  print("alpha_core", alpha_core, alpha_core.shape)
  d_core = (options["basis_coeff"][core_ind]).reshape(-1, options['ng'])
  print("d_core", d_core, d_core.shape)
  proj = (options["proj_mat"][cpp_orb_ind[core_true],core_ind]).reshape(-1)
  print("proj", proj, proj.shape)
  print()
 
  gauss_val = gaussian_r_val(cpp_e_xyz, orb_type, alpha_core, d_core, all_nuc_xyz[basis_center_ind], cpp_nuc_xyz, alpha, d, on_center, orth_orb_shifted, proj, get_max=False)
  #gauss_val = gaussian_r_val(r_e_orb, orb_type, alpha_core, d_core, cpp_nuc_xyz, cpp_e_xyz, alpha, d, on_center, orth_orb, proj, get_max=False)
  cusp_val = s_cusp(cpp_e_nuc_dist, r_cusp, a0_cusp, zeta, gauss_val)
  slater_val = slater_func(cpp_e_nuc_dist, a0_cusp, zeta)
  print("slater sizes", slater_val.shape, cpp_slater_val.shape,flush=True)

  print()
  print()
  print()
  rtol=10
  if np.count_nonzero(np.round(cpp_e_nuc_dist-r_e_nuc,rtol)) == 0:
    print("r e-nuc\tPASSED")
  else:
    raise RuntimeError("r e-nuc\tFAILED - diff", cpp_e_nuc_dist-r_e_nuc)

  if np.count_nonzero(np.round(cpp_e_orb_dist-r_e_orb,rtol)) == 0:
    print("r e-orb\tPASSED")
  else:
    raise RuntimeError("r e-orb\tFAILED - diff", cpp_e_orb_dist-r_e_orb)

  if np.count_nonzero(np.round(cpp_b-b_val,rtol)) == 0:
    print("b\tPASSED")
  else:
    raise RuntimeError("b\tFAILED - diff", cpp_b-b_val)

  if np.count_nonzero(np.round(cpp_proj-proj,6)) == 0:
    print("proj\tPASSED")
  else:
    raise RuntimeError("proj\tFAILED - diff", cpp_proj-proj)

  if np.count_nonzero(cpp_core_ind-core_ind) == 0:
    print("core_ind\tPASSED")
  else:
    raise RuntimeError("core_ind\tFAILED - diff", cpp_core_ind-core_ind)

  if np.count_nonzero(np.round(cpp_zeta-zeta,1)) == 0:
    print("zeta\tPASSED")
  else:
    raise RuntimeError("zeta\tFAILED - diff", cpp_zeta-zeta)

  #np.set_printoptions(precision=12, threshold=sys.maxsize, linewidth=np.inf)
  if np.count_nonzero(np.round(slater_val - cpp_slater_val,rtol)) == 0:
    print("slater orbital\tPASSED")
  else:
    print()
    print(np.array2string(cpp_slater_val, separator=' ', formatter={'float_kind': lambda x: "%.12E" % x}),flush=True)
    print()
    print(np.array2string(slater_val, separator=' ', formatter={'float_kind': lambda x: "%.12E" % x}),flush=True)
    print()
    print("diff:", np.array2string((slater_val - cpp_slater_val), separator=' ', formatter={'float_kind': lambda x: "%.12E" % x}),flush=True)
    raise RuntimeError("slater orbital\tFAILED - diff", cpp_slater_val-slater_val)

  if np.count_nonzero(np.round(gauss_val - cpp_gauss_val,rtol)) == 0:
    print("gaussian orbital\tPASSED")
    print()
    print(cpp_gauss_val,flush=True)
    print()
    print(gauss_val,flush=True)
    print()
    print("diff:", gauss_val - cpp_gauss_val,flush=True)
  else:
    print()
    print(cpp_gauss_val,flush=True)
    print()
    print(gauss_val,flush=True)
    raise RuntimeError("gaussian orbital\tFAILED - diff", np.round(cpp_gauss_val-gauss_val,6))

  if np.count_nonzero(np.round(cusp_val - cpp_cusp_val,rtol)) == 0:
    print("cusp orbital\tPASSED")
  else:
    print()
    print(cpp_cusp_val,flush=True)
    print()
    print(cusp_val,flush=True)
    raise RuntimeError("cusp orbital\tFAILED - diff", cpp_cusp_val-cusp_val)

  print()
  print("C++ X_mat evaluation is equivalent to python!!!")
  print()

if __name__ == '__main__': 
  print("Testing linear combo slater func on Carbon 1s")

  with open(sys.argv[1], 'rb') as fp:
    options = pickle.load(fp)

  a0_ind_to_opt = np.argwhere(np.abs(options["cusp_a0"]) > 0.)

  for ij in a0_ind_to_opt:
    test_diff_cusp_funcs(ij, options)

  #print("Calculate matrix of cusp radii!")
#  #global_cusp = None if options.get("cusp_radius") == None else options['cusp_radius']
#
#  #r_cusp_array, r_cusp_matrix, core_ind_array, orth_orb_array = set_basis_orb_info(global_cusp, options['pyscf_basis'], len(options['basis_orb_type']), options['Z'])
#  #options['cusp_radii_mat'] = r_cusp_matrix 	# comment out if you want a constant r_cusp defined above
#  #options['core_ind_array'] = np.reshape(core_ind_array, [-1,1])	# make column form
#  #options['orth_orb_array'] = np.reshape(orth_orb_array, [-1,1]) 
#  #print("core_ind_array", options['core_ind_array'])  
#  #print("orth_orb_array", options['orth_orb_array'])  
#  #print("cusp_radii_mat", options['cusp_radii_mat'])  
#
#  #options['proj_mat'] = get_proj_mat(options, orth_orb_array)
#  #print("proj_mat", options['proj_mat'])  
#  #np.set_printoptions(threshold=sys.maxsize)
#  #print(options)
#  #tic_all = time.perf_counter()
#  #get_a0_matrix(options)
#  #toc_all = time.perf_counter()
#
#  #print()
#  #print("==========================================")
#  #print("|| time to calc a0_mat", toc_all-tic_all, " ||")
#  #print("==========================================")
#  #print()
#  ################################
