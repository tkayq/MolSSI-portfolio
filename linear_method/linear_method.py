import numpy as np
import pickle

### Script to perform orbital optimization via the LM of a VMC trail wavefunction  
#     (e.g. a single-Slater Jastrow wavefunction)
#
#   To run, a dictionary (internal_options_dict) is provided to the the main function:
#     do_linear_method() as well as some other parameters
#     following the LM step the orbital coefficient matrix internal_options["mocoeff"]
#     will be updated with the optimized coefficients
#
#   An example input dictionary for propene (6-31G basis) is provided in examples/
#   calling this function directly will produce an updated orbital coefficient matrix 

def cm_array_to_matrix_ind(ind, rows):
  """ convert column major vector index to 2D matrix index

  params:
     ind - array of vector indices (column major) 
    rows - scalar, # of rows 

  return:
      ij - [N,2] array of matrix row by column index
    
  """
  ind = np.atleast_1d(ind)
  i = ind % rows
  j = ind // rows
  ij = np.stack((i.reshape([-1]),j.reshape([-1])), axis=-1) # [N,2] array
  return ij 

def matrix_ind_to_cm_array(ind, rows, cols):
  """ convert 2D matrix index to column major vector index  

  params:
       ind - [MO, 2] array of matrix indices 
      rows - scalar, # of rows in matrix
      cols - scalar, # of columns in matrix

  return:
    cm_ind - [MO, ] vector indices of column major matrix 
    
  """
  cm_ind = rows * ind[:,1] + ind[:,0]  # total_rows * col + row 
  return cm_ind 

def fixed_MO_norm(C_mat, other_fixed_ind=[]):
  """ Remove largest AO from existing optimizable coefficients per MO
 
  params:
            C_mat - AO x MO mocoeff numpy array
          fix_ind - array of existing indices (column major) to freeze,

  return:
    all_fixed_ind - updated array of indices (column major) in mocoeff to freeze, including norm

  """
  rows, cols = C_mat.shape

  # isoate only the optimizable params to norm wrt
  if len(other_fixed_ind) > 0:
    fix_mat_ind = (cm_array_to_matrix_ind(other_fixed_ind, rows)).astype(int)

  # get ind of each coeff to norm wrt
  col_ind = np.arange(cols)  # index of each MO
  max_MO_ind = np.argmax(np.abs(C_mat), axis=0) 
  matrix_ind = np.stack((max_MO_ind, col_ind), axis=-1) # [MO, 2] array
  vec_ind_of_norm = matrix_ind_to_cm_array(matrix_ind, rows, cols) 

  # combine already fixed and norm indices 
  all_fixed_ind = np.sort(np.unique(np.concatenate((vec_ind_of_norm, other_fixed_ind), axis=None)))

  return all_fixed_ind

def delta_E_filter_C_mat_unsorted(C_mat, F, S, delta_E):
  """ filter C mat by check each AOs contribution by setting removing and calc E_MO change 

  delta_E = | (c^T.F.c)/cTSc - (c'^T.F.c')/cTSc|
            where c = full MO vec, 
                  c'= c_ij element in c set to 0
          if < thresh, let c_ij = 0
          else, keep c_ij unchanged (larger thresh, more filtering)

  params:
          C - mocoeff matrix, in orthogonalized and localized basis
          F - fock matrix (from pyscf), in orthogonalized and localized basis
          S - overlap matrix (from pyscf), in orthogonalized and localized basis
    delta_E - threshold to set mocoeff element to 0 if under

  return:
          C - energy filtered (per MO) mocoeff matrix

  """
  num_AO, num_MO = C_mat.shape
  C = C_mat.copy()

  # calculate Fock energies
  norm_S = (C.T @ S @ C)            # [MO,MO]
  fock_E = (C.T @ F @ C) / norm_S   # [MO,MO]

  zeros_per_mo = []
  for j in np.arange(num_MO):			# loop through num MO 
    mo_vec = C[:,j]
    E_0 = fock_E[j,j]                   # energy of MO from full mocoeff vec 
    #print("MO", j, " with energy", E_0, flush=True)

    set_zero_ind_list = []
    for i in np.arange(num_AO):
      mo_vec[i] = 0.0   # test setting ith element to zero
      E_new = mo_vec.T @ F @ mo_vec / (mo_vec.T @ S @ mo_vec)  # energy of MO with ith element set to 0
      if np.abs(E_new - E_0) < delta_E:  
        set_zero_ind_list.append(i)

      mo_vec[i] = C_mat[i,j]    # reset c element no matter result
    mo_vec[set_zero_ind_list] = 0. # updated C matrix getting returned
    zeros_per_mo.append(len(set_zero_ind_list))
    if len(set_zero_ind_list) == num_AO:
       raise RuntimeError("all coefficients in MO", j, "have been set to 0 in delta_E_filter_C_mat_unsorted with an epsilons of", delta_E)
    
  return C

def get_filtered_basis_ind(C, F_hf, S_hf, epsilon, fixed_ind_opt, rows, cols):
  """ Determine basis of params to opt after energy filtering. Required: do_constrained = selected_LCAO = True.
    
  params: 
                      C - AO x MO matrix of mocoeff from calc + full accepted delta_p update, before filtering
             F_hf, S_hf - AO x AO fock and overlap matrices from pyscf HF calc
                epsilon - energy (Ht) and MO is allowed to change by when an lcao coeff -> 0, 
                          if delta_E < epsilon, coeff will be set to 0
          fixed_ind_opt - original CM ind of row/cols to remove from opt corresponding to the full possible 
                          dp update in nvp basis (not LM nvp+1 yet)
                   rows - num AOs
                   cols - num MOs
      
  return:
    fixed_ind_acc_basis - CM indices of mocoeff set to 0 by energy filter of updated C matrix

  """

  C_filtered = delta_E_filter_C_mat_unsorted(C, F_hf, S_hf, epsilon) 

  zero_post = np.argwhere(C_filtered.T.reshape(-1) == 0.0).reshape(-1) # CM in [nvp,nvp] basis
  fixed_ind_nacc_basis = np.union1d(fixed_ind_opt, zero_post) # elements set at fixed union with elements 0 after filtering C+dp 

  return fixed_ind_nacc_basis 

def vec_within_deltaE_fock(C,thresh,F,S,fock_E):
  """
  params:
     thresh - threshold MO energy is allowed to change by
          C - C_0 + dp being tested, C_0 is the mocoeff mat to be updated
          F - fock matrix (from pyscf)
          S - overlap matrix (from pyscf)
     fock_E - diag(C_0.T @ F @ C_0 / C_0.T @ S @ C_0) 
              vector of MO energies before param update  

  return:
    accept_dp - bool, True if all MO energies change (relative to C_0) < thresh 
  """
  
  num_AO, num_MO = C.shape
  for j in np.arange(num_MO):			# loop through num MO 
    mo_vec = C[:,j]
    E_0 = fock_E[j]               # energy of MO from full mocoeff vec 

    E_new = mo_vec.T @ F @ mo_vec / (mo_vec.T @ S @ mo_vec)  # energy of MO with ith element set to 0

    E_diff = np.abs(E_new - E_0)
    print("\tMO",j,E_diff,flush=True)
  
    if E_diff > thresh: # MO energy cannot be raised by more than thresh
      print("\tMO", j, "has absolute E_diff of", E_diff, " > ", thresh, " --- check next eigenvector", flush=True)
      return False
  
  return True # All MO energies change < thresh

def param_vec_to_mat(param_vec, rows, cols, no_opt_ind=np.array([])):
  """ Convert vector of parameter updates to a matrix 

  params:
     param_vec - nvp length vector of parameters to update
          rows - number of rows of desired matrix
          cols - number of columns of desired matrix
    no_opt_ind - column major vector indices of fixed parameters 
                 in (nvp x nvp matrix) 

  return:
   delta_p_mat - AO x MO matrix of updated MO coefficients

  """
  delta_p_vec = np.zeros(rows*cols)
  opt_ind = np.setdiff1d(np.arange(rows*cols), no_opt_ind) # remove inds of fix params from all mo coeff inds

  delta_p_vec[opt_ind] = param_vec
  delta_p_mat = np.transpose(delta_p_vec.reshape(cols,rows))  # AO x MO column major matrix from array
  #print("\ndelta_p Matrix:\n", delta_p_mat, flush=True)

  return delta_p_mat 

def reweight_dp(S, delta_p, xi=0.5):
  """ further normalize params based on Toulouse and Umrigars 2008 paper
  params:
          S - LM overlap matrix in nvp basis
    delta_p - vector normalized wrt dp0 in nvp basis
         xi - 0.5 default recommended magnitude

  return:
    w_delta_p - [nvp,] param vec weighted by Toulouse scheme

  """
  one_minus_xi = 1. - xi

  delta_p = delta_p.reshape(1,-1)

  pS = delta_p.dot(S) # [1,nvp]
  #print("shape of pS should be 1xnvp\n", pS.shape)

  ppS = delta_p.dot(pS.reshape(-1,1)) # scalar 

  N = -( ( one_minus_xi * pS ) / ( one_minus_xi + xi * np.sqrt( 1 + ppS ) ) ) # [1,nvp]
  #print("N for weighting:", N.shape, N, )

  Np = np.sum(N * delta_p)  # scalar 
  w_delta_p = delta_p / ( 1. - Np ) # [1, nvp]

  return w_delta_p.reshape(-1), (1-Np)

def hamiltonian_shift(cI, H):
  """ Add level shift to diagonal Hamiltonian to stabilize optimization 

  params:
    cI - scalar, shift magnitude, penalizes deviations 
         from the energy (cI->inf, recover gradient descent)
     H - nvp+1 x nvp+1 Hamiltonian matrix
  """
  A_mat = np.diag(np.full(H.shape[0], cI))
  A_mat[0,0] = 0.0
  return H + A_mat 

def solve_and_sort_eigen_prob(H, S, cI):
  """ Return real unnormalized, sorted eigenvalues/vectors of shifted_Hp = ESp 

  params:
                       H - Hamiltonian LM matrix 
                       S - Overlap LM matrix
                      cI - diagonal shift on H to dampen step size

  return:
       sorted_real_evals - real LM eigenvalues in ascending order  
    sorted_real_evec_mat - correspinding unnormalized eigenvectors 
  
  """

  shifted_H = hamiltonian_shift(cI, H)
  S_inv = np.linalg.pinv(S, rcond=1e-6, hermitian=True)	# should this have reasonable values?
  S_invH = S_inv @ shifted_H

  evals, evec_mat = np.linalg.eig(S_invH)

  all_sorted_idx = (evals.real).argsort()	# indices that sort eval array in acsending order
  sorted_imag_ind = np.nonzero(evals[all_sorted_idx].imag)
  sorted_real_ind = np.delete(all_sorted_idx, sorted_imag_ind) 

  sorted_real_evals = evals[sorted_real_ind]
  sorted_real_evec_mat = evec_mat[:,sorted_real_ind]

  return sorted_real_evals, sorted_real_evec_mat  # [nvp+1,] [nvp+1, nvp+1]

def delta_p_dE_fock(H_lm, S_lm, F_hf, S_hf, rows, cols, LM_dict, fixed_param_ind):
  """ Solving generalized eigenvalue problem for vec that changes Fock MO energies below some threshold 

            H delta_p = E S delta_p
  params:
              H_lm - nopt+1 x nopt+1 linear method accumulated Hamiltonian matrix
              S_lm - nopt+1 x nopt+1 linear method accumulated overlap matrix
              F_hf - Fock matrix from pyscf HF calc
              S_hf - MO x MO orbital overlap matrix from pyscf HF calc 
   fixed_param_ind - in nvp basis, to convert from dp_vec to dp_mat # must correspond to the reduced dim of H and S_lm

  return:
    delta_p_mat - AO x MO matrix of normalized update parameters to add to C

  reassign:
    LM_dict['cI'] - final hamiltonian shift used in accepted dp

  """
  mag_max = 0.25                        # hardcoded upper limit to dp element
  curr_en = LM_dict["total_en"]         # scalar, VMC energy
  dE_thresh = LM_dict['max_delta_p']    # threshold MO energies are allowed to change by
  C = LM_dict["mocoeff"].copy()         # mocoeff to update for next iter

  norm_S_hf = (C.T @ S_hf @ C)          # [MO,MO]
  fock_E_before = np.diag((C.T @ F_hf @ C) / norm_S_hf)   # [MO,MO], fock MO eneries before param addition

  delta_p_return = False 
  while delta_p_return == False:
    print("-------------------------------------------------")
    print("Check eigenvectors: cI of", LM_dict["cI"], "and deltaE of", dE_thresh, flush=True)

    # Get possible parameter updates at given cI shift
    evals, evec_mat = solve_and_sort_eigen_prob(H_lm, S_lm, LM_dict["cI"])

    E_diff=np.real(np.round(evals-curr_en,6))
    print("")
    print("\tsorted and real evals - total_total_e:", E_diff[:10], flush=True)
    print("")

    possible_eval_ind = np.where(E_diff < 0.)[0]
    # Go through possible parameter updates as long as there is an eval < VMC energy
    if len(possible_eval_ind) > 0:
      # Check if any of these eigenvectors would not change and MO energy by some amount dE_thresh
      for ind in possible_eval_ind: # loop through indices of sorted eval/vec lower than VMC

        print("Eval sorted ind and E_diff",ind, E_diff[ind],flush=True)

        p_vec = np.real(evec_mat[1:,ind]/evec_mat[0,ind])  # normalized by first element 
        p_vec, w_denom = reweight_dp(S_lm[1:,1:], p_vec)   # weighted by cyrus's xi scheme
        print("\t\tdenominator from cyrus's parm weighting scheme:", w_denom,flush=True)
       
        if any(x > mag_max for x in np.abs(p_vec)) == False:  # if no param update element is > max_param, accept evec at delta_p` 

          delta_p_mat = param_vec_to_mat(p_vec, rows, cols, fixed_param_ind)

          C_plus_dp = C + delta_p_mat

          # if p_vec accepted delta_p_return == True
          delta_p_return = vec_within_deltaE_fock(C_plus_dp,dE_thresh,F_hf,S_hf,fock_E_before)
          if delta_p_return:
            if len(LM_dict["dp_mat"]) > 0:
              print("dp_mat in LM_dict --- type: ", type(LM_dict["dp_mat"]), "len: ", len(LM_dict["dp_mat"]), flush=True)

              already_calced = [np.allclose(delta_p_mat, dp_mat) for dp_mat in LM_dict["dp_mat"]]

              if True in already_calced:
                print("\t\t\tSkip this vector, chosen in previous diverged calculation", flush=True)
                delta_p_return = False
              else:
                print("\t\t\tEigenvector is unique and accepted as param_vec! cI =", LM_dict["cI"], "; dE_thresh =", dE_thresh, flush=True)
                return delta_p_mat 

            else:
              print("\t\t\tEigenvector accepted as param_vec! cI =", LM_dict["cI"], "; dE_thresh =", dE_thresh, flush=True)
              return delta_p_mat 

        else: # element in mat was > 0.25
          print("\t\tEigenvector contains element with mag >", mag_max," (hardcoded upper limit), reject without evaulating MO energy and move to next.",flush=True)

    # If nothing got accepted at this cI shift - increase and repeat
    print("\t\tRediaginalize with larger cI.", flush=True)
    LM_dict["cI"] *= 10

def reduced_basis_matrix(matrix, remove_ind,  nbat=1): 
  """ Get LM H and S matrices in basis of optimizable parameters (nopt <= nvp)

  params:
         matrix - list of [N,N] matrices to delete rows and columns of 
     remove_ind - [m,], indices of rows and columns to delete 

  return:
         matrix - list of [N-m,N-m] matrices in reduced basis 
  """

  # delete fixed rows and columns from matrix for fixed paramters
  reduced_mat = []
  for mat in matrix:
    mat = np.delete(np.delete(mat, remove_ind, 0), remove_ind, 1) 
    reduced_mat.append(mat)

  return reduced_mat 

def get_param_mat(LM_dict, fixed_ind, F_hf, S_hf, rows, cols):
  """
  params:
          LM_dict - dictionary of all relevant linear method pieces 
        fixed_ind - CM ind of row/cols to remove from opt in nvp basis (not LM nvp+1 yet)
       F_hf, S_hf - AO x AO fock and overlap matrices from pyscf HF calc
             rows - num AOs
             cols - num MOs

  return:
        param_mat - AO x MO dp update matrix to mocoeff

  reassign: 
    LM_dict['cI'] - final hamiltonian shift used in accepted dp

  """
  no_opt_LM_ind = fixed_ind + 1 if len(fixed_ind) > 0 else [] # now in nvp+1 basis

  # freeze certain orbs based on normalization or sLCAO algorithm
  Hmat, Smat = reduced_basis_matrix([LM_dict["Hmat"], LM_dict["Smat"]], no_opt_LM_ind)  # nvp+1 x nvp+1 dimensions
  print("\nDimensions of H and S mat is basis of nopt+1 params:", Hmat.shape, Smat.shape, "Tot opt: ", np.prod(Hmat.shape))

  # LM_dict[cI_key] gets reassigned
  param_mat = delta_p_dE_fock(Hmat, Smat, 
                              F_hf, S_hf, 
                              rows, cols, 
                              LM_dict,  
                              fixed_ind)  # nvp dimension
  return param_mat  # AO x MO

def linear_method_step(LM_dict, F_hf, S_hf):
  """ Performs linear method step for next iteration

  params:
                      LM_dict - dictionary of all relevant linear method pieces 
                   F_hf, S_hf - AO x AO fock and overlap matrices from pyscf HF calc

  return:
                    param_mat - AO x MO dp update matrix to mocoeff

  reassign: 
                LM_dict["cI"] - final hamiltonian shift used in accepted dp
  
  appends:
            LM_dict["dp_mat"] - appending latest param update and its corresponding fixed_ind

  adds:
   LM_dict["fixed_param_ind_acc"] - to basis of accepted params (larger fixed basis than nopt basis)
                                    (only added if "selecedLCAO"="constrained_opt"=True)

  """

  rows, cols = LM_dict["mocoeff"].shape
  num_fixed_og = LM_dict["fixed_param_ind"].size
  cI_init = LM_dict["cI"]   # min cI copied from internal_options

  # LM_dict["cI"] gets reassigned within this func by delta_p_dE_fock()
  param_mat = get_param_mat(LM_dict, 
                            LM_dict["fixed_param_ind"],   # copy of internal_options here
                            F_hf, S_hf, 
                            rows, cols)

  print("Max 3 magnitudes in accepted eigenvec (nopt): ", np.flip(np.sort(np.abs(param_mat.reshape(-1)))[-3:]),flush=True)

  # add newly calculated param mat to list of param mats used so far
  print("LM_dict[\"dp_mat\"] length: ", "\n", len(LM_dict["dp_mat"]), flush=True)
  LM_dict["dp_mat"].append(param_mat)

  if LM_dict['constrained_opt'] == True and LM_dict['selected_LCAO'] == True and LM_dict['epsilon'] != 0.0000:  # rediag unless unCon 
    print("")
    print("Resolve Hp=ESp in basis of energy filtered parameters", flush=True)

    updated_C = LM_dict["mocoeff"] + param_mat

    #  Filters C + dp, collects naccepted ind
    LM_dict["fixed_param_ind_acc"] = get_filtered_basis_ind(updated_C, 
                                                        F_hf, S_hf, 
                                                        LM_dict['epsilon'], LM_dict["fixed_param_ind"], 
                                                        rows, cols)  # update to the naccept basis

    # reset to recalc dp in basis of accepted parameters
    LM_dict["cI"] = cI_init 

    # LM_dict["cI"] gets reassigned within this func by delta_p_dE_fock()
    param_mat = get_param_mat(LM_dict, 
                              LM_dict["fixed_param_ind_acc"],   # correspond to nacc basis now
                              F_hf, S_hf, 
                              rows, cols)

    print("Max magnitudes in accepted eigenvec (nacc): ", np.sort(np.abs(param_mat.reshape(-1)))[-3:],flush=True)
    print("Fixed length of fixed_ind from nopt and nacc bases", num_fixed_og, LM_dict["fixed_param_ind"].size, flush=True)

  if param_mat.shape != LM_dict["mocoeff"].shape:
    raise RuntimeError("Calculated linear method update in linear_method_step() has incorrect dimensions! Is", param_mat.shape, "should be", LM_dict["mocoeff"].shape)

  return param_mat  # AO x MO update matrix


def set_param_ind(C, Z, basis_centers, nn, con_opt, zeros_fixed, selected_LCAO, opt_these_orbs, add_neighboors=True):
  """ Set up all paramters to freeze at next iteration, including normalization
  params: 
                  C - AO x MO mocoeff matrix to freeze params wrt 
                  Z - array of atom charges
      basis_centers - AO length array of scalars corresponding to nuceli AO is centered on
                 nn - list of nearest neighboors where each row ind correspond nuclei attached to the the ind nuclei  
            LM_dict - dictionary with relevant linear method pieces (LM_dict OR internal_options if setting up opt)
            con_opt - bool, constrained optimization?
        zeros_fixed - bool, if con_opt do not opt params = 0
      selected_LCAO - bool, if con_opt use selected LCAO algorithm
     opt_these_orbs - CM array of ind to only include in opt if con_opt, else empty 
     add_neighboors - False, opt bfs on atoms turned on
                      True, " " and each atoms nearest neighboor
                      only applies if selectedLCAO == True

  return:
    fixed_param_ind - CM indices of parameters to remove from opt in nvp basis, includes normalization

  """

  print("")
  print("Set parameters to be optimized next", flush=True)

  if con_opt is True:
    print("---> Constrained Optimization\n", flush=True)

    if zeros_fixed == True:
      print("--------> Fixing Zero Indices", flush=True)
      fixed_param_ind = np.argwhere(C.T.reshape(-1) == 0.0).astype(int) #zero_indices

    elif selected_LCAO == True:
      print("--------> Selection Algorithm", flush=True) 

      # update already resolved in the bases of accepted parameters, keep these as what to fix, now just add neighboors
      fixed_param_ind = select_fixed_ind(C, Z, basis_centers, nn, add_neighboors)

    else: # opt_these_orbs
      print("--------> Optimizing select AOs: ", 
            opt_these_orbs, flush=True) 

      det_nvp = np.prod(C.shape)
      no_opt_ind = np.setdiff1d(np.arange(det_nvp), opt_these_orbs).astype(int)  # cm_array ind of AO coeffs to freeze
      fixed_param_ind = no_opt_ind.astype(int) 
      print("params to fix", fixed_param_ind)

    # fix largest AO of existing params per MO
    fixed_param_ind = fixed_MO_norm(C, fixed_param_ind).astype(int) 

  else:
    # fixed_param_ind already contains each largest AO in
    print("---> Un-constrained Optimization", flush=True)
    fixed_param_ind = fixed_MO_norm(C).astype(int) # only fix largest AO per MO

  #print_certain_el_in_Cmat(LM_dict["mocoeff"], fixed_param_ind)
  print("-----------------------------------------------")
  return fixed_param_ind

def orth_lm_matrices(nvp, acc_dict, total_total_e):
  """ Build linear method matrices (no H shift) in orthogonalized basis from accumulators
        Eq's can be found in Tolouse and Umrigar, JCP 2008.
  params:
              nvp - number of variational parameters
         acc_dict - dictionary of accumulators to build matrices from
    total_total_e - scalar, total energy, H[0,0]
             nbat - num batches = 1 for all (TKQ) LM cases

  return:
   full_orth_Hmat - nvp+1 x nvp+1 orthogonalized Hamiltonian matrix
   full_orth_Smat - nvp+1 x nvp+1 orthogonalized overlap matrix
  """

  # initialize LM info
  full_orth_Smat = np.zeros((nvp+1, nvp+1))	
  full_orth_Hmat = np.zeros((nvp+1, nvp+1))	

  # result [nvp,]
  total_gradE = np.mean(acc_dict["AccumulatorGradE"], axis=0)
  total_PvP0 = np.mean(acc_dict["AccumulatorPvP0"], axis=0)
  total_PvP0El = np.mean(acc_dict["AccumulatorPvP0El"], axis=0)

  # Average each param over the num blocks, 
  #   matrix info np.mean([nblocks, nvp*nvp], axis=0).reshape(nvp,nvp) => [nvp, nvp]
  total_SxEmat = np.mean(acc_dict["AccumulatorSxEmat"], axis=0).reshape(nvp,nvp) 
  total_PvP0GradEmat = np.mean(acc_dict["AccumulatorPvP0GradEmat"], axis=0).reshape(nvp,nvp)
  total_orth_Smat = np.mean(acc_dict["AccumulatorOrthSmat"], axis=0).reshape(nvp,nvp)

  # [nvp, nvp]
  PvP0_PvP0El = np.outer(total_PvP0, total_PvP0El)

  # build full orthogonalized S and H matrices
  full_orth_Smat[1:,1:] = total_orth_Smat - np.outer(total_PvP0, total_PvP0) 
  full_orth_Smat[0,0] = 1.0 

  #print("- psi_i * (psi_j * E ) - psi_j * (psi_i * E ) \n", - PvP0_PvP0El - np.transpose(PvP0_PvP0El) 
  #    ,flush=True)
  full_orth_Hmat[1:,1:] = (total_SxEmat 
                          - PvP0_PvP0El 
                          - np.transpose(PvP0_PvP0El) 
                          + np.outer(total_PvP0, total_PvP0) * total_total_e
                          + total_PvP0GradEmat 
                          - np.outer(total_PvP0, total_gradE)
                          )
  full_orth_Hmat[0,0] = total_total_e 
  full_orth_Hmat[1:,0] = total_PvP0El - total_PvP0 * total_total_e  # g_L, column
  full_orth_Hmat[0,1:] = total_PvP0El - total_PvP0 * total_total_e + total_gradE # g_R, row

  return full_orth_Hmat, full_orth_Smat 

def make_LM_dict(options, a):
  """ Creates a dictionary of key linear method pieces per iteration 

  params:
    options - dictionary of wfn info that went into VMC calc
          a - LM iteration number 

  return
    LM_dict - dictionary of LM specific pieces
  """
  LM_dict = {

    # name of dictionary output will be saved to
    "name" : options["file_name"],
  
    # CM indices of mocoeff elements to remove from opt
    "fixed_param_ind" : options["fixed_param_ind"], 
    
    # iteration number of VMC LM optimization
    "iteration" : a,

    # magnitude of hamiltonian shfift - initial point
    "cI" : options["cI"],

    # Cmatrix used in calc to accumulate this LM info 
    "mocoeff" : options["mocoeff"],

    ## do opt with more than just MO normalization removed from calc
    "constrained_opt" : options["constrained_opt"],

    ## use selected-LCAO algorithm
    "selected_LCAO" : options["selected_LCAO"],

    # delta energy (Ht) filter to use if selection alg is turned on
    "epsilon" : options["epsilon"],

    # initialize list of param updates at this step to make sure same update isnt chosen if diverged
    "dp_mat" : [],

    # max allowable change to MO energy by any one LCAO coeff
    "max_delta_p": options["max_delta_p"],
  }

  return LM_dict

def divergence_check(a, vmc_step_count, total_total_e, prior_iter_E, prior_iter_std_err, internal_options):
  """ If diverged calc new mocoeff, change seed, and prepare internal_options for repeat VMC calc

  return:
    bool - True = repeat calc, else False = move onto optimization

  if True reassign:
         internal_options: "mocoeff", "fixed_param_ind", "seed", "cI_prev", "vmc_step_count_prev" #,"dE_thresh_prev"  
  """
  if a == 0:
    return False, 0   # move on to optimization step

  print()
  print("#################################")
  print("Check if the energy has diverged!")
  print("#################################")
  print()

  print("Divergence check:")
  print("\ttotal_total_e", total_total_e)
  print("\tprior_iter_E", prior_iter_E)
  print("\tprior_iter_std_err", prior_iter_std_err)
  print("\tEnergy difference", (total_total_e - prior_iter_E))

  print("Note: divergence step check is at 4 sigma, actual change:", (total_total_e - prior_iter_E)/prior_iter_std_err)
  if (total_total_e - prior_iter_E) > 4*prior_iter_std_err:
    # if step diverged, resolve dp from iter-1 at lower dE_thresh to get a new C and rerun calc
    vmc_step_count = vmc_step_count + 1

    print("")
    print("calculated energy:", np.round(total_total_e,4), "--- Diverged (count", vmc_step_count,
          "; iteration", a, "), VMC energy > 3*last iters std error, resolve previous C = C + dp with smaller dE_thresh", flush=True)
    print("prior_iter_E", prior_iter_E)
    print("prior_iter_std_err", prior_iter_std_err)
    #print("\tTxtal E (would have been): ", total_total_e, " +/- ", iter_E_std_err, flush=True) 

    # recalc dp(dE_thresh) from iter a-1, update mocoeff(a) = mocoeff(a-1) + dp and corresponding fixed_param_ind(a)
    LM_prior = internal_options["file_name"]+'_LM_dict_iter'+str(a-1)+'.pkl'
    with open(LM_prior, 'rb') as fp:
      LM_dict_prior = pickle.load(fp)

    LM_dict_prior['cI'] = internal_options["cI"]  

    # updates LM_dict_prior["cI"]
    param_update = linear_method_step(LM_dict_prior, 
                                      internal_options["fock_mat"], 
                                      internal_options["overlap_mat"]) # AO x MO update matrix

    internal_options["mocoeff"] = LM_dict_prior["mocoeff"] + param_update

    # determine indices of orbitals to not update with the LM for MO normalization OR sLCAO algorithm
    internal_options["fixed_param_ind"] = set_param_ind(internal_options["mocoeff"], 
                                                        internal_options["Z"], 
                                                        internal_options["basis_centers"], 
                                                        internal_options["nearest_neighboors"], 
                                                        internal_options["constrained_opt"],
                                                        internal_options["zeros_fixed"],
                                                        internal_options["selected_LCAO"],
                                                        internal_options["opt_these_orbs"],
                                                        add_neighboors=True) 

    # resave  LM_dict corresponding to the previous calc - reassigns "max_delta_p", "cI", "dp_mat"
    internal_options["seed"] -= 4516 

    with open(LM_prior, 'wb') as file_name:
      pickle.dump(LM_dict_prior, file_name)

    # for opt bookkeeping
    internal_options["cI_prev"] = LM_dict_prior["cI"] 
    internal_options["dE_thresh_prev"] = LM_dict_prior["max_delta_p"]
    internal_options["vmc_step_count_prev"] = vmc_step_count

    return True, vmc_step_count     # repeat calculation  
    
  else:
    print("\n---> Energy did NOT diverge, move on to LM update")
    return False, 0                 # move on to optimization step
  
def do_linear_method(internal_options, acc_dict, total_total_e, iter_E_std_err, a): 
  """ Determine next iterations mocoeff and fixed_param_ind based on accumulated VMC data from C++ 

  params:
    internal_options - dictionary of wfn info that went into VMC calc
            acc_dict - dictioanry of accumulator populated in VMC calc
       total_total_e - VMC energy
      iter_E_std_err - VMC standard error on energy
                   a - iteration number

  files getting saved if LM on (nvp>0):
    LM_dict..._itera.pkl - save file containing accumulated orth. H and S data from VMC calc if next step 
                           diverges and orbitals need to be recalced with larger damping factor (cI)

  reassigned dictionary items:
    internal_options: mocoeff, fixed_param_ind, cI_prev, dE_thresh_prev
             LM_dict: cI, fixed_param_ind (based on accepted param update)
  """
  print()
  print("#################################")
  print("Computing linear method update!!!")
  print("#################################")
  print()

  nvp = internal_options['nvp']  # number of elements in orbital coeff matrix if being optimized, else 0

  if nvp > 0:

    LM_dict = make_LM_dict(internal_options, a)    # assigns file_name, iteration, fixed_param_ind, cI, mocoeff, max_delta_p, constrained_opt, selected_LCAO, epsilon
    LM_dict["Hmat"], LM_dict["Smat"] = orth_lm_matrices(nvp, acc_dict, total_total_e) # full nvp+1 x n+1 dimensions
    LM_dict["total_en"] = total_total_e
    LM_dict["std_err_en"] = iter_E_std_err

    # updates LM_dict["cI"]
    param_update = linear_method_step(LM_dict, internal_options["fock_mat"], internal_options["overlap_mat"]) # AO x MO update matrix

    internal_options['mocoeff'] = internal_options['mocoeff'] + param_update

    # TESTING turning off sLCAO after 18 iters to fix the current test parameter set
    if a == 18: # allow last filtering to happen then freeze
      print("Note, turning off selection algorithim and freezing params after iter 19")
      internal_options["selected_LCAO"] = False # stops rediagonalization of LM update in the basis of accepted params in linear_method_step()
      internal_options["zeros_fixed"] = True # fixes existing 0's for rest of calc

    # determine indices of orbitals to not update with the LM for MO normalization OR sLCAO algorithm
    internal_options["fixed_param_ind"] = set_param_ind(internal_options["mocoeff"], 
                                                        internal_options["Z"], 
                                                        internal_options["basis_centers"], 
                                                        internal_options["nearest_neighboors"], 
                                                        internal_options["constrained_opt"],
                                                        internal_options["zeros_fixed"],
                                                        internal_options["selected_LCAO"],
                                                        internal_options["opt_these_orbs"],
                                                        add_neighboors=True) # if a>19 zeros_fixed, add_neighboors does not ge evaluated

    # parameters used to calculate the mocoeff being returned 
    internal_options["cI_prev"] = LM_dict["cI"] 
    internal_options["dE_thresh_prev"] = LM_dict['max_delta_p']

    ## Save the dictionary to a pickle file
    #file_path = internal_options['file_name']+'_LM_dict_iter'+str(a)+'.pkl'
    #with open(file_path, 'wb') as file_name:
    #  pickle.dump(LM_dict, file_name)

    print("\n==========================\n", 
          "Go on: C matrix updated",
          "\n==========================\n", flush=True)
  else:
    print("No Linear Method Optimization step")

if __name__ == "__main__":
  # Example: Linear method (LM) orbital optimization of propene -- single slater Jastrow wavefunction in a 6-31G basis
  #     (simply, a Hartree-Fock wavefunction multiplied by a symmetric 2-body correlation factor) 

  # INPUTS
  internal_options_pkl = './example/propene_STO3G_internal_options_dict_iter1.pkl' # dictionary of information for the current wavefunction
  acc_dict_pkl = './example/propene_STO3G_LM_acc_dict_iter1.pkl' # dictionary of information accumulated in the VMC sample needed to take the LM step
  vmc_step_count = 0          # ongoing count of how many time LM has diverged and had to be recalcuated
  prior_total_en = -116.77713 # VMC energy of prior LM step
  total_en = -116.79412       # VMC energy 
  prior_en_std_err = 0.003423 # VMC standard error of the energy of prior LM step
  en_std_err = 0.00302        # VMC standard error of the energy 
  a = 1                       # LM iteration number, second update step being calculated

  with open(internal_options_pkl, 'rb') as fp: 
    internal_options = pickle.load(fp)

  with open(acc_dict_pkl, 'rb') as fp: 
    acc_dict = pickle.load(fp)

  ## Step 1. after first LM step, test if the VMC energy diverged with respect to the previous step
  #    if diverged: recalculate previous LM update with larger dampening shift and prepare internal_options 
  #    for another VMC calculation
  divergence_check(a, vmc_step_count, total_en, prior_total_en, prior_en_std_err, internal_options)  

  ## Step 2. not having diverged, move on to calculate the LM update for the orbital 
  #     parameters using the LM terms accumulated in the VMC procedure
  do_linear_method(internal_options, acc_dict, total_en, en_std_err, a)

  ## Result: internal_options dictionary updated with new orbital coefficient and initialized for new VMC calcuation