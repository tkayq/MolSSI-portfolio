import numpy as np
import sys
import math
import time
#from scipy.linalg import ishermitian
import pickle
import os

np.set_printoptions(precision=12, linewidth=np.inf, threshold=np.inf) #  formatter={'float_kind':'{:0.6f}'.format},

# comment out for plotting script to work
from essqc.essqc_cpp import vmc_take_samples_detail, vmc_take_relative_samples_multipole, vmc_take_samples_detail_multipole, vmc_take_relative_samples, get_number_of_threads, get_accumulator_array_dimensions, vmc_take_sample_for_nuc_cusp_testing, vmc_take_sample_for_LM_grad_testing, vmc_take_sample_for_orb_deriv_testing
from essqc.tools import printmat

#sys.path.insert(1,'/home/trine/Desktop/essqc_vmc/tfes-1/essqc/src/essqc/vmc/')
#sys.path.insert(1,'/global/scratch/users/trinequady/containerized_tfes/essqc/src/essqc/vmc/')
#import cusp_orbitals 
from essqc import cusp_orbitals #import pyscf_result, r_from_xyz, spherical_to_cartesian, get_mol_xyz
from essqc import linear_method #import pyscf_result, r_from_xyz, spherical_to_cartesian, get_mol_xyz
from essqc import integrand_at_rtp #import pyscf_result, r_from_xyz, spherical_to_cartesian, get_mol_xyz
#from essqc.get_init_elec_pos import get_elec_pos_near_bond
from essqc.make_vmc_input import make_full_input

#def test_vmc(n):
#
#  nblk = 20
#  nspb = 84000
#  nsamp = nblk*nspb;
#  nelec = 4
#  samples = np.zeros([nsamp, nelec, 3]) # particle positions
#  les = np.zeros([nsamp]) # local energies
#
#  for jA in [ 0.1, ]:
#
#    a = np.random.random([n,n])
#    ainv = 1.0 * a
#    d = np.array([0.0])
#    vmc_basic_testing(jA, d, ainv, samples, les) # should put det and inverse of a into d, ainv
#    derror = np.abs( d[0] - np.linalg.det(a) )
#    print("", flush=True)
#    print("determinant value = %.4e" % d[0])
#    print("determinant error = %.4e" % derror)
#    res = a @ ainv
#    for i in range(n):
#      for j in range(n):
#        if i != j and res[i,j] < 0.0:
#          res[i,j] = -1.0 * res[i,j]
#    res = np.round(res, 13)
#    print("inverse     error = %.4e" % np.linalg.norm(np.eye(n) - res))
#    print("")
#    print(res)
#    print("")
#
#    print("average distance of an electron cartesian coord from zero is ", np.mean(np.abs(samples)))
#    print("")
#
#    rles = np.reshape(les, [nblk, nspb])
#    bes = [ np.mean(rles[i,:]) for i in range(nblk) ]
#    for b in bes:
#      print("block average was %20.12f" % b)
#    print("")
#    print("stdev of block averages was %20.12f" % np.std(bes))
#    print("")
#
#    print("            jA = %.4f" % jA)
#    print("average energy = %20.12f" % np.mean(les))
#    print("   uncertainty = %20.12f" % ( np.std(bes) / np.sqrt(1.0*nblk) ) )
#    print("", flush=True)
def get_elec_pos_on_bond(z_array, nuc_pos, bond_n): 
  print("IN get_elec_pos_on_bond", flush=True)
  total_elec = np.sum(z_array)
  if total_elec % 2 != 0:
    raise RuntimeError("Only closed even # electrons allowed")

  n_elec_half = int(total_elec/2)
  a_pos = np.zeros((n_elec_half,3))
  b_pos = np.zeros((n_elec_half,3))
  nuc_elec_count = [0] * len(z_array) #np.zeros_like(z_array)

  move_on = [0]
  add_to_a = True 
  curr_num_elec = 0
  
  place_next = 'alpha'
# place all elec on a bond
  for nuc_ind, Z in enumerate(nuc_pos):
      #nuc_ind = i - 1
      num_elec_on_nuc = int(z_array[nuc_ind])
      #print("\nNUC ", nuc_ind, " w/ ", num_elec_on_nuc, " ELECTRONS", 
      #        "all nn of nuc", nuc_ind, ":", bond_n[nuc_ind], flush=True)
      print(bond_n[nuc_ind],flush=True)
      for nn in bond_n[nuc_ind]:	# loop through bonding neighboors
           
          #print("curr_num_elec/2 Total placed do far:", curr_num_elec, "z of nn", z_array[nn], "curr num elec from nn", nuc_elec_count[nn],nn)
          if nuc_elec_count[nn] < z_array[nn] and nuc_ind < nn:  # if neighboor has a electrons to donate
            delta = nuc_pos[nn] - Z # delta between nuc and its neighboor
            norm_delta = delta / np.linalg.norm(delta)
            r = np.sqrt(np.sum(delta**2))

            a_pos[int(np.round(curr_num_elec,0)),:] = Z + (r/2) * norm_delta   # place up and down spin elec btwn bond
            b_pos[int(np.round(curr_num_elec,0)),:] = Z + (r/2) * norm_delta
            #print("\tBond between ----", nuc_ind, nn, "add alpha and beta",
            #"to row ", curr_num_elec)

            nuc_elec_count[nn] += 1
            nuc_elec_count[nuc_ind] += 1

            #print("\t\t# of electrons in each nuc now", nuc_elec_count, "next row ", curr_num_elec)
            curr_num_elec += 1                       # increment row of apos/bpos to populate

      remaining_e_on_nuc = num_elec_on_nuc - nuc_elec_count[nuc_ind]    # place on core
      #print("\tleft over elec to place --- ", remaining_e_on_nuc)

      if remaining_e_on_nuc > 1: #and remaining_e_on_nuc % 2 == 0: # and remaining_e_on_nuc % 2 == 0: 
        rows = int(remaining_e_on_nuc/2) if remaining_e_on_nuc % 2 == 0 else int((remaining_e_on_nuc)/2)
        #print("num of rows to add to core", rows)
        for i in np.arange(rows): # even
          a_pos[int(curr_num_elec),:] = Z #+ (r/2) * norm_delta   # place up and down spin elec btwn bond
          b_pos[int(curr_num_elec),:] = Z #+ (r/2) * norm_delta
          nuc_elec_count[nuc_ind] = nuc_elec_count[nuc_ind] + 2
          #print("\tCore elec on nuc ----", nuc_ind, "add alpha and beta",
          #        "to row ", curr_num_elec, "Elec on this nuc placed so far:", nuc_elec_count[nuc_ind])
          curr_num_elec = curr_num_elec + 1                       # increment row of apos/bpos to populate

        remaining_e_on_nuc = num_elec_on_nuc - nuc_elec_count[nuc_ind]    # place on core
        #print("remaining",remaining_e_on_nuc,  flush=True)
      #print("\tFinal odd left over elec to place", remaining_e_on_nuc)

  for nuc_ind, Z in enumerate(nuc_pos):
      #nuc_ind = i - 1
      num_elec_on_nuc = int(z_array[nuc_ind])
      #print("\nNUC ", nuc_ind, " w/ ",nuc_elec_count[nuc_ind],  "of", num_elec_on_nuc, "elec placed")
      remaining_e_on_nuc = num_elec_on_nuc - nuc_elec_count[nuc_ind]    # place on core

      if remaining_e_on_nuc == 1: # and remaining_e_on_nuc % 2 == 0: 
        #print("\tOdd core elec on nuc ----", nuc_ind, "add", place_next,
        #"to row ", curr_num_elec)
        if place_next == 'alpha':
          a_pos[int(curr_num_elec),:] = Z #+ (r/2) * norm_delta   # place up and down spin elec btwn bond
          place_next = 'beta'
          #print("\t\tadding single elec to apos, row ", curr_num_elec)
        elif place_next == 'beta':
          b_pos[int(curr_num_elec),:] = Z #+ (r/2) * norm_delta
          place_next = 'alpha'
          curr_num_elec = curr_num_elec + 1                       # increment row of apos/bpos to populate

        nuc_elec_count[nuc_ind] += 1
        #print("\t\t# of electrons in each nuc now", nuc_elec_count)

        #remaining_e_on_nuc = num_elec_on_nuc - nuc_elec_count[nuc_ind]
      print("Final num of electrons in each nuc now", nuc_elec_count)

  if np.array_equal(nuc_elec_count,z_array):
    print("Bond-centered Alpha positions in bohr:\n", a_pos)
    print("Bond-centered Beta positions in bohr:\n", b_pos)
    return a_pos, b_pos
  else:
    raise RuntimeError("did not place correct num electrons", nuc_elec_count, "should be", z_array)

def get_elec_pos_near_bond(z_array, nuc_pos, bond_n): 
  print("get elec pos near bond")
  n_elec = int(np.sum(z_array)/2)
  a_pos, b_pos = get_elec_pos_on_bond(z_array, nuc_pos, bond_n) 
  nuc_dist = 0.25 # Bohr

  a_rand = np.random.normal(loc=0.0, scale=nuc_dist, size=n_elec*3).reshape(n_elec,3)
  b_rand = np.random.normal(loc=0.0, scale=nuc_dist, size=n_elec*3).reshape(n_elec,3)
  #print("a random scatter added", a_rand)
  #print("b random scatter added", b_rand)
  a_pos = a_rand + a_pos
  b_pos = b_rand + b_pos 

  print("Perturbed Alpha positions near bond in bohr:\n", a_pos)
  print()
  print("Perturbed Beta positions near bond in bohr:\n", b_pos)
  print()
  #print(np.array2string(a_pos.transpose().flatten(), separator=', ', formatter={'float_kind': lambda x: "%.12f" % x}))
  #print(np.array2string(b_pos.transpose().flatten(), separator=', ', formatter={'float_kind': lambda x: "%.12f" % x}))
  return a_pos, b_pos


def calculate_multipole_for_batch(batch_j, internal_options, acc_dict):

  # get number of electrons that are active in this block
  n = internal_options["num_active"]

  # get number of samples
  ns = internal_options["nsamp_per_block"] * internal_options["nblock"]

  # get sum of electron positions
  rx = np.sum(acc_dict["AccumulatorPosXi"], axis=0)

  # get sum of electron position products
  rxx = np.sum(acc_dict["AccumulatorPosXiXj"], axis=0)

  # get sum of electron positions times charges (charge for each one is -1 / ( n_sample ) )
  X_1 = rx * ( -1.0 / ns )

  # get sum of electron position products times charges (charge for each one is -1 / ( n_sample ) )
  X_2 = rxx * ( -1.0 / ns )

  # use the average active electron position as the origin
  multipole_origin = rx / ( 1.0 * n * ns )

  # get the monopole
  mono = -1.0 * n

  # Build dipole moment ( X_1 - total_charge * origin )
  dip = X_1 - ( -1.0 * n ) * multipole_origin
  
  # Build quadrupole moment
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # !! DO WE HAVE OUR "FACTOR OF n" RIGHT???  !!
  # !! We think so....                        !!
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  quad = np.zeros([3,3])
  for k in range(3):
    for l in range(k,3):
      quad[k,l] = 3.0 * (   X_2[ 3*k + l - (k * (k+1)) // 2 ]
                          - X_1[k] * multipole_origin[l]
                          - X_1[l] * multipole_origin[k]
                          + multipole_origin[k] * multipole_origin[l] * ( -1.0 * n )
                        )
      if k == l:
        for kk in range(3):
          quad[k,l] += -1.0 * (   X_2[ 3*kk + kk - (kk * (kk+1)) // 2 ]
                                - 2.0 * X_1[kk] * multipole_origin[kk]
                                + multipole_origin[kk] * multipole_origin[kk] * ( -1.0 * n )
                              )
      else:
        quad[l,k] = quad[k,l]

  # store multipole origin for this batch
  internal_options["m_origin"][batch_j] = multipole_origin

  # store multipole moments for this batch
  internal_options["monopole"][0][batch_j] = mono
  for k in range(3):
    internal_options["dipole"][k][batch_j] = dip[k]
  for k in range(3):
    for l in range(3):
      internal_options["quad"][k+3*l][batch_j] = quad[k,l]


def process_positions(input_pos, optionsName, name_to_print, starting_str):
  """Converts an input into xyz positions"""

  #print("%spositions of the %s:" % ( starting_str, name_to_print ))
  #print("")

  if type(input_pos) == type('aaa'):
    pos = np.array([ float(x) for x in input_pos.replace(',', ' ').replace(';', ' ').split() ])
  elif type(input_pos) == type(np.array([1.0])):
    pos = 1.0 * input_pos.flatten()
  else:
    raise RuntimeError('%s should be a numpy array or a string with positions in xyz format, e.g. "0 0 0, 0 0 1, 1.5 0 2.2"' % optionsName)

  if len(pos) % 3 != 0:
    raise RuntimeError('%s should contain a number of numbers that is a multiple of three' % optionsName)

  #for i in range(len(pos)):
    #print(" %20.12f" % pos[i], end="")
    #if (i+1) % 3 == 0:
      #print("")
  #print("")

  return pos


def process_positions_in_options(options, optname, name_to_print, starting_str=""):
  optionsName = 'options["%s"]' % optname
  if optname not in options:
    raise RuntimeError('you need to specify the %s positions via %s' % ( name_to_print, optionsName ))
  return process_positions(options[optname], optionsName, name_to_print, starting_str)

def make_accumulator_data_array(name, options):
  """Creates an array that will be used by the accumulator associated with the given name"""
  #print("\n IN simple_vmc.py make_accumulator_data_array() for: ", name)
  dims = np.zeros([4])
  get_accumulator_array_dimensions(name, options, dims)
  #print("dims", dims, flush=True)
  #print("\n make_accumulator_data_array() for: ", name, "\n\tRESULT: \n", np.zeros( [ math.floor(x+0.1) for x in dims if math.floor(x+0.1) > 0 ] ))

  return np.zeros( [ math.floor(x+0.1) for x in dims if math.floor(x+0.1) > 0 ] )

def make_accumulator_dict(names, options, num_vp=None, num_block=None):
  """Creates a dictionary of accumulator names and the corresponding data arrays"""
  acc_dict = {}
  #print("\n******************************")
  #print("accumulator dictionary contents made IN simple_vmc.py make_accumulator_dict():")
  for name in names:
    #print(name, flush=True)
    if num_vp is not None and num_block is not None and name in ["AccumulatorOrthSmat", "AccumulatorSxEmat", "AccumulatorPvP0GradEmat", ]:
      acc_dict[name] = np.zeros((num_block, num_vp*num_vp)) 
    elif num_vp is not None and num_block is not None and name in ["AccumulatorGradE", "AccumulatorPvP0", "AccumulatorPvP0El"]: # "AccumulatorHPvP0", 
      acc_dict[name] = np.zeros((num_block, num_vp)) 
    elif name == "AccumulatorLocalEAll":
      acc_dict[name] = np.zeros((options["nblock"], options["nsamp_per_block"])) 
    else:
      acc_dict[name] = make_accumulator_data_array(name, options)
  #print("******************************\n")
  return acc_dict

def do_absolute_energy_old(nsamp, nbatch, batch_size, internal_options):

  # get number of nuclei
  nnuc = internal_options["nuclei"].size // 3

  # get number of alpha electrons
  nalp = internal_options["apos"].size // 3

  # initialize an array to hold the average energy associated with each nuclei in each batch
  nuc_e_per_batch = np.zeros([nbatch, nnuc])

  # fraction of batches complete
  batch_fraction = 0.0

  # take and process samples in batches
  for bi in range(nbatch):

    # prepare sampled positions array (on c++ side this will look like a  # of samples  by  3*(# of electrons)  matrix that is row major)
    samples = np.zeros([batch_size, 2*nalp, 3])

    # prepare local energy array (on c++ side this will look like a  # of samples  by  # of nuclei  matrix that is row major)
    lea = np.zeros([batch_size, nnuc])

    # take samples
    print("", flush=True, end="")
    vmc_take_samples_detail(samples, lea, internal_options, dict())
    print("", flush=True, end="")

    # record the average energy associated with each nucleus for this batch
    nuc_e_per_batch[bi,:] = np.mean(lea, axis=0)

    # use the final sample to start the next batch
    internal_options["apos"] = ( 1.0 * samples[-1,:nalp,:] ).flatten()
    internal_options["bpos"] = ( 1.0 * samples[-1,nalp:,:] ).flatten()

    # change the random number seed for each batch
    internal_options['seed'] = internal_options['seed'] + 1

    # once the first batch is done, we stop warming up
    internal_options['nwarmup'] = 0

    # print progress from time to time
    new_batch_fraction = ( bi + 1.0 ) / ( 1.0 * nbatch )
    if new_batch_fraction >= batch_fraction + 0.1:
      batch_fraction = new_batch_fraction
      print("fraction of batches completed = %.3f" % batch_fraction)

  # get the total energy for each batch
  tot_e_per_batch = np.sum(nuc_e_per_batch, axis=1)

  # get the overall average energy and associated uncertainty for each nuclei
  nuc_e = np.mean(nuc_e_per_batch, axis=0)
  nuc_u = np.std( nuc_e_per_batch, axis=0) / np.sqrt(1.0*nbatch)

  # get the average total energy over all batches
  tot_e = np.mean(tot_e_per_batch)
  tot_u = np.std( tot_e_per_batch) / np.sqrt(1.0*nbatch)

  # check that total energy is the sum of the energies assigned to each nucleus
  if np.abs( np.sum(nuc_e) - tot_e ) > 1.0e-12:
    raise RuntimeError('total energy did not match the sum of the energies assigned to each nucleus')

  # print the results
  print("")
  for i in range(nnuc):
    print("energy associated with nucleus %4i = %20.12f   +/-  %20.12f" % (i, nuc_e[i], nuc_u[i]))
  print("--------------------------------------------------------------------------------------")
  print("                       total energy = %20.12f   +/-  %20.12f" % (tot_e, tot_u))
  print("", flush=True)

def region_ratio_stats(num_matrix, vgr_vec, do_ratio_unc=True):
  avg_A = np.mean(num_matrix, axis=0)
  std_A = np.std(num_matrix, axis=0)
  avg_B = np.mean(vgr_vec)
  std_B = np.std(vgr_vec)
  sqrtN = np.sqrt( 1.0 * vgr_vec.size )
  ratio_avg = avg_A / avg_B
  if do_ratio_unc:
    cov_AB = np.mean(   ( num_matrix - np.reshape(avg_A, [1,-1]) )
                      * np.reshape(vgr_vec - avg_B, [-1,1]),
                      axis=0)
    std_A_over_B = np.sqrt(    ( avg_A / avg_B ) * ( avg_A / avg_B )
                            * (   ( std_A / avg_A ) * ( std_A / avg_A )
                                + ( std_B / avg_B ) * ( std_B / avg_B )
                                - 2.0 * cov_AB / ( avg_A * avg_B )
                              )
                          )
    ratio_unc = std_A_over_B / sqrtN
  else:
    ratio_unc = np.zeros_like(ratio_avg)
  vgr_avg = avg_B
  vgr_unc = std_B / sqrtN
  return ratio_avg, ratio_unc, vgr_avg, vgr_unc

def region_ratio_difference_stats(num_matrix_1, vgr_vec_1, num_matrix_2, vgr_vec_2, do_diff_unc=True, print_covAB=True):
  sqrtN = np.sqrt( 1.0 * vgr_vec_1.size )
  ratio_avg_1, ratio_unc_1, vgr_avg_1, vgr_unc_1 = region_ratio_stats(num_matrix_1, vgr_vec_1, do_diff_unc)
  ratio_avg_2, ratio_unc_2, vgr_avg_2, vgr_unc_2 = region_ratio_stats(num_matrix_2, vgr_vec_2, do_diff_unc)
  avg_A = ratio_avg_1
  avg_B = ratio_avg_2
  diff_avg = avg_A - avg_B
  if do_diff_unc:
    std_A = ratio_unc_1 * sqrtN
    std_B = ratio_unc_2 * sqrtN
    cov_AB = np.mean(   ( num_matrix_1 / vgr_avg_1 - np.reshape(avg_A, [1,-1]) )
                      * ( num_matrix_2 / vgr_avg_2 - np.reshape(avg_B, [1,-1]) ),
                      axis=0)
    if print_covAB:
      print("In A_minus_B, cov_AB is: ", end="")
      printmat(np.reshape(cov_AB, [1,-1]))
      print("")
    sqrt_arg1 = np.reshape(std_A * std_A + std_B * std_B, [-1])
    sqrt_arg2 = np.reshape(2.0 * cov_AB, [-1])
    #for i in range(sqrt_arg1.shape[0]):
    #  if sqrt_arg1[i] - sqrt_arg2[i] < 0.0:
    #    print("In region_ratio_difference_stats, sqrt argument is less than zero: %14.6e vs %14.6e" % (sqrt_arg1[i] - sqrt_arg2[i], sqrt_arg1[i]) )
    #  if np.abs(sqrt_arg1[i]-sqrt_arg2[i]) / sqrt_arg1[i] < 1.0e-10: # avoid nan on negative numbers that come from roundoff error
    #    sqrt_arg1[i] = 0.0
    #    sqrt_arg2[i] = 0.0
    std_A_minus_B = np.reshape( np.sqrt( sqrt_arg1 - sqrt_arg2 ), std_A.shape )
    diff_unc = std_A_minus_B / sqrtN
  else:
    diff_unc = np.zeros_like(diff_avg)
  return diff_avg, diff_unc

# a class for holding and doing statistics on the data for the value of a quantinty in each nuclear region across our sampling blocks
class RegionStatsProcessor:

  def __init__(self, name, vgr, num_region_data, do_unc=True):

    # save the name of this type of data
    self.name = name

    # save the number of sampling blocks (each block will have an average value of our quantity in each nuclear region)
    self.nb = num_region_data.shape[0]

    # save the number of nuclear regions
    self.nr = num_region_data.shape[1]

    # save the value guiding ratio averages (they are the same for all regions, so we have one per sampling block)
    self.vgr = np.reshape(vgr, [-1,1])

    # save the numerator data in a new array in which the first column is the sum of the region contributions
    self.num_data = np.concatenate( [ np.reshape(np.array([np.sum(num_region_data, axis=1)]), [-1,1]), num_region_data ], axis=1 ) 

    # whether we should do uncertainty estimation for this quantity
    self.du = do_unc

  def table_entry(self, val, unc):
    if self.du:
      return '  %12.6f +/- %12.6f' % (val, unc)
    return '  %12.6f     %12s' % (val, '')

  def table_header(self):
    retval = ''
    retval = retval + '  %14s' % ('name%4s' % '')
    retval = retval + '  %29s' % ('sum over all regions%4s' % '')
    for i in range(self.nr):
      retval = retval + '  %29s' % ('region %2i%8s' % (i,''))
    return retval

  def table_dashes(self):
    retval = ''
    retval = retval + '  --------------'
    retval = retval + '  -----------------------------'
    for i in range(self.nr):
      retval = retval + '  -----------------------------'
    return retval

  def table_stderr_E(self):
    ratio_avg, ratio_unc, vgr_avg, vgr_unc = region_ratio_stats(self.num_data, self.vgr, self.du)
    retval = self.table_entry(ratio_avg[0], ratio_unc[0])
    return retval

  def table_row(self):
    ratio_avg, ratio_unc, vgr_avg, vgr_unc = region_ratio_stats(self.num_data, self.vgr, self.du)
    retval = ''
    retval = retval + '  %14s' % (self.name)
    retval = retval + self.table_entry(ratio_avg[0], ratio_unc[0])
    for i in range(self.nr):
      retval = retval + self.table_entry(ratio_avg[i+1], ratio_unc[i+1])
    return retval

  def table_diff_row(self, other):
    diff_avgs, diff_uncs = region_ratio_difference_stats(self.num_data, self.vgr, other.num_data, other.vgr, self.du, False)
    retval = ''
    retval = retval + '  %14s' % (self.name)
    retval = retval + self.table_entry(diff_avgs[0], diff_uncs[0])
    for i in range(self.nr):
      retval = retval + self.table_entry(diff_avgs[i+1], diff_uncs[i+1])
    return retval


def do_relative_energy_old(nsamp, nbatch, batch_size, internal_options):

  # get number of nuclei
  nnuc = internal_options["nuclei"].size // 3

  # get number of alpha electrons
  nalp = internal_options["apos"].size // 3

  # get number of threads
  nthread = get_number_of_threads()
  print("")
  print("have %i threads for do_relative_energy" % nthread)
  print("")

  ## initialize arrays to hold the average numerator values for each nuclei in each batch
  #num_1_per_batch = np.zeros([nbatch, nnuc])
  #num_2_per_batch = np.zeros([nbatch, nnuc])

  ## initialize arrays to hold the average denominator values in each batch
  #den_1_per_batch = np.zeros([nbatch])
  #den_2_per_batch = np.zeros([nbatch])

  # initialize array to hold each batch's estimate of the energy differences
  ediff_per_batch = np.zeros([nbatch, nnuc])

  # Prepare an array to hold the electron positions for each thread.
  # To start we set this to the initial electron positions, but once we've taken
  # samples we just keep each thread's most recent sample here.
  elec_pos = np.zeros([nthread, 2, nalp, 3])
  for i in range(nthread):
    elec_pos[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
    elec_pos[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])

  # prepare dictionaries that holds the data arrays for the different accumulators
  acc_dict1 = make_accumulator_dict( ["AccumulatorVGR", "AccumulatorKE"], internal_options)
  acc_dict2 = make_accumulator_dict( ["AccumulatorVGR", "AccumulatorKE"], internal_options)

  # fraction of batches complete
  batch_fraction = 0.0

  # take and process samples in batches
  for bi in range(nbatch):

    # prepare arrays to hold value to guiding ratios for the two geometries
    vgr1 = np.zeros([batch_size])
    vgr2 = np.zeros([batch_size])

    # prepare arrays to hold each nuclei's local energy values for the two geometries (on c++ side these will look like  # of samples  by  # of nuclei  matrices that are row major)
    le1 = np.zeros([batch_size, nnuc])
    le2 = np.zeros([batch_size, nnuc])

    # run the sampling
    print("", flush=True, end="")
    raise RuntimeError('vmc_take_relative_samples no longer has this interface')
    vmc_take_relative_samples(internal_options, acc_dict1, acc_dict2, vgr1, vgr2, le1, le2, elec_pos)
    print("", flush=True, end="")

    ## record the averages for each nucleus for this batch
    #num_1_per_batch[bi,:] = np.mean( np.reshape(vgr1, [batch_size,1]) * le1, axis=0)
    #num_2_per_batch[bi,:] = np.mean( np.reshape(vgr2, [batch_size,1]) * le2, axis=0)
    #den_1_per_batch[bi] = np.mean(vgr1)
    #den_2_per_batch[bi] = np.mean(vgr2)

    # record the energy differences for this batch
    ediff_per_batch[bi,:] = 1.0 * (   1.0 * np.mean(np.reshape(vgr1, [batch_size,1]) * le1, axis=0) / np.mean(vgr1)
                                    - 1.0 * np.mean(np.reshape(vgr2, [batch_size,1]) * le2, axis=0) / np.mean(vgr2) )

    ## use the final sample to start the next batch
    #internal_options["apos"] = ( 1.0 * final_elec_pos[0,:,:] ).flatten()
    #internal_options["bpos"] = ( 1.0 * final_elec_pos[1,:,:] ).flatten()

    # change the random number seed for each batch
    internal_options['seed'] = internal_options['seed'] + 1

    # once the first batch is done, we stop warming up
    internal_options['nwarmup'] = 0

    # print progress from time to time
    new_batch_fraction = ( bi + 1.0 ) / ( 1.0 * nbatch )
    if new_batch_fraction >= batch_fraction + 0.1:
      batch_fraction = new_batch_fraction
      print("fraction of batches completed = %.3f" % batch_fraction)

  # get total energy differences for each batch
  tot_ediff_per_batch = np.sum(ediff_per_batch, axis=1)

  # print which atoms are in which groups
  if len(internal_options["atom_groups"]) > 0:
    print("")
  for i in range(len(internal_options["atom_groups"])):
    print("atoms in group %4i:" % i, end="")
    for j in internal_options["atom_groups"][i]:
      print(" %4i" % j, end="")
    print("")

  # get energy differences for specified groups of atoms
  group_diffs = []
  for group in internal_options["atom_groups"]:
    group_diffs.append(0.0)
    for g in group:
      group_diffs[-1] = group_diffs[-1] + ediff_per_batch[:,g]

  # make Q-Q plots to see how close to normally distributed the batch estimates look
  from scipy import stats
  import matplotlib.pyplot as plt
  for i in range(nnuc):
    plt.clf()
    fig = plt.figure()
    stats.probplot((1.0 * ediff_per_batch[:,i]).flatten(), plot=plt)
    fig.savefig("%s_nuc%i.pdf" % ( internal_options["qq_plot_name"], i ), bbox_inches='tight')
  for i in range(len(group_diffs)):
    plt.clf()
    fig = plt.figure()
    stats.probplot(group_diffs[i], plot=plt)
    fig.savefig("%s_grp%i.pdf" % ( internal_options["qq_plot_name"], i ), bbox_inches='tight')
  plt.clf()
  fig = plt.figure()
  stats.probplot(tot_ediff_per_batch, plot=plt)
  fig.savefig("%s_total.pdf" % internal_options["qq_plot_name"], bbox_inches='tight')

  # get the average energy differences and associated uncertainties
  ediff_avgs = np.mean(ediff_per_batch, axis=0)
  ediff_uncs = np.std( ediff_per_batch, axis=0) / np.sqrt(1.0*nbatch)

  # get the average group energy difference and associated uncertainties
  grp_ediff_avg = [ np.mean(gdiff) for gdiff in group_diffs ]
  grp_ediff_unc = [ np.std( gdiff) / np.sqrt(1.0*nbatch) for gdiff in group_diffs ]

  # get the average total energy difference and associated uncertainty
  tot_ediff_avg = np.mean(tot_ediff_per_batch)
  tot_ediff_unc = np.std( tot_ediff_per_batch) / np.sqrt(1.0*nbatch)

  ## print kinetic energy contribution from each nuclear region
  #print("")
  #print("for each sampling block, the average kinetic energy contribution for each nuclear region")
  #for i in range(acc_dict["AccumulatorKE"].shape[1]):
  #  for j in range(acc_dict["AccumulatorKE"].shape[0]):
  #    print("%16.8f" % acc_dict["AccumulatorKE"][i,j], end="")
  #  print("")

  # print the results
  print("")
  print("Printing differences in energies as  (energy before transformation) - (energy after transformation)")
  print("")
  for i in range(nnuc):
    print("energy difference for nucleus %4i = %20.12f   +/-  %20.12f" % (i, ediff_avgs[i], ediff_uncs[i]))
  if len(internal_options["atom_groups"]) > 0:
    print("")
  for i in range(len(group_diffs)):
    print("energy difference for group   %4i = %20.12f   +/-  %20.12f" % (i, grp_ediff_avg[i], grp_ediff_unc[i]))
  print("")
  #print("-------------------------------------------------------------------------------------------------")
  #print("")
  print("           total energy difference = %20.12f   +/-  %20.12f" % (tot_ediff_avg, tot_ediff_unc))
  print("", flush=True)

def do_test(internal_options):

  vmc_take_samples_detail(internal_options, acc_dict, elec_pos)

def calc_stats_v1(El_b, El_2_b, nb, print_res=False):
  """ calc statistics of average value (Local E)
        var_of_avg = var_blocked_avg / number_of_blocks
        which will plateau with sufficient block size
        (ie each block is an iid sample)

  params:
     El_b - [nb, ] array local energy block average 
   El_2_b - [nb, ] array local energy^2 block average 
       nb - number of bins

  return:
         var - variance of the averaged value 
   error_bar - error bar of the averaged value  
  """
  #print("\t# blocks in calc_stats_v1", nb, flush=True) #, "scaling factor, sqrt( 1/(Nb-1) ) = ", np.sqrt(1 / (nb-1)), flush=True)

  if type(El_b) != type(np.array([])) or type(El_2_b) != type(np.array([])):
    raise RuntimeError("input to calc_stats_v1 must be an array, instead ", type(El_b))
  elif len(El_b) != nb and len(El_2_b) != nb:
    raise RuntimeError("input to calc_stats_v1 must both be", nb, "long, instead ", len(El_b), len(El_2_b))

  El_b_2_val = np.sum(El_b**2)   
  #print("\tglobal average", El_b_2_val)
  El_2_b_val = np.sum(El_2_b)
  #print("\taverage of squared energy ", El_2_b_val)

  var_of_b = (El_2_b_val - El_b_2_val)
  var_of_mean = var_of_b / (nb-1)
  error_bar = np.sqrt(var_of_mean)

  if print_res:
     print("\n\t\tvariance of blocks from: ", var_of_b)
     print("\n\t\tvariance of mean from: ", var_of_mean)
     print("\n\t\terror bar of avg value np.sqrt( var ) : ", error_bar)

  return var_of_b, var_of_mean, error_bar

def get_all_stats(acc_loc_E_all, acc_loc_E, acc_loc_E_sq, spb, binning=False, print_out=True):
  """ calc statistics for only 1 batch case

  params:
   acc_loc_E_all - [nb,spb] array of local energy accumulator
       acc_loc_E - [nb,] array of local energy accumulator
    acc_loc_E_sq - [nb,] array of local energy squared accumulator
             spb - samples per block

  return:
        num_bin, bin_len, variance_of_blocks, 
        variance_of_mean, error_bar, 
        auto_corr_of_blocks, auto_corr_of_mean - scalars or list depending on binning bool
  """
  if type(acc_loc_E) != type(np.array([])):
    raise RuntimeError("input to get_all_stats must be an array, instead ", type(acc_loc_E))
  elif acc_loc_E_all != []:
    if acc_loc_E_all.shape[1] != spb:
      raise RuntimeError("acc_loc_E_all to get_all_stats must be [nb,spb], instead ", acc_loc_E_all.shape)

  #print()
  
  total_avg_E = np.mean(acc_loc_E)
  nb = len(acc_loc_E)

  # further bin blocks to see of the errorbar is not changing
  if binning == True:
    num_bin_list = []
    bin_len = []

    binning_vars_b = []
    binning_vars_mean = []
    binning_errors = []
    auto_corr_b = []
    auto_corr_mean = []

    #print("Stats on accumulated data AND further binning analysis", flush=True)
    count = 0
    if acc_loc_E_all != []: # all samples available
      print("USING AccumLocalEAll")
      acc_loc_E = acc_loc_E_all.flatten()
      acc_loc_E_sq = acc_loc_E**2
      print("for variance of seperate calcs: square then average all samp in calc:", np.mean(acc_loc_E_sq,axis=-1), flush=True)
      print("for variance of seperate calcs: square then average the nblock accum array from calc:", np.mean(np.mean(acc_loc_E,axis=-1)**2), flush=True)

    else: # only accumulator
      print("USING AccumLocalE and AccumLocalESquared")

    num_bins = nb = acc_loc_E.shape[0]       # number of blocks in provided Accumulator

    while num_bins <= nb and num_bins > 30:
      #print("\n\nnum_blocks", num_bins, flush=True)

      split_dat = np.array_split(acc_loc_E, num_bins)   # len num_block list containing subarrays 
      split_dat_sq = np.array_split(acc_loc_E_sq, num_bins)   # len num_block list containing subarrays 
      num_samp_per_block = len(split_dat[0])
      
      #print(len(split_dat), num_samp_per_block, flush=True)

      bin_len.append(num_samp_per_block)
      num_bin_list.append(num_bins)
      #print()
      #print("order of 2^l, l = ", count, "\tnum blocks", num_bins, "\tnum samp per block", num_samp_per_block)

      avg_E_per_block = np.array([np.mean(block_dat) for block_dat in split_dat]).reshape(-1)
      avg_E_sq_per_block = np.array([np.mean(block_dat_2) for block_dat_2 in split_dat_sq]).reshape(-1)
      #tc_b, tc_mean = calc_autocorrelation(avg_E_per_block)  
      #auto_corr_b.append(tc_b)
      #auto_corr_mean.append(tc_mean)

      #print("Before calc_stats_v1", avg_E_per_block.shape, flush=True)
      var_b, var_mean, error_bar_b = calc_stats_v1(avg_E_per_block, avg_E_sq_per_block, num_bins, print_res=False) 
      binning_vars_b.append(var_b)
      binning_vars_mean.append(var_mean)
      binning_errors.append(error_bar_b)

      count+=1
      num_bins = np.floor_divide(nb, 2**count)   

    if print_out:
      print("Average total energy \t", np.mean(avg_E_per_block))
      print()
      print("number of blocks \t", num_bin_list)
      print("samples per block \t", bin_len)
      print()
      print("variances of blocks \t", binning_vars_b)
      print("variances of mean \t", binning_vars_mean)
      print()
      print("error bars\t", binning_errors)
      print()
      print("autocorrelation of blocks ", auto_corr_b)
      print("autocorrelation of mean ", auto_corr_mean)
      print()
    
    return num_bin_list, bin_len, binning_vars_b, binning_vars_mean, binning_errors, auto_corr_b, auto_corr_mean

  else: # just calc stats on the accumulator
    print("Stats on accumulated data only calc_stats_v1")
    var_b, var_mean, error_bar = calc_stats_v1(acc_loc_E, acc_loc_E_sq, nb, print_res=True) 
    tc_b = tc_mean = 0.0
    #tc_b, tc_mean = calc_autocorrelation(acc_loc_E)  

    if print_out:
      print("Average total energy \t", np.mean(acc_loc_E))
      print()
      print("number of blocks \t", nb)
      print("samples per block \t", spb)
      print()
      print("variances of blocks \t", var_b)
      print("variances of mean \t", var_mean)
      print()
      print("error bars\t", error_bar)
      print()
      print("autocorrelation of blocks ", tc_b)
      print("autocorrelation of mean ", tc_mean)
      print()

    return nb, spb, var_b, var_mean, error_bar, tc_b, tc_mean


def get_and_print_vmc_results(internal_options, acc_dict, a):
  # a - iteration

  total_total_e = 0
  total_total_e_squared = 0

  total_e = 0
  total_e_squared = 0
  total_e_ass = 0
  total_ass = 0
  total_e_aos = 0
  total_aos = 0

  # save for plotting script inputs
  if a == 0:  # save accumulator after first iteration
      np.savetxt(internal_options["file_name"]+'_AccumLocalE.txt', acc_dict["AccumulatorLocalE"])
      if "AccumulatorLocalEAll" in acc_dict:
        np.savetxt(internal_options["file_name"]+'_AccumLocalEAll.txt', acc_dict["AccumulatorLocalEAll"])
  
  print("Length of LocalE and LocalESquared accum should == number of blocks", internal_options["nblock"], flush=True)
  print(acc_dict["AccumulatorLocalE"].shape, flush=True)
  print(acc_dict["AccumulatorLocalESquared"].shape, flush=True)
  #print("LocalEAll accum should be nblocks x nsamp/block: ", acc_dict["AcumulatorLocalEAll"].shape, flush=True)
  
  # collect terms for steepest descent pieces
  total_e += np.mean(acc_dict["AccumulatorLocalE"], axis=0)
  total_e_squared += np.mean(acc_dict["AccumulatorLocalESquared"], axis=0)
  total_e_ass += np.mean(acc_dict["AccumulatorLocalEAss"])
  total_ass += np.mean(acc_dict["AccumulatorAss"])
  total_e_aos += np.mean(acc_dict["AccumulatorLocalEAos"])
  total_aos += np.mean(acc_dict["AccumulatorAos"])
  
  #print("Variance of the local energy accumulator!!!", np.var(acc_dict["AccumulatorLocalE"]), " of shape", acc_dict["AccumulatorLocalE"].shape, flush=True)
  
  # process the data we accumulated during sampling and print the results
  stats_processors = []
  for ad in [ acc_dict ]:
    acc_data_total_e = ad["AccumulatorKE"] + ad["AccumulatorEEE"] + ad["AccumulatorENE"] + ad["AccumulatorNNE"]
    stats_processors.append( [
      RegionStatsProcessor(     'kinetic', ad['AccumulatorVGR'], ad[ "AccumulatorKE"],  True),
      RegionStatsProcessor(   'elec-elec', ad['AccumulatorVGR'], ad["AccumulatorEEE"],  True),
      RegionStatsProcessor(    'elec-nuc', ad['AccumulatorVGR'], ad["AccumulatorENE"],  True),
      RegionStatsProcessor(     'nuc-nuc', ad['AccumulatorVGR'], ad["AccumulatorNNE"], False),
      RegionStatsProcessor('total energy', ad['AccumulatorVGR'],     acc_data_total_e,  True),
    ] ) 
 
    # NO BATCHING
    #print("")
    #print("absolute quantities for batch", j)
    #print("")
  
    print(stats_processors[-1][0].table_header())
    print(stats_processors[-1][0].table_dashes())
    for rsp in stats_processors[-1]:
      print(rsp.table_row())
  
  iter_E_std_err = float((stats_processors[-1][-1].table_stderr_E()).rsplit(' ', 1)[-1])
  # = float((stats_processors[-1][-1].table_stderr_E()).rsplit(' ', 1)[-1])
  total_total_e += total_e	# sum up all batch energies
  total_total_e_squared += total_e_squared	# sum up all batch energies
  
  
  #print("iter_E_std_err = ", internal_options["iter_E_std_err"])
  #print("\tTotal e: ", total_total_e)
  #print("Total total e: ", total_total_e, "\tTotal e: ", total_e)
  
  # for steepest descent
  #delta_Ass += 0.0 #0.1 * (2 * total_e_ass - 2 * total_e * total_ass)
  #delta_Aos += 0.0 #0.1 * (2 * total_e_aos - 2 * total_e * total_aos)

  return total_total_e, iter_E_std_err

## work here
def do_absolute_energy(internal_options, iteration_start=0,savefiles=True):
  """
  start of VMC calculation, all c++ info in 'internal_options'
  outerloop through optimization iterations, and through batches
  set iteration_start to iter you want to pick back up 
  """

  internal_options['iter_E'] = 0.0 # initialize iteration energy
  internal_options['iter_E_std_err'] = 0.0 # initialize iteration energy
  #print()
  print("internal_options dictionary in do_absolute_energy:", internal_options)
  #print()

  # get number of nuclei
  nnuc = internal_options["nuclei"].size // 3

  # get number of alpha electrons
  nalp = internal_options["apos"].size // 3

  # get number of threads
  nthread = get_number_of_threads()

  # set number of parameters and indices, default nvp = 0, 
  # default nvp = 0, 
  nvp = internal_options['nvp'] 

  nbat = len(internal_options["batches"])

  num_iter = internal_options['iter']		# if no optimization, evaluate only 1 iteration by default

  print("")
  print("have %i threads for do_absolute_energy" % nthread)
  print("")

  print("Initial xyz\n", np.reshape(internal_options["nuclei"], [nnuc,3]))
  print()

  if savefiles: append_text_to_file(internal_options['file_name']+'.dat', "VMC_energy\tVMC_std_error\tpercent_non0\tcI_val\tdE_thresh\trecalc_count") 
  internal_options["vmc_step_count_prev"] = internal_options["cI_prev"] = internal_options["dE_thresh_prev"] = "n/a"  # default value since first energy had no LM params

  prior_iter_E = 0. 
  prior_iter_std_err = 0. 

  ######################
  # BEGIN CALCULATION  #
  ######################
  print("----------------------------------")
  for a in range(iteration_start, num_iter):
    print("Iter. #:", a, " of ", num_iter-1, flush=True)
    print("----------------------------------", flush=True)

    if a > 0: # to compare against if need to recalc step
      prior_iter_E = internal_options['iter_E'] 
      prior_iter_std_err = internal_options["iter_E_std_err"]
      internal_options["vmc_step_count_prev"] = 0

    vmc_step_count = 0

    vmc_step = True # flag that repeats vmc calc until energy is accepted (based on divergence criteria), then vmc_step == False
    while vmc_step: # and vmc_step_count < 3: 

      print("vmc_step_count", vmc_step_count, flush=True)
      #if vmc_step_count > 3:
      #  raise RuntimeError("This iteration has diverged too many time, check on calc")

      # save mocoeff to *.mat for plotting 
      if savefiles: np.savetxt(internal_options["file_name"]+'_iter'+str(a)+'.mat', internal_options["mocoeff"])

      per_C = linear_method.print_percent_of_mat_greater(internal_options["mocoeff"], 0., print_val=False)
      print("Plot this - % non-zero_elements in C mat going into calc", per_C, "%") 

      print("C matrix: \n", internal_options["mocoeff"], flush=True)
      print("Ass: ", internal_options["jAss"], flush=True)
      print("Aos: ", internal_options["jAos"], flush=True)

      print("Seed: ", internal_options["seed"], flush=True)    
      print()

      tic = time.perf_counter()

      delta_Ass = 0
      delta_Aos = 0

      acc_names = ["AccumulatorVGR", "AccumulatorKE", "AccumulatorEEE", "AccumulatorENE", "AccumulatorNNE", "AccumulatorLocalE", "AccumulatorLocalESquared", "AccumulatorAss", "AccumulatorLocalEAss", "AccumulatorAos", "AccumulatorLocalEAos"] #, "AccumulatorLocalEAll", ]
      # prepare dictionary that holds the data arrays for the different accumulators    
      if nvp > 0:
         #zero_indices = internal_options['fixed_param_ind']
         #print("Fix max AO per MO, indices to remove from opt", internal_options["fixed_param_ind"])
         acc_names = acc_names+["AccumulatorPvP0", "AccumulatorGradE", "AccumulatorOrthSmat", "AccumulatorSxEmat", "AccumulatorPvP0El", "AccumulatorPvP0GradEmat",]
         #print(acc_names,flush=True)
         acc_dict = make_accumulator_dict(acc_names, internal_options, nvp, internal_options["nblock"])
      else:
         acc_dict = make_accumulator_dict(acc_names, internal_options)

      for j in range(nbat):
        #print("\nBatch #:", j+1, flush=True)
        internal_options["active_batch"] = j

      # Prepare an array to hold the electron positions for each thread.
      # To start we set this to the initial electron positions, but once we've taken
      # samples we just keep each thread's most recent sample here.
        elec_pos = np.zeros([nthread, 2, nalp, 3])
      # hack for thread test to get set elec pos to each block --- see git history for this 08082024 #
        for i in range(nthread):
          elec_pos[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
          elec_pos[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])

        # take the sample and accumulate the data we need
        tic_cpp = time.perf_counter()
        vmc_take_samples_detail(internal_options, acc_dict, elec_pos)
        toc_cpp = time.perf_counter()

        total_total_e, iter_E_std_err = get_and_print_vmc_results(internal_options, acc_dict, a)
        # end of batches loop

      toc = time.perf_counter()
      print("========================")
      print("TIME of energy calc(min):", (toc-tic)/60.0)
      print("\ttime of cpp energy calc(min):", (toc_cpp-tic_cpp)/60.0)
      print("========================")

      ###################################
      ####      DIVERGENCE CHECK     ####
      ################################### 
      vmc_step, vmc_step_count = linear_method.divergence_check(a, vmc_step_count, total_total_e, prior_iter_E, prior_iter_std_err, internal_options)
    
    # Calculation Complete! 
    print("\nTotal E: ", total_total_e, " +/- ", iter_E_std_err)
    print("\nplot_opt_and_percent ", total_total_e, iter_E_std_err, per_C)

    #print("\nCusp test - save pkl if it contains a bad cusp (diff by > 0.001):")
    #do_1e_cusp_test(internal_options, mo_list=np.arange(0,internal_options["mocoeff"].shape[-1]), acc_names=["AccumulatorCuspTestingLE", "AccumulatorCuspTestingPsiSquared", "AccumulatorCuspTestingKE", ], which_orb_all=['cusp', ],savefiles=False,printerronly=True, name=internal_options["file_name"]+'_iter'+str(a))

    if savefiles:
      print("Now saving internal_options dictionary for iter", a, flush=True)
      # Save the dictionary to a pickle file
      file_path = internal_options['file_name']+'_internal_options_dict_iter'+str(a)+'.pkl'
      with open(file_path, 'wb') as file_name:
        pickle.dump(internal_options, file_name)

      # delete dictionary pickle from a-2
      old_int_dict = internal_options['file_name']+'_internal_options_dict_iter'+str(a-2)+'.pkl'
      if a>1 and os.path.exists(old_int_dict): 
        os.remove(old_int_dict)

      old_int_LM_dict = internal_options['file_name']+'_LM_dict_iter'+str(a-2)+'.pkl'
      if a>1 and os.path.exists(old_int_LM_dict): 
        os.remove(old_int_LM_dict)

      print("vmc_step_count_prev", internal_options["vmc_step_count_prev"], flush=True)
      # save info used to calc latest energy
      append_text_to_file(internal_options['file_name']+'.dat', 
            str(total_total_e)+"\t"+str(iter_E_std_err)+"\t"+str(per_C)+"\t"+str(internal_options["cI_prev"])+"\t"+str(internal_options["dE_thresh_prev"])+"\t"+str(internal_options["vmc_step_count_prev"])) 
    print("Info from calc: ",
          str(total_total_e)+"\t"+str(iter_E_std_err)+"\t"+str(per_C)+"\t"+str(internal_options["cI_prev"])+"\t"+str(internal_options["dE_thresh_prev"])+"\t"+str(internal_options["vmc_step_count_prev"]), flush=True)
    ###################################
    ####      LM OPTIMIZATION      ####
    ################################### 
    linear_method.do_linear_method(nvp, internal_options, acc_dict, total_total_e, iter_E_std_err, a, nbat=1)

    #num_vp_on.append(len(np.nonzero(internal_options["mocoeff"])))
    internal_options["iter_E_std_err"] = iter_E_std_err
    internal_options['iter_E'] = total_total_e

    internal_options["seed"] += 12345
    ##internal_options["jAss"] -= delta_Ass
    ##internal_options["jAos"] += delta_Aos
    print("\n----------------------------------\n", flush=True)
  print("CALCULATION COMPLETED")

def append_text_to_file(file_path, text_to_append): 
  try: 
    with open(file_path, 'a') as file: 
      file.write(text_to_append + '\n') 
    print(f"Text appended to {file_path} successfully.") 
  except Exception as e: 
    print(f"Error: {e}") 

def check_symmetry(H, nvp):
  """ Check symmetry of some matrix H 

      prints difference between elements in upper right triangle
  """
  diff_mat = H - H.T

  print("\nMatrix symmetry: \n",  np.array2string(diff_mat, threshold=sys.maxsize, max_line_width=sys.maxsize, separator=', ', formatter={'float_kind': lambda x: "%.3f" % x}))

def do_cusp_comp_test(options, n_list):
  """ calculate cusp coefficients of both cusp types to visualize difference """
  print("in do_cusp_comp_test", flush=True) 
  all_cusp_options = options.copy()
  
  if all_cusp_options["cusp_type"] == "slater_poly":
    print("CALC SLATER_POLY_PLUS_RATIO", flush=True) 
    # need to calc plus_ratio
    slater_poly_cusp_coeff_matrix = all_cusp_options["cusp_coeff_matrix"]

    all_cusp_options["cusp_type"] = "slater_poly_plus_ratio"
    slater_poly_plus_ratio_cusp_coeff_matrix = cusp_orbitals.get_cusp_coeff_matrix(all_cusp_options, stretched=False)

  elif all_cusp_options["cusp_type"] == "slater_poly_plus_ratio":
    print("CALC SLATER_POLY", flush=True) 
    slater_poly_plus_ratio_cusp_coeff_matrix = all_cusp_options["cusp_coeff_matrix"]

    all_cusp_options["cusp_type"] = "slater_poly"
    slater_poly_cusp_coeff_matrix = cusp_orbitals.get_cusp_coeff_matrix(all_cusp_options, stretched=False)

  return slater_poly_cusp_coeff_matrix, slater_poly_plus_ratio_cusp_coeff_matrix


def do_thru_nuc_test(options, acc_names=["AccumulatorCuspTestingLEAllE", "AccumulatorCuspTestingKE", "AccumulatorCuspTestingPsiSquared"], which_orb_all={"cusp","gaussian"}): #,"slater"}):
  """ which_orb takes in a set of string to eval the KE and/or local_E depending on input to acc_names
        slater eval will be singular for 1e test because Beta det is 0
  """
  print("Accumulating:\n", acc_names, flush=True)
  acc_name_key = {
    "AccumulatorCuspTestingLEAllE": "LE",
    "AccumulatorCuspTestingKE": "KE",
    "AccumulatorCuspTestingPsiSquared": "PsiSquared",
  }
  # copy current cusp_test_options to a new dict and modify it for the cusp testing
  cusp_test_options = options.copy()

  nucpos = cusp_test_options["nuclei"]
  print("nucpos:\n", nucpos, flush=True)

  # get number of nuclei
  nnuc = cusp_test_options["nuclei"].size // 3
  
  # get number of alpha electrons
  nalp = cusp_test_options["apos"].size // 3
  
  cusp_test_options["WFNPieces"] = [
                              #"RMPJastrowAA",  # jastrow terms tend to 1 since e- are far apart
                              #"RMPJastrowBB",
                              #"RMPJastrowAB",
                              "AlphaDet", 
                              "BetaDet"]   # need beta for probability


  cusp_test_options["do_batch"] = False
  cusp_test_options["do_sherman_morrison"] = True
  
  # specify number of points to calculate and the step size --> make sure to change the array size in c++/accumulator_ct_ke.cpp
  npts = 6000
  # specify number of points to calculate and the step size --> make sure to change the array size in c++/accumulator_ct_ke.cpp, _le.cpp. and _psi_squared.cpp
  step_size = 0.0001
  #step_size = 0.0001 #0.00001

  pos_pos = np.arange(step_size, step_size*npts/2+step_size, step_size)
  neg_pos = -1.0 * np.flip(pos_pos)
  r = np.concatenate((neg_pos, pos_pos), axis=0)
  np.savetxt('C_in_MeOH_r_vals.txt', r)
  #print("r:\n", r, flush=True)
  eval_dict = { 
      "r": r, 
      #"r": np.arange(step_size, step_size*npts+step_size, step_size),

      "mocoeff": cusp_test_options["mocoeff"], 
  }
  #print("eval_dict:", eval_dict, flush=True)

  ## DO NORMAL SAMPLING HERE TO GET GOOD STARTING GEOM
  ############################################################################
  ############################################################################
  
  nthread = get_number_of_threads()
  
  acc_names_vanilla = ["AccumulatorVGR"]
  #acc_names = ["AccumulatorVGR", "AccumulatorKE", "AccumulatorEEE", "AccumulatorENE", "AccumulatorNNE", "AccumulatorLocalE", "AccumulatorLocalESquared", "AccumulatorAss", "AccumulatorLocalEAss", "AccumulatorAos", "AccumulatorLocalEAos", "AccumulatorLocalEAll", ]
  
  elec_pos = np.zeros([nthread, 2, nalp, 3])
  for i in range(nthread):
    elec_pos[i, 0, :, :] = np.reshape(cusp_test_options["apos"], [nalp,3])
    elec_pos[i, 1, :, :] = np.reshape(cusp_test_options["bpos"], [nalp,3])
  
  acc_dict_vanilla = make_accumulator_dict(acc_names_vanilla, cusp_test_options)
  
  vmc_take_samples_detail(cusp_test_options, acc_dict_vanilla, elec_pos)
  
  cusp_test_options["apos"] = elec_pos[0, 0, :, :].flatten()
  cusp_test_options["bpos"] = elec_pos[0, 1, :, :].flatten()

  ############################################################################
  ############################################################################
  # find closest alpha electron to the nucleus and switch it with the 0th index
  apos_van = cusp_test_options["apos"].copy()
  print("Nucpos:\n", nucpos, flush=True)
  print("Apos after vanilla round:\n", apos_van, flush=True)
  adist = np.zeros((nalp))

  nuc_ind = 0
  for k in range(nalp):
      adist[k] = np.sqrt( (nucpos[nuc_ind*3 + 0] - apos_van[k*3 + 0])**2.0 + (nucpos[nuc_ind*3 + 1] - apos_van[k*3 + 1])**2.0 + (nucpos[nuc_ind*3 + 2] - apos_van[k*3 + 2])**2.0 )

  print("Alpha distances:\n", adist, flush=True)
  order = np.argsort(adist)
  for k in range(nalp):
      cusp_test_options["apos"][k*3:k*3+3] = apos_van[order[k]*3:order[k]*3+3]

  print("Apos after sorting:\n", cusp_test_options["apos"], flush=True)
  
  ############################################################################
  ############################################################################

  nthread = 1
  print("")
  print("have %i threads for do_thru_nuc_test" % nthread)
  print("")
 
  cusp_test_options["nsamp_per_block"] = npts * (nnuc)
  cusp_test_options["nblock"] = 10 * get_number_of_threads()
  cusp_ke_positions = []

  # NEED TO ADD POSITIONS ON OTHER SIDE OF NUC
  # generating positions for KE testing
  for i in range(nnuc):
  #for i in range(nnuc):
    cusp_ke_positions_nuc_neg = []
    cusp_ke_positions_nuc_pos = []
    for j in range(int(npts/2)):
    #for j in range(npts/2):
      cusp_ke_positions_nuc_pos.append( 1.0 * nucpos[3*i:3*(i+1)] )
      cusp_ke_positions_nuc_neg.append( 1.0 * nucpos[3*i:3*(i+1)] )
      #cusp_ke_positions.append( 1.0 * nucpos[3*i:3*(i+1)] - np.array([(npts/2)*step_size, 0.0, 0.0]) )
      #cusp_ke_positions[-1][0] -= 0.00001 * (j+1) 
      cusp_ke_positions_nuc_pos[-1][0] += step_size * (j+1) 
      cusp_ke_positions_nuc_neg[-1][2] -= step_size * (j+1) 
    neg_pos = np.concatenate(cusp_ke_positions_nuc_neg, axis=0)
    pos_pos = np.concatenate(cusp_ke_positions_nuc_pos, axis=0)
    neg_pos = np.flip(neg_pos)
    cusp_ke_positions.append(neg_pos)
    cusp_ke_positions.append(pos_pos)
      #cusp_ke_positions[-1][1] += 0.00001 * (j+1) 
  cusp_test_options["cusp_testing_positions"] = np.concatenate(cusp_ke_positions, axis=0)
  #print("Cusp position array:\n", cusp_test_options["cusp_testing_positions"], flush=True)
 
  nbat = len(cusp_test_options["batches"])

  for which_orb in which_orb_all:
    print("which_orb in do_cusp_test_hack() ", which_orb)
    ########################################
    if which_orb == "slater":
      cusp_test_options["useCuspGTO"] = True 
      cusp_test_options["get_slater_derivs_cusp"] = True 
    elif which_orb == "gaussian":
      cusp_test_options["useCuspGTO"] = False
      cusp_test_options["get_slater_derivs_cusp"] = False 
    elif which_orb == "cusp":
      cusp_test_options["useCuspGTO"] = True 
      cusp_test_options["get_slater_derivs_cusp"] = False 
    else:
      raise Exception("orbital type to eval KE at not recognized")
    ########################################
    num_iter = 1 


    for a in range(num_iter):

      # Prepare an array to hold the electron positions for each thread.
      # To start we set this to the initial electron positions, but once we've taken
      # samples we just keep each thread's most recent sample here.
      elec_pos = np.zeros([nthread, 2, nalp, 3])
      for i in range(nthread):
        elec_pos[i, 0, :, :] = np.reshape(cusp_test_options["apos"], [nalp,3])
        elec_pos[i, 1, :, :] = np.reshape(cusp_test_options["bpos"], [nalp,3])
              
      # prepare dictionary that holds the data arrays for the different accumulators
      #acc_names = ["AccumulatorCuspTestingKE"]
      acc_dict = make_accumulator_dict(acc_names, cusp_test_options)
        
      # take the sample and accumulate the data we need
      vmc_take_sample_for_nuc_cusp_testing(cusp_test_options, acc_dict, elec_pos)

      #np.savetxt('C_in_MeOH_LE.txt', acc_dict["AccumulatorCuspTestingLEAllE"])

      for acc_type in acc_names: 
        print(acc_type, acc_dict[acc_type].size, flush=True)

        for i in range(acc_dict[acc_type].size):  # loop throu npts

          ke_vals = []
          if (i+1)%npts == 0:

            ke_vals.append(acc_dict[acc_type][i-(npts-1):i+1])
            nuc_num = int((i+1)/npts-1)
              
            eval_dict[acc_name_key[acc_type]+"_"+which_orb+"_nuc"+str(nuc_num)] = ke_vals[0]
            print("\t\tsaving: ", acc_name_key[acc_type]+"_"+which_orb+"_nuc"+str(nuc_num)) #, " first value at r= ", step_size, ": ", ke_vals[0][0], " slater and cusp should match (closely) ")

            print("")

            #pickle.dump(eval_dict, open(cusp_test_options['file_name']+'_'+acc_type[-2:]+'_test_dict.pkl', "wb"))
      pickle.dump(eval_dict, open(cusp_test_options['file_name']+'_thru_nuc_test_dict.pkl', "wb"))

def do_LM_grad_test(options_pkl):
  """ test to check accumulated LM derivatives against finite difference

  params: options - internal_optiions pkl dictionary

  """
  # TODO check to see if threading is causing any issue with this
  #print('OMP_NUM_THREADS', os.environ['OMP_NUM_THREADS'], flush=True)
  #og_omp = os.environ['OMP_NUM_THREADS']
  #os.environ['OMP_NUM_THREADS'] = "1"
  print('OMP_NUM_THREADS ', os.environ['OMP_NUM_THREADS'], flush=True)

  with open(options_pkl, 'rb') as fp:
    options = pickle.load(fp)

  print()
  print("******************** IN do_LM_grad_test() ********************",flush=True)

  print("FIXED LM for H systems as of 02112025",flush=True)

  nvp = options['nvp'] 
  mocoeff = options["mocoeff"].copy()   # reference C mat
  grad_test_options = options.copy()
  #print("Initial C mat\n",mocoeff, flush=True)
  norb = mocoeff.shape[0]
  nalp = options["apos"].size // 3
  nthread = 1 
  print("")
  print("have %i threads for do_LM_grad_test" % nthread,flush=True)
  print("")

  # Prepare an array to hold the electron positions for each thread.
  # To start we set this to the initial electron positions, but once we've taken
  # samples we just keep each thread's most recent sample here.
  elec_pos = np.zeros([nthread, 2, nalp, 3])
  for i in range(nthread):
    elec_pos[i, 0, :, :] = np.reshape(grad_test_options["apos"], [nalp,3])
    elec_pos[i, 1, :, :] = np.reshape(grad_test_options["bpos"], [nalp,3])

  # want only 1 samp
  grad_test_options["nsamp_per_block"] = 1 
  #grad_test_options["nblock"] = 1 
  grad_test_options["nblock"] = 10 * get_number_of_threads()
  grad_test_options["nwarmup"] = 0 

  diff = 1e-6

  fd_PvP0 = []
  fd_gradKE = []
  fd_gradLE = []
  fd_gradglp = []

  # reference calc to get LM derivs to compare against
  #acc_names = ["AccumulatorPvP0","AccumulatorGradE",] # only need these for ref
  acc_names = ["AccumulatorVGR", "AccumulatorKE", "AccumulatorEEE", "AccumulatorENE", "AccumulatorNNE", "AccumulatorLocalE","AccumulatorGradE","AccumulatorPvP0",]
  acc_dict = make_accumulator_dict(acc_names, grad_test_options, nvp, grad_test_options["nblock"])
  for key, value in acc_dict.items():
    print(key, value.shape, flush=True)

  # initialize finite diff to be populated after all calcs
  fd_ddv_gradlogpsi = np.zeros((3,2*nalp,norb,nalp))

  fd_ddv_laplogpsi = np.zeros((norb,nalp))

  # intialize arrays to be filled on c++ side
  wfn_ref =  np.array([0.0])
  ke_ref = np.array([0.0])
  gradlogpsi_ref = np.zeros(3 * 2 * nalp)
  ddv_gradlogpsi_ref = np.zeros((3,2 * nalp * nalp * norb)) # column-major noxna array alpha then beta for each row = mu
  ddv_laplogpsi_ref = np.zeros(nalp * norb) # column-major noxna array alpha then beta for each row = mu

  print("\n\nReference calc (LM evalutions):", flush=True)

  vmc_take_sample_for_LM_grad_testing(grad_test_options, acc_dict, elec_pos, ke_ref, wfn_ref, gradlogpsi_ref, ddv_gradlogpsi_ref, ddv_laplogpsi_ref, np.array([diff]))

  # check1.0 to make sure ke_ref is just grad.grad term
  #py_ke_ref = -0.5 * np.dot(gradlogpsi_ref, gradlogpsi_ref)
  #print("check1.ref --- KE from calc",  ke_ref, " should be ", py_ke_ref, "diff: ", np.round(ke_ref-py_ke_ref,10),flush=True)

  #print("local E accumulated: ",acc_dict["AccumulatorLocalE"][0], flush=True) # [nvp,1] vector
  #print("KE simple_vmc.py",ke_ref,flush=True)
  #print("Wavefunction eval simple_vmc.py",wfn_ref,flush=True)
  #print("Grad log psi eval simple_vmc.py",np.transpose(gradlogpsi_ref.reshape(2*nalp,3)),flush=True)

  ### NEED TO CHECK IF THESE MATCH FINITE DIFF TEST  
  # PASSED 
  PvP0_LM = np.transpose(np.reshape(acc_dict["AccumulatorPvP0"][0], [nalp, norb])) # [nvp,1] vector -> [no,na]
  #print("LM: PvP0 ")
  #print(PvP0_LM,flush=True)

  # PASSED 
  gradE_LM = np.transpose(np.reshape(acc_dict["AccumulatorGradE"][0], [nalp, norb])) # [nvp,1] vector -> [no,na]
  #print("LM: GradE ")
  #print(gradE_LM,flush=True)
  #print()

  # PASSED 
  ddv_gradlogpsi_LM = np.transpose(ddv_gradlogpsi_ref.reshape(3,2*nalp,nalp,norb),(0,1,3,2)) # [3,2*nalp,norb,nalp]

  # TDB 
  #print("ddv_laplogpsi_ref.shape",ddv_laplogpsi_ref.shape,flush=True)
  ddv_laplogpsi_LM = np.transpose(ddv_laplogpsi_ref.reshape(nalp,norb)) # [norb,nalp]

  #print("At given elec config and C mat: ")
  #print()
  #print("Reference KE", ke_ref, flush=True)
  #print()
  #print("Reference WFN", wfn_ref, flush=True)
  #print()
  #print("LM:  PvP0 (length", PvP0_LM.shape,")", flush=True)
  #print(PvP0_LM, flush=True)
  #print()
  #print("LM: GradE (length", gradE_LM.shape,")")
  #print(gradE_LM,flush=True)
  #print()
  #print("LM: ddv_gradlogpsi = d/dc_ij grad_m log(psi)\n",ddv_gradlogpsi_LM,flush=True)

  print("\n\nOn to finite diff calc:", flush=True)

  #print("\n==================================================\n", flush=True)
  # loop through the parameters (in column major) to get gradient of each 
  for col in np.arange(int(mocoeff.shape[1])):
    for row in np.arange(int(mocoeff.shape[0])):
      #print("\n============ Param ", row, col,"==================\n", flush=True)
      # for given param in Cmat, finite diff wrt +/- 10e-6
      wfn_list = []
      ke_list = []
      le_list = []
      glp_list = []
      for sgn in [1, -1]:
        #print("\nCmat before\n", grad_test_options["mocoeff"],flush=True)
        # update mocoeff element
        grad_test_options["mocoeff"][row,col] = mocoeff[row,col] + sgn * diff
        #print("\nCmat after\n", grad_test_options["mocoeff"],flush=True)

        # make acc storage dict
        acc_dict = make_accumulator_dict(acc_names, grad_test_options, nvp, grad_test_options["nblock"])

        # intialize arrays to hold results
        ke = np.array([0.0])
        wfn_val = np.array([0.0])
        gradlogpsi = np.zeros(3 * 2 * nalp)
        ddv_gradlogpsi = np.zeros((3,2 * nalp * nalp * norb)) # column-major noxna array alpha then beta for each row = mu
        ddv_laplogpsi = np.zeros(nalp * norb) # column-major noxna array alpha then beta for each row = mu
  
        # accumulate/populate the data we need at given elec config
        vmc_take_sample_for_LM_grad_testing(grad_test_options, acc_dict, elec_pos, ke, wfn_val, gradlogpsi, ddv_gradlogpsi, ddv_laplogpsi, np.array([diff]))

        # check1.1 to make sure ke_ref is just grad.grad term
        py_ke= np.dot(gradlogpsi, gradlogpsi)
        #print("check1.ij  --- KE from calc",  ke, " should be ", py_ke, "diff: ", np.round(ke-py_ke,10),flush=True)

        #print("KE from fd calc",  ke, flush=True)
        #print("WFN from fd calc",  wfn_val, flush=True)
        localE = acc_dict["AccumulatorLocalE"][0] # [nvp,1] vector

        le_list.append(localE)
        ke_list.append(ke)
        wfn_list.append(wfn_val)
        glp_list.append(gradlogpsi) # still in vector form

        # reset mocoeff for next param check
        grad_test_options["mocoeff"][row,col] = mocoeff[row,col]

      #print("\nWfn list:", wfn_list, flush=True)
      #print("\nKE list:", ke_list, flush=True)
      #print("\ngrad log psi list:", glp_list, flush=True)
      #print("\nLE list:", le_list, flush=True)

      ## Finite difference calculation for this d/dc_i,j - CORRECT values to be compared against
      ### f'(x) = ( f(x+a) - f(x-a) ) / 2a ###
      pvp0_el = ((wfn_list[0] - wfn_list[1]) / (2*diff*wfn_ref))      # FD of psi/psi < dont forget non-deriv denom
      fd_PvP0.append(pvp0_el[0])

      grad_ke = ((ke_list[0] - ke_list[1]) / (2*diff))      
      fd_gradKE.append(grad_ke[0])

      # Should be equivalent to grad_KE
      grad_le = ((le_list[0] - le_list[1]) / (2*diff))      
      fd_gradLE.append(grad_le)

      grad_glp = ((glp_list[0] - glp_list[1]) / (2*diff))  # vector form: (xyz)_m=1...(xyz)_m=nelec
      fd_gglp = np.transpose(grad_glp.reshape(2*nalp,3))   # [3,nelec]

      # [3,nelec,no,na] - fd_gglp will fill in the i,j element for each elec noxna matrix for x y and z
      for mu in range(3):
        for elec in range(2*nalp):
          fd_ddv_gradlogpsi[mu,elec,row,col] = fd_gglp[mu][elec]

      # Not checked if working
      lap_glp = ((glp_list[0] - 2*gradlogpsi_ref + glp_list[1]) / (diff**2))  # vector form: (xyz)_m=1...(xyz)_m=nelec
      fd_lglp = np.transpose(lap_glp.reshape(2*nalp,3))   # [3,nelec]
      # [no,na] - fd_lglp will fill in the i,j element summed over each elec and mu noxna matrix
      for mu in range(3):
        for elec in range(2*nalp):
# is this right?
          fd_ddv_laplogpsi[row,col] += fd_gglp[mu][elec]


      #print("\n-------Finite Difference--------\n",flush=True)
      #print("Calculated PvP0 vector =  ", "\n", pvp0_el)
      #print("Calculated GradE vector = ", "\n", grad_ke)
      #print("Calculated LocalE vector = ", "\n", grad_le)
      #print("running PvP0 FD vec:", fd_PvP0)
      #print("running gradKE FD vec:", fd_gradKE)
      #print("FD of grad log psi\n", fd_ddv_gradlogpsi, flush=True)  # it'll pop 1 element of eah matrix

  gradlogpsi_ref = np.transpose(gradlogpsi_ref.reshape(2*nalp,3)) # [3,2*nalp]

  ## Reshape finite diff results from vectors to matrices
  fd_PvP0 = np.transpose(np.asarray(fd_PvP0).reshape(nalp,norb))  # [norb,nalp]
  fd_gradKE = np.transpose(np.asarray(fd_gradKE).reshape(nalp,norb))  # [norb,nalp]
  fd_gradLE = np.transpose(np.asarray(fd_gradLE).reshape(nalp,norb))  # [norb,nalp]

  rnd_to = 4
  print()
  print("\n#################################\n")
  print("PvP0")
  #print("LM ref\n", np.round(PvP0_LM,9))
  #print("finite diff - fd_PvP0\n", fd_PvP0)
  #print("diff rounded: \n", np.round(PvP0_LM - fd_PvP0,rnd_to))
  print("diff rounded: \n", np.sum(np.round(PvP0_LM - fd_PvP0,rnd_to)))
  print("\n#################################\n", flush=True)

  print("\n#################################\n")
  print("GradKE")
  #print("\nLM ref - current dottig gradlogpsi and compute_ddv_gradlogpsi output\n", gradE_LM)
  #print("\nfinite diff - fd_gradKE\n", fd_gradKE)
  #print("diff rounded: \n", np.round(gradE_LM - fd_gradKE,rnd_to))
  print("diff rounded: \n", np.sum(np.round(gradE_LM - fd_gradKE,rnd_to)))
  print("\n#################################\n", flush=True)

  ## equiv to grad_KE
  #print("\n#################################\n")
  #print("GradLE")
  ##print("LM ref\n", gradE_LM)
  ##print("finite diff\n", fd_gradLE)
  #print("diff rounded: \n", np.round(gradE_LM - fd_gradLE,rnd_to))
  #print("\n##################np.round###############\n")

  #print()
  #print("\n#################################\n")
  #print("ddv grad_log_psi piece of KE")
  ##print("ddv_gradlogpsi_LM")
  ##print(ddv_gradlogpsi_LM)
  ##print("fd_ddv_gradlogpsi\n", fd_ddv_gradlogpsi)
  ddvgpl_check = np.round(ddv_gradlogpsi_LM - fd_ddv_gradlogpsi,rnd_to)
  #print("diff rounded: ", ddvgpl_check)
  #print("\n#################################\n")
  #################
  ## Check to make sure the finite diff ddv_gradlogpsi is correct by computing gradE and comparing to fd gradE
  #print()
  #print("Checking my finite diff reference used above for correctness: (grad(mu) logpsi_m) . (d/dc_ij grad(mu) log_psi):")
  #grad_fd_ddv_gradlogpsi = np.zeros((norb,nalp)) 
  #for mu in range(3):
  #  for elec in range(2*nalp):
  #    # multiply the d/dc_ij gradlogpsi no x na mat by the grad log psi element for that elec/mu's matrix
  #    grad_fd_ddv_gradlogpsi += (gradlogpsi_ref[mu, elec] * fd_ddv_gradlogpsi[mu,elec,:,:]) # scalar * noxna matrix
  #print("Should be equiv:",
  #      "shape of grad_fd_ddv_gradlogpsi", grad_fd_ddv_gradlogpsi.shape,
  #      "shape of fd_gradKE", fd_gradKE.shape, flush=True)
  #print()
  #check3_diff = fd_gradKE - 2*grad_fd_ddv_gradlogpsi
  #check3_diff_sum = np.sum(np.sum(check3_diff)).round(rnd_to)
  #print("check3.fd  --- fd_gradKE should analytically match 2 * grad_fd_ddv_gradlogpsi = ", check3_diff_sum, " should be zero \n",)
  #################

  #print()
  ##check3_diff_sum = 0.0
  #if check3_diff_sum == 0.0:
  #  print("(d/dc_ij grad_m ln(psi))")
  #  #print("diff rounded: ", np.round(ddv_gradlogpsi_LM - fd_ddv_gradlogpsi,rnd_to))
  #  #print("ddv_gradlogpsi_LM\n", ddv_gradlogpsi_LM)
  #  #print("fd_ddv_gradlogpsi\n", fd_ddv_gradlogpsi)
  #  grad_LM_ddv_gradlogpsi = np.zeros((norb,nalp)) 
  #  for spin in [0,1]:
  #    for mu in range(3):
  #      for elec in range(nalp):
  #        # multiply the d/dc_ij gradlogpsi no x na mat by the grad log psi element for that elec/mu's matrix
  #        #print("\n\nmu, elec", mu, elec)
  #        #print("scale for ddv_gradlogpsi_LM (aka ddmu_i_scalar)", gradlogpsi_ref[mu, spin * nalp + elec])
  #        #print("ddv_gradlgpsi_LM\n", ddv_gradlogpsi_LM[mu, spin * nalp + elec,:,:])
  #        #print("grad_LM_ddv_gradlogpsi running total for this mu elec\n", (gradlogpsi_ref[mu, spin * nalp + elec] * ddv_gradlogpsi_LM[mu, spin * nalp + elec,:,:]))
  #        grad_LM_ddv_gradlogpsi += (gradlogpsi_ref[mu, spin * nalp + elec] * ddv_gradlogpsi_LM[mu, spin * nalp + elec,:,:]) # scalar * noxna matrix
  #    print("grad_E = 2*grad_LM_ddv_gradlogpsi running total\n", 2*grad_LM_ddv_gradlogpsi)
  #    
  #  check3LM_diff = fd_gradKE - 2*grad_LM_ddv_gradlogpsi
  #  check3LM_diff_sum = np.sum(np.sum(check3LM_diff)).round(rnd_to)
  #  print("check3.LM  --- fd_gradKE should analytically match 2 * grad_LM_ddv_gradlogpsi = ", check3LM_diff_sum, " should be zero \n",)
  #
  #print()
  #print("fd_gradKE\n", fd_gradKE, "2*grad_fd_ddv_gradlogpsi\n", 2*grad_fd_ddv_gradlogpsi,"which should be whats in = grad_E_LM\n", gradE_LM, "AKA also 2*grad_LM_ddv_gradlogpsi\n", 2*grad_LM_ddv_gradlogpsi)
  #print()
  #print("(d/dc_ij lap_m ln(psi))")
  #print("diff rounded: ", np.round(ddv_laplogpsi_LM - fd_ddv_laplogpsi,rnd_to))

  #os.environ['OMP_NUM_THREADS'] = og_omp 
  #print('OMP_NUM_THREADS', os.environ['OMP_NUM_THREADS'], flush=True)
  if np.sum(np.round(PvP0_LM - fd_PvP0,rnd_to)) != 0.0:
      raise RuntimeError("PvP0 finite difference test failed")

  if np.sum(np.round(gradE_LM - fd_gradKE,rnd_to)) != 0.0:
    if ddvgpl_check != 0.0:
      raise RuntimeError("GradE finite difference test failed -- ddv gradient term (gradlogpsi) in KE does not match LM to finite difference")
    raise RuntimeError("GradE finite difference test failed  likely because of issue with the d/dv laplacian tern")


def do_1e_cusp_test(options, mo_list, acc_names=["AccumulatorCuspTestingLE", "AccumulatorCuspTestingKE", "AccumulatorCuspTestingPsiSquared"], which_orb_all={"cusp","gaussian"},savefiles=True, printerronly=False, name=None): #,"slater"}):
  """ which_orb takes in a set of string to eval the KE and/or local_E depending on input to acc_names
        slater eval will be singular for 1e test because Beta det is 0
  """
  if name == None: name=options['file_name']
  print("Accumulating:\n", acc_names, flush=True)
  acc_name_key = {
    "AccumulatorCuspTestingLE": "LE",
    "AccumulatorCuspTestingKE": "KE",
    "AccumulatorCuspTestingPsiSquared": "PsiSquared",
  }
  # copy current cusp_test_options to a new dict and modify it for the cusp testing
  cusp_test_options = options.copy()
  mocoeff = cusp_test_options["mocoeff"].copy()

  # adding dummy H nucleus
  new_H_for_test = np.array([260.0, 0.0, 0.0])
  cusp_test_options["nuclei"] = np.concatenate((cusp_test_options["nuclei"], new_H_for_test), axis=0)
  cusp_test_options["Z"] = np.concatenate((cusp_test_options["Z"], np.array([[1.0],])), axis=0)
  # adding dummy H nucleus

  # get number of nuclei
  nnuc = cusp_test_options["nuclei"].size // 3

  nucpos = cusp_test_options["nuclei"]

  # set positions of electrons
  cusp_test_options["apos"] = np.array([0.0, 0.0, 0.0])
  cusp_test_options["bpos"] = new_H_for_test + 0.1

  cusp_test_options["batch_mat_a"] = np.array([[1.0],])
  cusp_test_options["batch_mat_b"] = np.array([[1.0],])
  cusp_test_options["batch_mat_n"] = np.ones((nnuc,1))
  cusp_test_options["batches"] = np.array([[0, 1],])
  cusp_test_options["num_batches"] = len(cusp_test_options["batches"])  

  # get number of alpha electrons
  nalp = cusp_test_options["apos"].size // 3
  
  cusp_test_options["frag_nuc_list"] = np.arange(0, nnuc).reshape(1,nnuc)  
  cusp_test_options["frag_nuc_mat"] = np.ones((nnuc, 1)) 
  cusp_test_options["elec_frag_sizes"] = np.array([2*nalp]).reshape(1,1) 
  cusp_test_options["zone_mat"] = 1.0 * np.array([3]).reshape(1,1) 
  cusp_test_options["frag_elec_mat"] =  np.ones((2*nalp,1))
  cusp_test_options["num_frags"] = 1
  cusp_test_options["WFNPieces"] = [
                              #"RMPJastrowAA",  # jastrow terms tend to 1 since e- are far apart
                              #"RMPJastrowBB",
                              #"RMPJastrowAB",
                              "AlphaDet", 
                              "BetaDet"]   # need beta for probability
  #cusp_test_options["WFNPieces"] = ["AlphaDet","BetaDet",]

  #print("Old stuff:", flush=True)
  #print("basis_exp:\n", cusp_test_options["basis_exp"]) 
  #print("basis_coeff:\n", cusp_test_options["basis_coeff"]) 
  #print("basis_centers:\n", cusp_test_options["basis_centers"]) 
  #print("basis_orb_type:\n", cusp_test_options["basis_orb_type"]) 
  #print("mocoeff:\n", cusp_test_options["mocoeff"])
  #print("orth_orb_array:\n", cusp_test_options["orth_orb_array"]) 
  #print("cusp_radii_mat:\n", cusp_test_options["cusp_radii_mat"]) 
  #print("proj_mat:\n", cusp_test_options["proj_mat"]) 
  #print("cusp_a0:\n", cusp_test_options["cusp_a0"]) 

  # adding basis set info for dummy nucleus (6-31G basis). Make sure to change it if using a diff basis
  if cusp_test_options["basis_type"] == 'STO-3G':
    cusp_test_options["basis_exp"] = np.concatenate((cusp_test_options["basis_exp"], np.array([[ 3.425250914, 0.6239137298, 0.1688554040, ],])), axis=0)  # 2s
    cusp_test_options["basis_coeff"] = np.concatenate((cusp_test_options["basis_coeff"], np.array([[ 0.1543289673, 0.5353281423, 0.4446345422, ],])), axis=0)  # H: 2s
    cusp_test_options["basis_centers"] = np.concatenate((cusp_test_options["basis_centers"], np.array([[nnuc-1],])), axis=0) 
    cusp_test_options["basis_orb_type"] = np.concatenate((cusp_test_options["basis_orb_type"], np.array([[0],])), axis=0) 

    cusp_test_options["nbf"] = len(cusp_test_options["basis_exp"])
    nbf = cusp_test_options["nbf"]
    npbf = cusp_test_options["num_p_func"]
    print("num_p_func in 1e walker test", npbf, flush=True)

    # concatenating the cusp options with info for our dummy H nucleus
    cusp_test_options["orth_orb_array"] = np.concatenate((cusp_test_options["orth_orb_array"], np.array([[0,]])), axis=None) 
    cusp_test_options["cusp_radii_mat"] = np.concatenate((cusp_test_options["cusp_radii_mat"], np.zeros((nnuc-1,1))), axis=1)
    cusp_test_options["cusp_radii_mat"] = np.concatenate((cusp_test_options["cusp_radii_mat"], np.zeros((1,nbf))), axis=0)
    cusp_test_options["proj_mat"] = np.concatenate((cusp_test_options["proj_mat"], np.zeros((nbf-1,1))), axis=1)
    cusp_test_options["proj_mat"] = np.concatenate((cusp_test_options["proj_mat"], np.zeros((1,nbf))), axis=0)
    cusp_test_options["cusp_a0"] = np.concatenate((cusp_test_options["cusp_a0"], np.zeros((nnuc-1,1))), axis=1)
    cusp_test_options["cusp_a0"] = np.concatenate((cusp_test_options["cusp_a0"], np.zeros((1,nbf))), axis=0)
    cusp_test_options["cusp_coeff_matrix"] = np.concatenate((cusp_test_options["cusp_coeff_matrix"], np.zeros((nnuc-1, 1, npbf))), axis=1)  
    cusp_test_options["cusp_coeff_matrix"] = np.concatenate((cusp_test_options["cusp_coeff_matrix"], np.ones((1, nbf, npbf))), axis=0)

  elif cusp_test_options["basis_type"] in {'6-31G', '6-31G*', '6-31Gd'}:
    # adding a H atom which is equiv for 6-31G and 6-31G*
    cusp_test_options["basis_exp"] = np.concatenate((cusp_test_options["basis_exp"], np.array([[0.1873113696e02, 0.2825394365e01, 0.6401216923e00, 0.0, 0.0, 0.0],[0.1612777588e00, 0.0, 0.0, 0.0, 0.0, 0.0]])), axis=0)  # 2s
    cusp_test_options["basis_coeff"] = np.concatenate((cusp_test_options["basis_coeff"], np.array([[0.3349460434e-01, 0.2347269535e00, 0.8137573261e00, 0.0, 0.0, 0.0],[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])), axis=0)  # H: 2s
    cusp_test_options["basis_centers"] = np.concatenate((cusp_test_options["basis_centers"], np.array([[nnuc-1],[nnuc-1],])), axis=0) 
    cusp_test_options["basis_orb_type"] = np.concatenate((cusp_test_options["basis_orb_type"], np.array([[0],[1],])), axis=0) 

    cusp_test_options["nbf"] = len(cusp_test_options["basis_exp"])
    nbf = cusp_test_options["nbf"]
    npbf = cusp_test_options["num_p_func"]
    print("num_p_func in 1e walker test", npbf, flush=True)

    # concatenating the cusp options with info for our dummy H nucleus
    cusp_test_options["orth_orb_array"] = np.concatenate((cusp_test_options["orth_orb_array"], np.array([[0,0]])), axis=None) 
    cusp_test_options["cusp_radii_mat"] = np.concatenate((cusp_test_options["cusp_radii_mat"], np.zeros((nnuc-1,1))), axis=1)
    cusp_test_options["cusp_radii_mat"] = np.concatenate((cusp_test_options["cusp_radii_mat"], np.zeros((nnuc-1,1))), axis=1)
    cusp_test_options["cusp_radii_mat"] = np.concatenate((cusp_test_options["cusp_radii_mat"], np.zeros((1,nbf))), axis=0)
    cusp_test_options["proj_mat"] = np.concatenate((cusp_test_options["proj_mat"], np.zeros((nbf-2,1))), axis=1)
    cusp_test_options["proj_mat"] = np.concatenate((cusp_test_options["proj_mat"], np.zeros((nbf-2,1))), axis=1)
    cusp_test_options["proj_mat"] = np.concatenate((cusp_test_options["proj_mat"], np.zeros((1,nbf))), axis=0)
    cusp_test_options["proj_mat"] = np.concatenate((cusp_test_options["proj_mat"], np.zeros((1,nbf))), axis=0)
    cusp_test_options["cusp_a0"] = np.concatenate((cusp_test_options["cusp_a0"], np.zeros((nnuc-1,1))), axis=1)
    cusp_test_options["cusp_a0"] = np.concatenate((cusp_test_options["cusp_a0"], np.zeros((nnuc-1,1))), axis=1)
    cusp_test_options["cusp_a0"] = np.concatenate((cusp_test_options["cusp_a0"], np.zeros((1,nbf))), axis=0)
    cusp_test_options["cusp_coeff_matrix"] = np.concatenate((cusp_test_options["cusp_coeff_matrix"], np.zeros((nnuc-1, 1, npbf))), axis=1)  
    cusp_test_options["cusp_coeff_matrix"] = np.concatenate((cusp_test_options["cusp_coeff_matrix"], np.zeros((nnuc-1, 1, npbf))), axis=1)  
    cusp_test_options["cusp_coeff_matrix"] = np.concatenate((cusp_test_options["cusp_coeff_matrix"], np.ones((1, nbf, npbf))), axis=0)

  elif cusp_test_options["basis_type"] in {'cc-pcvdz',}:
    # adding a H atom which is equiv for 6-31G and 6-31G*

    H_exp=np.array([
         [1.301000000e+01, 1.962000000e+00, 4.446000000e-01, 1.220000000e-01, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
         [1.220000000e-01, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
         [7.270000000e-01, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
         [7.270000000e-01, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
         [7.270000000e-01, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
            ])
  
    H_coeff=np.array([
         [1.968500000e-02, 1.379770000e-01, 4.781480000e-01, 5.012400000e-01, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
         [1.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
         [1.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
         [1.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
         [1.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00],
            ])
    nHbf=H_coeff.shape[0] 

    cusp_test_options["basis_exp"] = np.concatenate((cusp_test_options["basis_exp"], H_exp), axis=0)  
    cusp_test_options["basis_coeff"] = np.concatenate((cusp_test_options["basis_coeff"], H_coeff), axis=0)  
    # should match numer of added AOs: 5
    cusp_test_options["basis_centers"] = np.concatenate((cusp_test_options["basis_centers"], np.array([[nnuc-1],[nnuc-1],[nnuc-1],[nnuc-1],[nnuc-1],])), axis=0) 
    cusp_test_options["basis_orb_type"] = np.concatenate((cusp_test_options["basis_orb_type"], np.array([[0],[1],[2],[3],[4],])), axis=0) 

    cusp_test_options["nbf"] = len(cusp_test_options["basis_exp"])
    nbf = cusp_test_options["nbf"]
    npbf = cusp_test_options["num_p_func"]
    print("num_p_func in 1e walker test", npbf, [0]*nHbf, flush=True)

    # concatenating the cusp options with info for our dummy H nucleus
    cusp_test_options["orth_orb_array"] = np.concatenate((cusp_test_options["orth_orb_array"], np.array([[0]*nHbf])), axis=None) 
    cusp_test_options["cusp_radii_mat"] = np.concatenate((cusp_test_options["cusp_radii_mat"], np.zeros((nnuc-1,nHbf))), axis=1)
    cusp_test_options["cusp_radii_mat"] = np.concatenate((cusp_test_options["cusp_radii_mat"], np.zeros((1,nbf))), axis=0)
    cusp_test_options["proj_mat"] = np.concatenate((cusp_test_options["proj_mat"], np.zeros((nbf-nHbf,nHbf))), axis=1)
    cusp_test_options["proj_mat"] = np.concatenate((cusp_test_options["proj_mat"], np.zeros((nHbf,nbf))), axis=0)
    cusp_test_options["cusp_a0"] = np.concatenate((cusp_test_options["cusp_a0"], np.zeros((nnuc-1,nHbf))), axis=1)
    cusp_test_options["cusp_a0"] = np.concatenate((cusp_test_options["cusp_a0"], np.zeros((1,nbf))), axis=0)
    cusp_test_options["cusp_coeff_matrix"] = np.concatenate((cusp_test_options["cusp_coeff_matrix"], np.zeros((nnuc-1, nHbf, npbf))), axis=1)  
    cusp_test_options["cusp_coeff_matrix"] = np.concatenate((cusp_test_options["cusp_coeff_matrix"], np.ones((1, nbf, npbf))), axis=0)

  cusp_test_options["do_batch"] = False
  cusp_test_options["do_sherman_morrison"] = True
  
  # specify number of points to calculate and the step size --> make sure to change the array size in c++/accumulator_ct_ke.cpp
  print("Hard coding do_1e* to 3 point only to get cusp")
  npts = 10000 # number of total samps to take
  nslope = 4 # number of those samps to calc slope from
  step_size = 0.00002

  #npts = 3000
  #step_size = 0.0001 #0.00001

  # specify number of points to calculate and the step size --> make sure to change the array size in c++/accumulator_ct_ke.cpp, _le.cpp. and _psi_squared.cpp
  eval_dict = { 
      "r": np.arange(step_size, step_size*npts+step_size, step_size),

      "mocoeff": cusp_test_options["mocoeff"], 
      }
  #print("eval_dict:", eval_dict, flush=True)

  nthread = 1 #get_number_of_threads()
  print("")
  print("have %i threads for do_cusp_test_hack" % nthread)
  print("")
 
  cusp_test_options["nsamp_per_block"] = npts * (nnuc-1)
  cusp_test_options["nblock"] = 10 * get_number_of_threads()
  cusp_ke_positions = []
  
  # generating positions for KE testing
  for i in range(nnuc-1):
    for j in range(npts):
      cusp_ke_positions.append( 1.0 * nucpos[3*i:3*(i+1)] )
      #cusp_ke_positions[-1][0] -= 0.00001 * (j+1) 
      cusp_ke_positions[-1][0] += step_size * (j+1) 
      #cusp_ke_positions[-1][1] += 0.00001 * (j+1) 
  cusp_test_options["cusp_testing_positions"] = np.concatenate(cusp_ke_positions, axis=0)
  #print("Cusp position array:\n", cusp_test_options["cusp_testing_positions"], flush=True)
 
  nbat = len(cusp_test_options["batches"])

  for which_orb in which_orb_all:
    print("which_orb in do_cusp_test_hack() ", which_orb)
    ########################################
    if which_orb == "slater":
      cusp_test_options["useCuspGTO"] = True 
      cusp_test_options["get_slater_derivs_cusp"] = True 
    elif which_orb == "gaussian":
      cusp_test_options["useCuspGTO"] = False
      cusp_test_options["get_slater_derivs_cusp"] = False 
    elif which_orb == "cusp":
      cusp_test_options["useCuspGTO"] = True 
      cusp_test_options["get_slater_derivs_cusp"] = False 
    else:
      raise Exception("orbital type to eval KE at not recognized")
    ########################################
    num_iter = 1 

    # looping over 1 short for Carbon because the last MO is singular - TODO could generalize in input
    if options["Z"].size == 1: # if atom avoid last mo  
      print("chopping off last MO for atom because it's singular!!!", flush=True)
      mo_list = mo_list[:2]

    bad_cusp=False
    print("will loop through these MOs", mo_list) #np.arange(0, len(mocoeff[0])))
    for m in mo_list: #np.arange(0, len(mocoeff[0])):
      cusp_test_options["mocoeff"] = np.concatenate((mocoeff[:,m].reshape(-1,1), np.array([[1.0],[1.0],])), axis=0)
      if printerronly == False:
        print("\n=============================")
        print("mocoeff", m, ) #":\n", np.array_str(cusp_test_options["mocoeff"].reshape(-1), precision=3))
        #print("------------------------------------------------------------------------------")
      for a in range(num_iter):

        # Prepare an array to hold the electron positions for each thread.
        # To start we set this to the initial electron positions, but once we've taken
        # samples we just keep each thread's most recent sample here.
        elec_pos = np.zeros([nthread, 2, nalp, 3])
        for i in range(nthread):
          elec_pos[i, 0, :, :] = np.reshape(cusp_test_options["apos"], [nalp,3])
          elec_pos[i, 1, :, :] = np.reshape(cusp_test_options["bpos"], [nalp,3])
              
        # prepare dictionary that holds the data arrays for the different accumulators
        #acc_names = ["AccumulatorCuspTestingKE"]
        acc_dict = make_accumulator_dict(acc_names, cusp_test_options)
        
        # take the sample and accumulate the data we need
        #vmc_take_sample_for_nuc_cusp_testing(cusp_test_options, acc_dict, elec_pos)

        slope = []
        y_int = []
        for acc_type in acc_names: 
          #print(acc_type, acc_dict[acc_type].size, flush=True)
          for i in range(acc_dict[acc_type].size):  # loop throu npts

            if (i-(nslope-1))%npts == 0 and acc_type[-2:] == "KE": # first four points from each nuc's r = 0
              nuc_num = int((i-(nslope-1))/npts)
              a, b = np.polyfit([(1/(step_size * (j+1))) for j in range(nslope)], acc_dict[acc_type][i-(nslope-1):i+1], 1)
              slope.append(a)
              y_int.append(b)
              ref_val = cusp_test_options['Z'][nuc_num]
              if printerronly == False:
                print("ref val", ref_val)
                print(#"\treal:  ", cusp_test_options['Z'][int((i-3)/npts)],
                    "\tmy cusp (slope): ", np.round(a,3)) #, "\t y-nt: ", np.round(b,3), flush=True)

              if acc_type == 'AccumulatorCuspTestingKE' and which_orb == 'cusp' and np.round(np.abs(a - ref_val),2) > 0.0:
                #bad_cusp=True
                # LE and PSiSquared must have already been processed for this to work
                rvals = eval_dict["r"][:nslope]
                #levals = eval_dict["LE_"+which_orb+"_mo"+str(m)+"_nuc"+str(nuc_num)][:10] 
                psi2vals = eval_dict["PsiSquared_"+which_orb+"_mo"+str(m)+"_nuc"+str(nuc_num)][:nslope] 
                mnorm = np.sum(eval_dict["mocoeff"][:,m]**2)
  
                #probxlesum = (4 * np.pi / mnorm) * np.sum(levals * rvals**2 * psi2vals)
                probsum = (4 * np.pi / mnorm) * np.sum(rvals**2 * psi2vals)
                if np.abs(probsum) > 10e-10:
                  bad_cusp = True
                  #print("~~~~~~~~~~~~~~ CUSP_ERROR ~~~~~~~~~~~~~~")
                  print("**** Cusp_ERROR -- MO:", m, " NUC:", nuc_num, "should be:", ref_val,
                      "but is:", np.round(a,3), " AND abs of sum of inner 10 (normalized 4pi r^2 * psi^2) > 10-10 = ", probsum) #, "LE scaled prob:", probxlesum) #, "\t y-nt: ", np.round(b,3), flush=True)
                else:
                  print("Cusp_issue -- MO:", m, " NUC:", nuc_num, "should be:", ref_val,
                        "but is:", np.round(a,4), " + abs of sum of inner 10 (normalized 4pi r^2 * psi^2) > 10e-10 = ", probsum) #, "LE scaled prob:", probxlesum) #, "\t y-nt: ", np.round(b,3), flush=True)
                #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

            ke_vals = []
            if (i+1)%npts == 0:

              ke_vals.append(acc_dict[acc_type][i-(npts-1):i+1])
              nuc_num = int((i+1)/npts - 1)
              
              eval_dict[acc_name_key[acc_type]+"_"+which_orb+"_mo"+str(m)+"_nuc"+str(nuc_num)] = ke_vals[0]
              #print("\t\tsaving: ", acc_name_key[acc_type]+"_"+which_orb+"_mo"+str(m)+"_nuc"+str(nuc_num)) #, " first value at r= ", step_size, ": ", ke_vals[0][0], " slater and cusp should match (closely) ")

              #print("")

            #pickle.dump(eval_dict, open(cusp_test_options['file_name']+'_'+acc_type[-2:]+'_test_dict.pkl', "wb"))
        # To save pickle file to plot
    if savefiles or bad_cusp==True: # save if any of the mo's are bad
      pickle.dump(eval_dict, open(name+'_1ecusp_test_dict.pkl', "wb"))
        #if bad_cusp:
        #  raise RuntimeError("KE cusp not satisfied within 1% of true value, check if tails over this region need a smaller threshold to cusp: ", a, " should be ", ref_val,flush=True) #cusp_test_options['Z'][int((i+1)/4 - 1)], flush=True)
  #return 1.0

#def do_1e_ke_cusp_test(options, which_orb_all={"cusp",}):
#
#  # copy current internal_options to a new dict and modify it for the cusp testing
#  internal_options = options.copy()
#  mocoeff = internal_options["mocoeff"].copy()
#
#  # adding dummy H nucleus
#  new_H_for_test = np.array([200.0, 0.0, 0.0])
#  internal_options["nuclei"] = np.concatenate((internal_options["nuclei"], new_H_for_test), axis=0)
#  internal_options["Z"] = np.concatenate((internal_options["Z"], np.array([[1.0],])), axis=0)
#
#  # get number of nuclei
#  
#  nnuc = internal_options["nuclei"].size // 3
#
#  nucpos = internal_options["nuclei"]
#
#  # set positions of electrons
#  internal_options["apos"] = np.array([0.0, 0.0, 0.0])
#  internal_options["bpos"] = np.array([200.1, 0.1, 0.1])
#
#  internal_options["batch_mat_a"] = np.array([[1.0],])
#  internal_options["batch_mat_b"] = np.array([[1.0],])
#  internal_options["batch_mat_n"] = np.ones((nnuc,1))
#  internal_options["batches"] = np.array([[0, 1],])
#  internal_options["num_batches"] = len(internal_options["batches"])  
#
#  # get number of alpha electrons
#  nalp = internal_options["apos"].size // 3
#  
#  internal_options["frag_nuc_list"] = np.arange(0, nnuc).reshape(1,nnuc)  
#  internal_options["frag_nuc_mat"] = np.ones((nnuc, 1)) 
#  internal_options["elec_frag_sizes"] = np.array([2*nalp]).reshape(1,1) 
#  internal_options["zone_mat"] = 1.0 * np.array([3]).reshape(1,1) 
#  internal_options["frag_elec_mat"] =  np.ones((2*nalp,1))
#  internal_options["num_frags"] = 1
#  internal_options["WFNPieces"] = []
#
#  print("Old stuff:")
#  print("basis_exp:\n", internal_options["basis_exp"]) 
#  print("basis_coeff:\n", internal_options["basis_coeff"]) 
#  print("basis_centers:\n", internal_options["basis_centers"]) 
#  print("basis_orb_type:\n", internal_options["basis_orb_type"]) 
#  print("mocoeff:\n", internal_options["mocoeff"])
#  print("orth_orb_array:\n", internal_options["orth_orb_array"]) 
#  print("cusp_radii_mat:\n", internal_options["cusp_radii_mat"]) 
#  print("proj_mat:\n", internal_options["proj_mat"]) 
#  print("cusp_a0:\n", internal_options["cusp_a0"]) 
#
#  # adding basis set info for dummy nucleus (6-31G basis). Make sure to change it if using a diff basis
#  internal_options["basis_exp"] = np.concatenate((internal_options["basis_exp"], np.array([[0.1873113696e02, 0.2825394365e01, 0.6401216923e00, 0.0, 0.0, 0.0],[0.1612777588e00, 0.0, 0.0, 0.0, 0.0, 0.0]])), axis=0)  # 2s
#  internal_options["basis_coeff"] = np.concatenate((internal_options["basis_coeff"], np.array([[0.3349460434e-01, 0.2347269535e00, 0.8137573261e00, 0.0, 0.0, 0.0],[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])), axis=0)  # H: 2s
#  internal_options["basis_centers"] = np.concatenate((internal_options["basis_centers"], np.array([[nnuc-1],[nnuc-1],])), axis=0) 
#  internal_options["basis_orb_type"] = np.concatenate((internal_options["basis_orb_type"], np.array([[0],[0],])), axis=0) 
#  internal_options["nbf"] = len(internal_options["basis_exp"])
#  internal_options["do_batch"] = False
#  internal_options["do_sherman_morrison"] = True
#  nbf = internal_options["nbf"]
#  
#  # concatenating the cusp options with info for our dummy H nucleus
#  internal_options["orth_orb_array"] = np.concatenate((internal_options["orth_orb_array"], np.array([[0,0]])), axis=None) 
#  internal_options["cusp_radii_mat"] = np.concatenate((internal_options["cusp_radii_mat"], np.zeros((nnuc-1,1))), axis=1)
#  internal_options["cusp_radii_mat"] = np.concatenate((internal_options["cusp_radii_mat"], np.zeros((nnuc-1,1))), axis=1)
#  internal_options["cusp_radii_mat"] = np.concatenate((internal_options["cusp_radii_mat"], np.zeros((1,nbf))), axis=0)
#  internal_options["proj_mat"] = np.concatenate((internal_options["proj_mat"], np.zeros((nbf-2,1))), axis=1)
#  internal_options["proj_mat"] = np.concatenate((internal_options["proj_mat"], np.zeros((nbf-2,1))), axis=1)
#  internal_options["proj_mat"] = np.concatenate((internal_options["proj_mat"], np.zeros((1,nbf))), axis=0)
#  internal_options["proj_mat"] = np.concatenate((internal_options["proj_mat"], np.zeros((1,nbf))), axis=0)
#  internal_options["cusp_a0"] = np.concatenate((internal_options["cusp_a0"], np.zeros((nnuc-1,1))), axis=1)
#  internal_options["cusp_a0"] = np.concatenate((internal_options["cusp_a0"], np.zeros((nnuc-1,1))), axis=1)
#  internal_options["cusp_a0"] = np.concatenate((internal_options["cusp_a0"], np.zeros((1,nbf))), axis=0)
#  
#  print("New stuff (make sure the addition of the dumer nucleus stuff worked):")
#  print("basis_exp:\n", internal_options["basis_exp"]) 
#  print("basis_coeff:\n", internal_options["basis_coeff"]) 
#  print("basis_centers:\n", internal_options["basis_centers"]) 
#  print("basis_orb_type:\n", internal_options["basis_orb_type"]) 
#  print("orth_orb_array:\n", internal_options["orth_orb_array"]) 
#  print("cusp_radii_mat:\n", internal_options["cusp_radii_mat"]) 
#  print("proj_mat:\n", internal_options["proj_mat"]) 
#  print("cusp_a0:\n", internal_options["cusp_a0"]) 
#
#  nthread = 1 #get_number_of_threads()
#  print("")
#  print("have %i threads for do_cusp_test_hack" % nthread)
#  print("")
#
#  # specify number of points to calculate and the step size --> make sure to change the array size in c++/accumulator_ct_ke.cpp
#  npts = 1000
#  step_size = 0.0001 #0.00001
# 
#  internal_options["nsamp_per_block"] = npts * (nnuc-1)
#  internal_options["nblock"] = 10 * get_number_of_threads()
#  cusp_ke_positions = []
#  
#  # generating positions for KE testing
#  for i in range(nnuc-1):
#    for j in range(npts):
#      cusp_ke_positions.append( 1.0 * nucpos[3*i:3*(i+1)] )
#      #cusp_ke_positions[-1][0] -= 0.00001 * (j+1) 
#      cusp_ke_positions[-1][1] += step_size * (j+1) 
#      #cusp_ke_positions[-1][1] += 0.00001 * (j+1) 
#  internal_options["cusp_testing_positions"] = np.concatenate(cusp_ke_positions, axis=0)
#  print("Cusp position array:\n", internal_options["cusp_testing_positions"], flush=True)
# 
#  nbat = len(internal_options["batches"])
#
#  num_iter = 1 
#
#  for m in range(len(mocoeff[0])-1):
#    internal_options["mocoeff"] = np.concatenate((mocoeff[:,m].reshape(-1,1), np.array([[1.0],[1.0],])), axis=0)
#    print("mocoeff:\n", internal_options["mocoeff"])
#      #print("------------------------------------------------------------------------------")
#    for a in range(num_iter):
#
#      # Prepare an array to hold the electron positions for each thread.
#      # To start we set this to the initial electron positions, but once we've taken
#      # samples we just keep each thread's most recent sample here.
#      elec_pos = np.zeros([nthread, 2, nalp, 3])
#      for i in range(nthread):
#        elec_pos[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
#        elec_pos[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])
#            
#      # prepare dictionary that holds the data arrays for the different accumulators
#      acc_names = ["AccumulatorCuspTestingKE"]
#      acc_dict = make_accumulator_dict(acc_names, internal_options)
#      
#      # take the sample and accumulate the data we need
#      vmc_take_sample_for_nuc_cusp_testing(internal_options, acc_dict, elec_pos)
#
#      print("")
#      print("cusp testing accumulator array for MO #: ", m)
#      slope = []
#      y_int = []
#       
#      # some writing to files for plotting later
#
#      filename = "mo_" + str(m) + "_ke_cusp.txt" 
#      f = open(filename, "w")
#      r_vals = []
#      f.write("r = np.array([")
#      for j in range(npts):
#        x = step_size * (j+1)
#        r_vals.append(x)
#        f.write("%.8f" % x)
#        f.write(",")
#        if (j+1) % 10==0:
#          f.write("\n")
#
#      f.write("])\n")
#      print("rvals:\n", r_vals)
#      for i in range(acc_dict["AccumulatorCuspTestingKE"].size):
#        ke_vals = []
#        if (i+1)%npts == 0:
#          #find line of best fit
#          #print([(1/(step_size * (i+1))) for i in range(4)])
#          a, b = np.polyfit([(1/(step_size * (i+1))) for i in range(npts)], acc_dict["AccumulatorCuspTestingKE"][i-(npts-1):i+1], 1)
#          slope.append(a)
#          y_int.append(b)
#          ke_vals.append(acc_dict["AccumulatorCuspTestingKE"][i-(npts-1):i+1])
#          print("made it to appending ke_vals", flush=True)
#          nuc_num = int((i+1)/npts-1)
#          # writing to file for plotting later
#          func = "y" + str(nuc_num) + " = np.array(["          
#          f.write(func)
#          for j in range(npts):
#            f.write("%.8f" % ke_vals[0][j])
#            f.write(",")
#            if (j+1) % 10==0:
#              f.write("\n")
#          f.write("])\n")
#
#          #print("x (r_vals)\n", r_vals)
#          print()
#          print("Kinetic energy data (y-values) for nuc #: ", nuc_num)
#          print(ke_vals)
#          print()
#          if np.abs(a - internal_options['Z'][int((i+1)/npts - 1)]) < 0.01:
#            print("\tslope: ", "%3.4e" %a, "\t y-nt: ", b)
#            print("\treal:  ", internal_options['Z'][int((i+1)/npts - 1)])
#            print("==========")
#          else:
#            print("cusp not satisfied within 1% of true value, check if tails over this region need a smaller threshold to cusp: ", a, " should be ", internal_options['Z'][int((i+1)/npts - 1)])
#            #raise Exception("cusp not satisfied within 1% of true value, check if tails over this region need a smaller threshold to cusp: ", a, " should be ", internal_options['Z'][int((i+1)/4 - 1)])
#      #print("", slope, y_int,flush=True)
#
#    f.close()  
#    #for g in range(len(mocoeff)):
#      #mocoeff[g][m] = internal_options["mocoeff"][g][0]
#    #print("")
#    
#  return 1.0
#def do_cusp_test_hack(internal_options, step_size):
#
#  # get number of nuclei
#  nnuc = internal_options["nuclei"].size // 3
#
#  # get number of alpha electrons
#  nalp = internal_options["apos"].size // 3
#
#  # get number of threads
#  nthread = get_number_of_threads()
#  print("")
#  print("have %i threads for do_cusp_test_hack" % nthread)
#  print("")
#
#  nbat = len(internal_options["batches"])
#
#  num_iter = 1
#  for a in range(num_iter):
#    for j in range(1): #range(nbat):
#
#      internal_options["active_batch"] = j
#      #print("Current active electrons: ", internal_options["active_elec"])
#
#      # Prepare an array to hold the electron positions for each thread.
#      # To start we set this to the initial electron positions, but once we've taken
#      # samples we just keep each thread's most recent sample here.
#      elec_pos = np.zeros([nthread, 2, nalp, 3])
#      for i in range(nthread):
#        elec_pos[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
#        elec_pos[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])
#
#      #print("elec_pos", elec_pos, flush=True)
#
#      # prepare dictionary that holds the data arrays for the different accumulators
#      acc_names = ["AccumulatorCuspTestingKE"]
#      acc_dict = make_accumulator_dict(acc_names, internal_options)
#
#      # take the sample and accumulate the data we need
#      vmc_take_sample_for_nuc_cusp_testing(internal_options, acc_dict, elec_pos)
#
#      print("")
#      print("cusp testing accumulator array:")
#      slope = []
#      y_int = []
#      for i in range(acc_dict["AccumulatorCuspTestingKE"].size):
#        #print("%30.16e" % acc_dict["AccumulatorCuspTestingKE"][i])
#        if (i+1)%4 == 0:
#          #find line of best fit
#          #print([(1/(step_size * (i+1))) for i in range(4)])
#          a, b = np.polyfit([(1/(step_size * (i+1))) for i in range(4)], acc_dict["AccumulatorCuspTestingKE"][i-3:i+1], 1)
#          slope.append(a)
#          y_int.append(b)
#          if np.abs(a - internal_options['Z'][int((i+1)/4 - 1)]) < 0.01:
#            print("\tslope: ", "%3.4e" %a, "\t y-nt: ", b)
#            print("\treal:  ", internal_options['Z'][int((i+1)/4 - 1)])
#            print("==========")
#          else:
#            raise Exception("cusp not satisfied within 1% of true value, check if tails over this region need a smaller threshold to cusp: ", a, " should be ", internal_options['Z'][int((i+1)/4 - 1)])
#      #print("", slope, y_int,flush=True)
#      #print("dictionary input in slope test\n", internal_options, flush=True)
#      return slope #np.asarray(slope), np.asarray(y_int)

def do_orb_deriv_test(options_pkl):
  """ test to check orbital derivatives with finite difference

  params: options - internal_optiions pkl dictionary

  """
  with open(options_pkl, 'rb') as fp:
    options = pickle.load(fp)

  print("******************** IN do_orb_deriv_test() ********************",flush=True)

  nvp = options['nvp'] 
  mocoeff = options["mocoeff"].copy()   # reference C mat
  grad_test_options = options.copy()
  print("Initial C mat\n",mocoeff, flush=True)
  norb = mocoeff.shape[0]
  nalp = options["apos"].size // 3
  nthread = 1 
  print("")
  print("have %i threads for do_orb_deriv_test" % nthread, flush=True)
  print("")

  acc_names = []
  acc_dict = make_accumulator_dict(acc_names, grad_test_options, nvp, grad_test_options["nblock"])
  for key, value in acc_dict.items():
    print(key, value.shape)

  # Initial pos to compare against
  # Prepare an array to hold the electron positions for each thread.
  # To start we set this to the initial electron positions, but once we've taken
  # samples we just keep each thread's most recent sample here.
  elec_pos = np.zeros([nthread, 2, nalp, 3])
  for i in range(nthread):
    elec_pos[i, 0, :, :] = np.reshape(grad_test_options["apos"], [nalp,3])
    elec_pos[i, 1, :, :] = np.reshape(grad_test_options["bpos"], [nalp,3])

  # want only 1 samp
  grad_test_options["nsamp_per_block"] = 1 
  grad_test_options["nblock"] = 1 
  grad_test_options["nwarmup"] = 0 

  diff = 1e-6

  orbs_ref = np.zeros(2 * nalp * norb)  # column-major naxno array
  der1_ref = np.zeros((3,2 * nalp * norb))  # column-major naxno array for x, y, then z
  der2_ref = np.zeros((3,2 * nalp * norb))

  vmc_take_sample_for_orb_deriv_testing(grad_test_options, acc_dict, elec_pos, orbs_ref, der1_ref, der2_ref, np.array([diff]))

  print("Orbital eval simple_vmc.py\n",orbs_ref,flush=True)
  print("Derivatives 1 eval simple_vmc.py\n",der1_ref,flush=True)
  print("Derivatives 2 eval simple_vmc.py\n",der2_ref,flush=True)  
  x_alp_ref = (np.reshape(der1_ref[0][:norb*nalp],[norb,nalp]).transpose())
  x_bet_ref = (np.reshape(der1_ref[0][norb*nalp:],[norb,nalp]).transpose())
  y_alp_ref = (np.reshape(der1_ref[1][:norb*nalp],[norb,nalp]).transpose())
  y_bet_ref = (np.reshape(der1_ref[1][norb*nalp:],[norb,nalp]).transpose())
  z_alp_ref = (np.reshape(der1_ref[2][:norb*nalp],[norb,nalp]).transpose())
  z_bet_ref = (np.reshape(der1_ref[2][norb*nalp:],[norb,nalp]).transpose())

  print("\n---------------- Reference der1 values that should also be in LM_grad test (make sure this passes FD) -----------------\n", flush=True)
  print("\nx_alp_ref\n",x_alp_ref,flush=True) 
  print("\nx_bet_ref\n",x_bet_ref,flush=True)
  print("\ny_alp_ref\n",y_alp_ref,flush=True)
  print("\ny_bet_ref\n",y_bet_ref,flush=True)
  print("\nz_alp_ref\n",z_alp_ref,flush=True)
  print("\nz_bet_ref\n",z_bet_ref,flush=True)

  print("\n---------------- On to finite difference -----------------\n", flush=True)

  spin_ind = [0,1]
  elec_pos_diff = elec_pos.copy()
  der1 = np.zeros((3,2 * nalp * norb))  # column-major naxno array for x, y, then z
  der2 = np.zeros((3,2 * nalp * norb))

  der1_x_a_matrix = np.empty((0, norb))
  der1_y_a_matrix = np.empty((0, norb))
  der1_z_a_matrix = np.empty((0, norb))

  der1_x_b_matrix = np.empty((0, norb))
  der1_y_b_matrix = np.empty((0, norb))
  der1_z_b_matrix = np.empty((0, norb))
  
  for elec_ind in np.arange(nalp):
    orb_list = []
    for xyz_ind in [0,1,2]:
      for sgn in [1, -1]:
        orbs = np.zeros(2 * nalp * norb)  # column-major naxno array
        #print("elec pos Before (shape", elec_pos.shape,")\n",elec_pos, "\n",flush=True)

        # adding to 0th electron
        elec_pos_diff[0, spin_ind, elec_ind, xyz_ind] += sgn * diff      
        #print("elec pos After\n",elec_pos_diff, "\n",flush=True)

        vmc_take_sample_for_orb_deriv_testing(grad_test_options, acc_dict, elec_pos_diff, orbs, der1, der2, np.array([diff]))

        orb_list.append(orbs)
        elec_pos_diff[0, spin_ind, elec_ind, xyz_ind] = elec_pos[0, spin_ind, elec_ind, xyz_ind]     

    #print("All calculated orbs", orb_list, flush=True)
    der1_x_fd = ((orb_list[0] - orb_list[1]) / (2*diff))      
    der1_y_fd = ((orb_list[2] - orb_list[3]) / (2*diff))      
    der1_z_fd = ((orb_list[4] - orb_list[5]) / (2*diff))      

    x_alp = (np.reshape(np.round(der1_ref[0][:norb*nalp]-der1_x_fd[:norb*nalp],10),[norb,nalp]).transpose())
    x_bet = (np.reshape(np.round(der1_ref[0][norb*nalp:]-der1_x_fd[norb*nalp:],10),[norb,nalp]).transpose())
    y_alp = (np.reshape(np.round(der1_ref[1][:norb*nalp]-der1_y_fd[:norb*nalp],10),[norb,nalp]).transpose())
    y_bet = (np.reshape(np.round(der1_ref[1][norb*nalp:]-der1_y_fd[norb*nalp:],10),[norb,nalp]).transpose())
    z_alp = (np.reshape(np.round(der1_ref[2][:norb*nalp]-der1_z_fd[:norb*nalp],10),[norb,nalp]).transpose())
    z_bet = (np.reshape(np.round(der1_ref[2][norb*nalp:]-der1_z_fd[norb*nalp:],10),[norb,nalp]).transpose())

    #print(x_alp,flush=True)
    #print(x_bet,flush=True)
    #print(y_alp,flush=True)
    #print(y_bet,flush=True)
    #print(z_alp,flush=True)
    #print(z_bet,flush=True)

    x_alp = x_alp[elec_ind] 
    x_bet = x_bet[elec_ind]
    y_alp = y_alp[elec_ind]
    y_bet = y_bet[elec_ind]
    z_alp = z_alp[elec_ind]
    z_bet = z_bet[elec_ind]

    der1_x_a_matrix = np.vstack([der1_x_a_matrix, x_alp])
    der1_y_a_matrix = np.vstack([der1_y_a_matrix, y_alp])
    der1_z_a_matrix = np.vstack([der1_z_a_matrix, z_alp])

    der1_x_b_matrix = np.vstack([der1_x_b_matrix, x_bet])
    der1_y_b_matrix = np.vstack([der1_y_b_matrix, y_bet])
    der1_z_b_matrix = np.vstack([der1_z_b_matrix, z_bet])

  print()
  print("Reference der1", der1_ref, flush=True)
  print()
  print("Finite difference",flush=True)
  print("x:", #der1_x_fd, 
          "\ndiff: alpha\n", der1_x_a_matrix,
          "\nbeta\n", der1_x_b_matrix, flush=True)
  print()
  print("y:", #der1_y_fd, 
          "\ndiff: alpha\n", der1_y_a_matrix,
          "\nbeta\n", der1_y_b_matrix, flush=True)
  print()
  print("z:", #der1_z_fd, 
          "\ndiff: alpha\n", der1_z_a_matrix,
          "\nbeta\n", der1_z_b_matrix, flush=True)

  #print()
  #print("Reference der2 sum", np.sum(der2_ref, axis=0), flush=True)

def do_absolute_energy_multipole(internal_options):

  # get number of nuclei
  nnuc = internal_options["nuclei"].size // 3

  # get number of alpha electrons
  nalp = internal_options["apos"].size // 3

  print("nuclei positions:\n", internal_options["nuclei"])

  # get number of threads
  nthread = get_number_of_threads()
  print("")
  print("have %i threads for do_absolute_energy" % nthread)
  print("")

  nbat = len(internal_options["batches"])

  num_iter = 1

  print("----------------------------------")
  for a in range(num_iter):
    print("Iter. #:", a+1)
    print("----------------------------------")

    print("Aos: ", internal_options["jAos"])
    print("Ass: ", internal_options["jAss"])
    
    delta_Ass = 0
    delta_Aos = 0
    total_total_e = 0

    # build discrete charge distributions for the multipole expansion
    for j in range(nbat):
    
      internal_options["active_batch"] = j
      internal_options["num_active"] = len(internal_options["batches"][j])
      print("Number of active electrons for this batch:", internal_options["num_active"])

      internal_options["m_origin_cur"] = internal_options["m_origin"][j]
      print("Current multipole origin for this batch:\n", internal_options["m_origin_cur"])

    # Prepare an array to hold the electron positions for each thread.
    # To start we set this to the initial electron positions, but once we've taken
    # samples we just keep each thread's most recent sample here.
      elec_pos_m = np.zeros([nthread, 2, nalp, 3])
      for i in range(nthread):
        elec_pos_m[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
        elec_pos_m[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])

      # prepare dictionary that holds the data arrays for the different accumulators
      acc_names_multipole = ["AccumulatorPosXi", "AccumulatorPosXiXj"] 
      #acc_names_multipole = ["AccumulatorPos", "AccumulatorPosXi", "AccumulatorPosXiXj"] 
      #acc_names_multipole = ["AccumulatorMonopole", "AccumulatorDipole", "AccumulatorQuad", "AccumulatorPos"] 
      acc_dict_multipole = make_accumulator_dict(acc_names_multipole, internal_options)

      # take the sample and accumulate the data we need to build multipoles
      vmc_take_samples_detail_multipole(internal_options, acc_dict_multipole, elec_pos_m)

      # calculate multipoles for this batch of active electrons
      calculate_multipole_for_batch(j, internal_options, acc_dict_multipole)

#      # Build multipole moments
#
#      # monopole
#      for i in range(n):
#        internal_options["monopole"][0][j] += -1.0
#
#      # dipole
#      for k in range(3):
#        for i in range(n*s*b):
#          internal_options["dipole"][k][j] += -1.0 * (acc_dict_multipole["AccumulatorPos"][i][k] - internal_options["m_origin"][j][k]) / (n*s*b)
#
#      # quadrupole
#      for i in range(n*s*b):
#        for k in range(3):
#          for l in range(3):
#            xk = acc_dict_multipole["AccumulatorPos"][i][k] - internal_options["m_origin"][j][k]
#            xl = acc_dict_multipole["AccumulatorPos"][i][l] - internal_options["m_origin"][j][l]
#            internal_options["quad"][k+3*l][j] += -1.0 * 3.0 * xk * xl / (n*s*b)
#            if k == l:
#              x = acc_dict_multipole["AccumulatorPos"][i][0] - internal_options["m_origin"][j][0]
#              y = acc_dict_multipole["AccumulatorPos"][i][1] - internal_options["m_origin"][j][1]
#              z = acc_dict_multipole["AccumulatorPos"][i][2] - internal_options["m_origin"][j][2]
#              r = np.sqrt(x**2.0 + y**2.0 + z**2.0)
#              internal_options["quad"][k+3*l][j] += r * r / (n*s*b)
#
#      print("")
#      print("%11s %20s %20s %20s" % ("k and l", "push-back-quad", "on-the-fly-quad", "difference"))
#      for k in range(3):
#        for l in range(3):
#          print("%5i %5i %20.12e %20.12e %20.12e"
#                % (k, l, internal_options["quad"][k+3*l][j], quad[k,l], quad[k,l] - internal_options["quad"][k+3*l][j]))
#      print("")


      #for i in range(3):
        #internal_options["dipole"][i][j] = acc_dict_multipole["AccumulatorDipole"][0][i]
      #for k in range(9):
        #internal_options["quad"][k][j] = acc_dict_multipole["AccumulatorQuad"][0][k]
        #internal_options["dipole"][i][j] = dip[i]
      print()
    internal_options["monopole"][:,[1,0]] = internal_options["monopole"][:, [0,1]]
    internal_options["dipole"][:,[1,0]] = internal_options["dipole"][:, [0,1]]
    internal_options["quad"][:,[1,0]] = internal_options["quad"][:, [0,1]]
    print("Did transfer to Python dict work for monopole:\n", internal_options["monopole"])
    print("Did transfer to Python dict work for dipole:\n", internal_options["dipole"])
    print("Did transfer to Python dict work for quad:\n", internal_options["quad"])

    # Switch multipole origins for actual sampling

    internal_options["m_origin"][[0,1]] = internal_options["m_origin"][[1,0]]
    print("Did swapping m_origins work:\n", internal_options["m_origin"])
 
    nuc_o_0 = internal_options["nuc_origin"][0]
    nuc_o_1 = internal_options["nuc_origin"][1]

    internal_options["nuc_origin"][0] = 2
    internal_options["nuc_origin"][1] = 0
    print("Did swapping nuc_origins work:\n", internal_options["nuc_origin"])

    internal_options["active_batch"] = 0
    
    for j in range(nbat):

      total_e = 0
      total_e_ass = 0
      total_ass = 0
      total_e_aos = 0
      total_aos = 0
      
      internal_options["active_batch"] = j
      #print("Current active electrons: ", internal_options["active_elec"])
      internal_options["m_origin_cur"] = internal_options["m_origin"][j]
      print("Current multipole origin for this batch:\n", internal_options["m_origin_cur"])

    # Prepare an array to hold the electron positions for each thread.
    # To start we set this to the initial electron positions, but once we've taken
    # samples we just keep each thread's most recent sample here.
      elec_pos = np.zeros([nthread, 2, nalp, 3])
      for i in range(nthread):
          elec_pos[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
          elec_pos[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])

      # prepare dictionary that holds the data arrays for the different accumulators
      acc_names = ["AccumulatorVGR", "AccumulatorKE", "AccumulatorEEE", "AccumulatorENE", "AccumulatorNNE", "AccumulatorLocalE", "AccumulatorAss", "AccumulatorLocalEAss", "AccumulatorAos", "AccumulatorLocalEAos"] #, "AccumulatorPvP0"]
      acc_dict = make_accumulator_dict(acc_names, internal_options)

      # take the sample and accumulaGte the data we need
      vmc_take_samples_detail(internal_options, acc_dict, elec_pos)

      # collect terms for steepest descent pieces
      total_e += np.mean(acc_dict["AccumulatorLocalE"], axis=0)
      total_e_ass += np.mean(acc_dict["AccumulatorLocalEAss"])
      total_ass += np.mean(acc_dict["AccumulatorAss"])
      total_e_aos += np.mean(acc_dict["AccumulatorLocalEAos"])
      total_aos += np.mean(acc_dict["AccumulatorAos"])
      
      # process the data we accumulated during sampling and print the results
      stats_processors = []
      for ad in [ acc_dict ]:
        acc_data_total_e = ad["AccumulatorKE"] + ad["AccumulatorEEE"] + ad["AccumulatorENE"] + ad["AccumulatorNNE"]
        stats_processors.append( [
          RegionStatsProcessor(     'kinetic', ad['AccumulatorVGR'], ad[ "AccumulatorKE"],  True),
          RegionStatsProcessor(   'elec-elec', ad['AccumulatorVGR'], ad["AccumulatorEEE"],  True),
          RegionStatsProcessor(    'elec-nuc', ad['AccumulatorVGR'], ad["AccumulatorENE"],  True),
          RegionStatsProcessor(     'nuc-nuc', ad['AccumulatorVGR'], ad["AccumulatorNNE"], False),
          RegionStatsProcessor('total energy', ad['AccumulatorVGR'],     acc_data_total_e,  True),
        ] )
        #print("Accumulator Ass: ", ad["AccumulatorAss"])
        #print("Accumulator Aos: ", ad["AccumulatorAos"])
        #print("Accumulator local e ass: ", ad["AccumulatorLocalEAss"])
        #print("Accumulator local e aos: ", ad["AccumulatorLocalEAos"])
        #print(len(ad["AccumulatorKE"]))
        #print("Total: ", acc_data_total_e)
        print("")
        print("absolute quantities for batch", j)
        print("")
        print(stats_processors[-1][0].table_header())
        print(stats_processors[-1][0].table_dashes())
        for rsp in stats_processors[-1]:
          print(rsp.table_row())

      total_total_e += total_e
      delta_Ass += 0.1 * (2 * total_e_ass - 2 * total_e * total_ass)
      delta_Aos += 0.1 * (2 * total_e_aos - 2 * total_e * total_aos)
    print("Change in parameter Ass: ", delta_Ass)
    print("Change in parameter Aos: ", delta_Aos)
    print("Total E: ", total_total_e)

    internal_options["jAss"] -= delta_Ass
    internal_options["jAos"] -= delta_Aos
    print("----------------------------------", flush=True)
    #print("Accumulator KE: ", acc_dict["AccumulatorKE"])
    #print(len(acc_dict["AccumulatorKE"]))

#def do_absolute_energy_multipole(internal_options):
#
#  # get number of nuclei
#  nnuc = internal_options["nuclei"].size // 3
#
#  # get number of alpha electrons
#  nalp = internal_options["apos"].size // 3
#
#  print("nuclei positions:\n", internal_options["nuclei"])
#
#  # get number of threads
#  nthread = get_number_of_threads()
#  print("")
#  print("have %i threads for do_absolute_energy" % nthread)
#  print("")
#
#  nbat = len(internal_options["batches"])
#
#  num_iter = 1
#
#  print("----------------------------------")
#  for a in range(num_iter):
#    print("Iter. #:", a+1)
#    print("----------------------------------")
#
#    print("Aos: ", internal_options["jAos"])
#    print("Ass: ", internal_options["jAss"])
#    
#    delta_Ass = 0
#    delta_Aos = 0
#    total_total_e = 0
#
#    # build discrete charge distributions for the multipole expansion
#    for j in range(nbat):
#    
#      internal_options["active_batch"] = j
#
#      internal_options["m_origin_cur"] = internal_options["m_origin"][j]
#      print("Current multipole origin for this batch:\n", internal_options["m_origin_cur"])
#
#    # Prepare an array to hold the electron positions for each thread.
#    # To start we set this to the initial electron positions, but once we've taken
#    # samples we just keep each thread's most recent sample here.
#      elec_pos_m = np.zeros([nthread, 2, nalp, 3])
#      for i in range(nthread):
#        elec_pos_m[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
#        elec_pos_m[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])
#
#      # prepare dictionary that holds the data arrays for the different accumulators
#      acc_names_multipole = ["AccumulatorMonopole", "AccumulatorDipole", "AccumulatorQuad"] 
#      acc_dict_multipole = make_accumulator_dict(acc_names_multipole, internal_options)
#
#      # take the sample and accumulate the data we need
#      vmc_take_samples_detail_multipole(internal_options, acc_dict_multipole, elec_pos_m)
#      print("Monopole accumulator:\n", acc_dict_multipole["AccumulatorMonopole"])
#      print("Dipole accumulator:\n", acc_dict_multipole["AccumulatorDipole"])
#      print("Quadrupole accumulator:\n", acc_dict_multipole["AccumulatorQuad"])
#
#      mon = np.mean(acc_dict_multipole["AccumulatorMonopole"], axis=0)
#      dip = np.mean(acc_dict_multipole["AccumulatorDipole"], axis=0)
#      #quad = np.mean(acc_dict_multipole["AccumulatorQuad"], axis=0)
#      #print("Quad average:\n", quad)
#
#      internal_options["monopole"][0][j] = mon
#     
#      for i in range(3):
#        internal_options["dipole"][i][j] = acc_dict_multipole["AccumulatorDipole"][0][i]
#      for k in range(9):
#        internal_options["quad"][k][j] = acc_dict_multipole["AccumulatorQuad"][0][k]
#        #internal_options["dipole"][i][j] = dip[i]
#      print()
#    internal_options["monopole"][:,[1,0]] = internal_options["monopole"][:, [0,1]]
#    internal_options["dipole"][:,[1,0]] = internal_options["dipole"][:, [0,1]]
#    internal_options["quad"][:,[1,0]] = internal_options["quad"][:, [0,1]]
#    print("Did transfer to Python dict work:\n", internal_options["monopole"])
#    print("Did transfer to Python dict work:\n", internal_options["dipole"])
#    print("Did transfer to Python dict work:\n", internal_options["quad"])
#
#    # Switch multipole origins for actual sampling
#
#    internal_options["m_origin"][[0,1]] = internal_options["m_origin"][[1,0]]
#    print("Did swapping m_origins work:\n", internal_options["m_origin"])
# 
#    nuc_o_0 = internal_options["nuc_origin"][0]
#    nuc_o_1 = internal_options["nuc_origin"][1]
#
#    internal_options["nuc_origin"][0] = 2
#    internal_options["nuc_origin"][1] = 0
#    print("Did swapping nuc_origins work:\n", internal_options["nuc_origin"])
#
#    internal_options["active_batch"] = 0
#    
#    for j in range(nbat):
#
#      total_e = 0
#      total_e_ass = 0
#      total_ass = 0
#      total_e_aos = 0
#      total_aos = 0
#      
#      internal_options["active_batch"] = j
#      #print("Current active electrons: ", internal_options["active_elec"])
#      internal_options["m_origin_cur"] = internal_options["m_origin"][j]
#      print("Current multipole origin for this batch:\n", internal_options["m_origin_cur"])
#
#    # Prepare an array to hold the electron positions for each thread.
#    # To start we set this to the initial electron positions, but once we've taken
#    # samples we just keep each thread's most recent sample here.
#      elec_pos = np.zeros([nthread, 2, nalp, 3])
#      for i in range(nthread):
#          elec_pos[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
#          elec_pos[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])
#
#      # prepare dictionary that holds the data arrays for the different accumulators
#      acc_names = ["AccumulatorVGR", "AccumulatorKE", "AccumulatorEEE", "AccumulatorENE", "AccumulatorNNE", "AccumulatorLocalE", "AccumulatorAss", "AccumulatorLocalEAss", "AccumulatorAos", "AccumulatorLocalEAos"] # + "AccumulatorLM"
#      acc_dict = make_accumulator_dict(acc_names, internal_options)
#
#      # take the sample and accumulate the data we need
#      vmc_take_samples_detail(internal_options, acc_dict, elec_pos)
#
#      # collect terms for steepest descent pieces
#      total_e += np.mean(acc_dict["AccumulatorLocalE"], axis=0)
#      total_e_ass += np.mean(acc_dict["AccumulatorLocalEAss"])
#      total_ass += np.mean(acc_dict["AccumulatorAss"])
#      total_e_aos += np.mean(acc_dict["AccumulatorLocalEAos"])
#      total_aos += np.mean(acc_dict["AccumulatorAos"])
#      
#      # process the data we accumulated during sampling and print the results
#      stats_processors = []
#      for ad in [ acc_dict ]:
#        acc_data_total_e = ad["AccumulatorKE"] + ad["AccumulatorEEE"] + ad["AccumulatorENE"] + ad["AccumulatorNNE"]
#        stats_processors.append( [
#          RegionStatsProcessor(     'kinetic', ad['AccumulatorVGR'], ad[ "AccumulatorKE"],  True),
#          RegionStatsProcessor(   'elec-elec', ad['AccumulatorVGR'], ad["AccumulatorEEE"],  True),
#          RegionStatsProcessor(    'elec-nuc', ad['AccumulatorVGR'], ad["AccumulatorENE"],  True),
#          RegionStatsProcessor(     'nuc-nuc', ad['AccumulatorVGR'], ad["AccumulatorNNE"], False),
#          RegionStatsProcessor('total energy', ad['AccumulatorVGR'],     acc_data_total_e,  True),
#        ] )
#        #print("Accumulator Ass: ", ad["AccumulatorAss"])
#        #print("Accumulator Aos: ", ad["AccumulatorAos"])
#        #print("Accumulator local e ass: ", ad["AccumulatorLocalEAss"])
#        #print("Accumulator local e aos: ", ad["AccumulatorLocalEAos"])
#        #print(len(ad["AccumulatorKE"]))
#        #print("Total: ", acc_data_total_e)
#        print("")
#        print("absolute quantities for batch", j)
#        print("")
#        print(stats_processors[-1][0].table_header())
#        print(stats_processors[-1][0].table_dashes())
#        for rsp in stats_processors[-1]:
#          print(rsp.table_row())
#
#      total_total_e += total_e
#      delta_Ass += 0.1 * (2 * total_e_ass - 2 * total_e * total_ass)
#      delta_Aos += 0.1 * (2 * total_e_aos - 2 * total_e * total_aos)
#    print("Change in parameter Ass: ", delta_Ass)
#    print("Change in parameter Aos: ", delta_Aos)
#    print("Total E: ", total_total_e)
#
#    internal_options["jAss"] -= delta_Ass
#    internal_options["jAos"] -= delta_Aos
#    print("----------------------------------", flush=True)
#    #print("Accumulator KE: ", acc_dict["AccumulatorKE"])
#    #print(len(acc_dict["AccumulatorKE"]))

def do_relative_energy_vanilla(internal_options, internal_options_s):

  # get number of nuclei
  nnuc = internal_options["nuclei"].size // 3

  # get number of alpha electrons
  nalp = internal_options["apos"].size // 3

  # get number of threads
  nthread = get_number_of_threads()
  print("")
  print("have %i threads for do_relative_energy" % nthread)
  print("", flush=True)

  nbat = len(internal_options["batches"])

  num_iter = 1

  print("----------------------------------")
  for a in range(num_iter):
    print("Iter. #:", a+1)
    print("----------------------------------")

    total_total_e = 0
    total_total_e_s = 0
    
    for j in range(nbat):

      total_e = 0
      total_e_s = 0
      
      internal_options["active_batch"] = j
      internal_options_s["active_batch"] = j
      #print("Current active electrons: ", internal_options["active_elec"])
      internal_options["m_origin_cur"] = internal_options["m_origin"][j]
      #print("Current multipole origin for this batch:\n", internal_options["m_origin_cur"])
      internal_options_s["m_origin_cur"] = internal_options_s["m_origin"][j]
      #print("Current multipole origin for this batch:\n", internal_options_s["m_origin_cur"])

    # Prepare an array to hold the electron positions for each thread.
    # To start we set this to the initial electron positions, but once we've taken
    # samples we just keep each thread's most recent sample here.
      elec_pos = np.zeros([nthread, 2, nalp, 3])
      for i in range(nthread):
          elec_pos[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
          elec_pos[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])

      # prepare dictionary that holds the data arrays for the different accumulators
      acc_names = ["AccumulatorVGR", "AccumulatorKE", "AccumulatorEEE", "AccumulatorENE", "AccumulatorNNE", "AccumulatorLocalE"] #, "AccumulatorPvP0"]
      acc_names2 = ["AccumulatorVGR", "AccumulatorKE", "AccumulatorEEE", "AccumulatorENE", "AccumulatorNNE", "AccumulatorLocalE", "AccumulatorLocalEPsiRatio", "AccumulatorPsiRatio"] #, "AccumulatorPvP0"]
      acc_dict1 = make_accumulator_dict(acc_names, internal_options)
      acc_dict2 = make_accumulator_dict(acc_names2, internal_options_s)
      
      # take the sample and accumulate the data we need
      vmc_take_relative_samples(internal_options, internal_options_s, acc_dict1, acc_dict2, elec_pos)

      total_e += np.mean(acc_dict1["AccumulatorLocalE"], axis=0)
      #total_e_s += np.mean(acc_dict2["AccumulatorLocalE"], axis=0)
   
      #print("AccumulatorPsiRatio for acc_dict2:\n", acc_dict2["AccumulatorPsiRatio"])
      #print("AccumulatorLocalEPsiRatio for acc_dict2:\n", acc_dict2["AccumulatorLocalEPsiRatio"])
      #one_over_psi_ratio = 1.0/np.mean(acc_dict2["AccumulatorPsiRatio"], axis=0)
      #print("One over psi ratio:\n", one_over_psi_ratio)
      #local_e_s = one_over_psi_ratio * acc_dict2["AccumulatorLocalEPsiRatio"] 

      #deltaE = local_e_s - acc_dict1["AccumulatorLocalE"]
      #print("Delta E array:\n", deltaE)
      #print("Average delta E:\n", np.mean(deltaE, axis=0))
      #uncertainty = np.std(deltaE, axis=0) / np.sqrt(24)
      #print("uncertainty:\n", uncertainty)
      

      #local_e_s = np.divide(acc_dict2["AccumulatorLocalEPsiRatio"], acc_dict2["AccumulatorPsiRatio"])
      #print("Weighted Local Energy for Geom. 2:\n", local_e_s)
      #print("Mean of weighted local energy for geom. 2:\n", np.mean(local_e_s, axis=0))
      #print("Uncertainty:\n", np.std(local_e_s, axis=0)/np.sqrt(32))
      #print("Local energy for geom. 1:\n", acc_dict1["AccumulatorLocalE"])
      #deltaE = local_e_s - acc_dict1["AccumulatorLocalE"]
      #print("Delta E for the stretch in correlated sampling:\n", deltaE)
      #print()
      #print("Mean delta:\n", np.mean(deltaE, axis=0))
      #print("Uncertainty:\n", np.std(deltaE, axis=0)/np.sqrt(24))

      # Do statistics for the weighted local energy estimator for the secondary geometry
      les_data = acc_dict2["AccumulatorLocalEPsiRatio"]
      psi_rat_data = acc_dict2["AccumulatorPsiRatio"]
      le_data = acc_dict1["AccumulatorLocalE"]
      print()
      print("AccumulatorLocalEPsiRatio array for geom. 2:\n", les_data)
      print()
      print("AccumulatorPsiRatio array for geom. 2:\n", psi_rat_data)
      print()
      print("AccumulatorLocalE array for geom. 1:\n", le_data)
      print()
      avg_les = np.mean(les_data, axis=0)
      #print("Average local e s for geom. 2:\n",avg_les) 
      #print()
      std_les = np.std(les_data, axis=0)
      avg_psi_rat = np.mean(psi_rat_data, axis=0)
      #print("Average psi rat for geom. 2:\n",avg_psi_rat) 
      #print()
      std_psi_rat = np.std(psi_rat_data, axis=0)
      sqrtN = np.sqrt( 1.0 * psi_rat_data.size )
      avg_weighted_les = avg_les / avg_psi_rat
      print("Average weighted local energy of geom. 2:\n", avg_weighted_les)
      #cov_les_psi = np.mean(   ( les_data - np.reshape(avg_les, [1,-1]) ) * (psi_rat_data - np.reshape(avg_psi_rat, [-1,1])), axis=0)
      cov_les_psi = np.mean(   ( les_data - np.reshape(avg_les, [1,-1]) ) * np.reshape(psi_rat_data - avg_psi_rat, [-1,1]), axis=0)
      print()
      #print("Covariance for LES and Psi Rat:\n", cov_les_psi)
      cov_les_psi_3 = np.mean( np.multiply(les_data, psi_rat_data),axis=0) - avg_les * avg_psi_rat
      #print("Covariance another way:\n", cov_les_psi_3)
      cov_les_psi_2 = np.mean( ( les_data - avg_les ) * np.reshape(psi_rat_data - avg_psi_rat, [-1,1]), axis=0)
      std_les_over_psi = np.sqrt(    ( avg_les / avg_psi_rat ) * ( avg_les / avg_psi_rat ) * (   ( std_les / avg_les ) * ( std_les / avg_les )+ ( std_psi_rat / avg_psi_rat ) * ( std_psi_rat / avg_psi_rat ) - 2.0 * cov_les_psi_3 / ( avg_les * avg_psi_rat ) ) )
      weighted_les_unc = std_les_over_psi / sqrtN
      print("Uncertainty of weighted local energy of geom. 2:\n", weighted_les_unc) 

      # Do statistics for the relative energy between geom. 1 and geom.2 
      # Var(X/Y - Z) 
      avg_le = np.mean(le_data, axis=0)
      std_le = np.std(le_data, axis=0)
      rel_e = avg_weighted_les - avg_le
      #print()
      #print("Average delta:\n", rel_e)                 
     
      avg_X = avg_les
      avg_Y = avg_psi_rat
      avg_Z = avg_le
      std_X = std_les
      std_Y = std_psi_rat
      std_Z = std_le
      cov_XY = np.mean( np.multiply(les_data, psi_rat_data),axis=0) - avg_les * avg_psi_rat
      cov_XZ = np.mean( np.multiply(les_data, le_data),axis=0) - avg_les * avg_le
      cov_YZ = np.mean( np.multiply(le_data, psi_rat_data),axis=0) - avg_le * avg_psi_rat
      diff_avg = avg_X / avg_Y - avg_Z
      var_XYZ = np.sqrt( (std_X / avg_Y)**2.0 + (avg_X * std_Y / avg_Y**2.0)**2.0 + std_Z**2.0 - 2 * avg_X * cov_XY / avg_Y**3.0 - 2 * cov_XZ / avg_Y + 2 * avg_X * cov_YZ / avg_Y**2.0 )
      diff_unc = var_XYZ / sqrtN
      #print()
      #print("Uncertainty of the difference:\n", diff_unc)

      print("")
      print("delta-and-uncertainty: %20.12f +/- %20.12f" % (rel_e, diff_unc))

      #avg_A = avg_weighted_les
      #avg_B = avg_le
      #diff_avg = avg_A - avg_B
      #std_A = weighted_les_unc * sqrtN
      #std_B = std_le
      #cov_AB = np.mean( (np.divide(les_data, psi_rat_data) - np.reshape(avg_A, [1, -1])) * (le_data - np.reshape(avg_B, [1,-1])), axis=0)
      #cov_AB = np.mean( np.multiply( np.divide(les_data, avg_psi_rat), le_data), axis=0) - avg_A * avg_B
      #cov_AB_2 = np.mean(    np.multiply( (les_data / avg_psi_rat - avg_A), (le_data - avg_B))   , axis=0)
      #print("Covariance between <f>/<g> - <h>:\n", cov_AB)
      #print("Covariance between <f>/<g> - <h> another way:\n", cov_AB_2)
      #sqrt_1 = std_A * std_A + std_B * std_B
      #print("sqrt_1:\n", sqrt_1)
      #sqrt_2 = 2.0 * cov_AB
      #print("sqrt_2:\n", sqrt_2)
      #std_A_minus_B = np.sqrt( std_A * std_A + std_B * std_B - 2.0 * cov_AB )
      #diff_unc = std_A_minus_B / sqrtN
      #print()
      #print("Average delta again:\n", diff_avg)
      #print()
      #print("Uncertainty of delta:\n", diff_unc)

      #avg_A = np.mean(num_matrix, axis=0)
      #std_A = np.std(num_matrix, axis=0)
      #avg_B = np.mean(vgr_vec)
      #std_B = np.std(vgr_vec)
      #sqrtN = np.sqrt( 1.0 * vgr_vec.size )
      #ratio_avg = avg_A / avg_B
      #if do_ratio_unc:
      #cov_AB = np.mean(   ( num_matrix - np.reshape(avg_A, [1,-1]) ) * np.reshape(vgr_vec - avg_B, [-1,1]), axis=0)
      #std_A_over_B = np.sqrt(    ( avg_A / avg_B ) * ( avg_A / avg_B ) * (   ( std_A / avg_A ) * ( std_A / avg_A )+ ( std_B / avg_B ) * ( std_B / avg_B ) - 2.0 * cov_AB / ( avg_A * avg_B ) ) )
      #ratio_unc = std_A_over_B / sqrtN
      #else:
      #ratio_unc = np.zeros_like(ratio_avg)
      #vgr_avg = avg_B
      #vgr_unc = std_B / sqrtN
      #return ratio_avg, ratio_unc, vgr_avg, vgr_unc


      #def region_ratio_difference_stats(num_matrix_1, vgr_vec_1, num_matrix_2, vgr_vec_2, do_diff_unc=True, print_covAB=True):
      #sqrtN = np.sqrt( 1.0 * vgr_vec_1.size )
      #ratio_avg_1, ratio_unc_1, vgr_avg_1, vgr_unc_1 = region_ratio_stats(num_matrix_1, vgr_vec_1, do_diff_unc)
      #ratio_avg_2, ratio_unc_2, vgr_avg_2, vgr_unc_2 = region_ratio_stats(num_matrix_2, vgr_vec_2, do_diff_unc)
      #avg_A = ratio_avg_1
      #avg_B = ratio_avg_2
      #diff_avg = avg_A - avg_B
      #if do_diff_unc:
      #std_A = ratio_unc_1 * sqrtN
      #std_B = ratio_unc_2 * sqrtN
    # save the value guiding ratio averages (they are the same for all regions, so we have one per sampling block)
      # process the data we accumulated during sampling and print the results
      stats_processors = []
      for ad in [ acc_dict1, acc_dict2 ]:
        acc_data_total_e = ad["AccumulatorKE"] + ad["AccumulatorEEE"] + ad["AccumulatorENE"] + ad["AccumulatorNNE"]
        stats_processors.append( [
          RegionStatsProcessor(     'kinetic', ad['AccumulatorVGR'], ad[ "AccumulatorKE"],  True),
          RegionStatsProcessor(   'elec-elec', ad['AccumulatorVGR'], ad["AccumulatorEEE"],  True),
          RegionStatsProcessor(    'elec-nuc', ad['AccumulatorVGR'], ad["AccumulatorENE"],  True),
          RegionStatsProcessor(     'nuc-nuc', ad['AccumulatorVGR'], ad["AccumulatorNNE"], False),
          RegionStatsProcessor('total energy', ad['AccumulatorVGR'],     acc_data_total_e,  True),
        ] )
        print("")
        print("absolute quantities for batch", j)
        print("")
        print(stats_processors[-1][0].table_header())
        print(stats_processors[-1][0].table_dashes())
        for rsp in stats_processors[-1]:
          print(rsp.table_row())

      total_total_e += total_e
      total_total_e_s += total_e_s
    print("Total E geom.1: ", total_total_e)
    #print("Total E geom.2: ", total_total_e_s)

    #print("----------------------------------")
    #print("AccumulatorPsiRatio for geom.2:\n", acc_dict2["AccumulatorPsiRatio"])
    #print("AccumulatorLocalEPsiRatio for geom.2:\n", acc_dict2["AccumulatorLocalEPsiRatio"])

def do_relative_energy(internal_options, internal_options_s):

  # get number of nuclei
  nnuc = internal_options["nuclei"].size // 3

  # get number of alpha electrons
  nalp = internal_options["apos"].size // 3

  # get number of threads
  nthread = get_number_of_threads()
  print("")
  print("have %i threads for do_relative_energy" % nthread)
  print("", flush=True)

  nbat = len(internal_options["batches"])

  num_iter = 1

  print("----------------------------------")
  for a in range(num_iter):
    print("Iter. #:", a+1)
    print("----------------------------------")

    total_total_e = 0
    total_total_e_s = 0
    
    # build discrete charge distributions for the multipole expansion
    for j in range(nbat):
    
      internal_options["active_batch"] = j
      internal_options_s["active_batch"] = j
      internal_options["num_active"] = len(internal_options["batches"][j])
      print("Number of active electrons for this batch:", internal_options["num_active"])
      internal_options_s["num_active"] = len(internal_options_s["batches"][j])
      print("Number of active electrons for this batch:", internal_options_s["num_active"])

      internal_options["m_origin_cur"] = internal_options["m_origin"][j]
      print("Current multipole origin for this batch:\n", internal_options["m_origin_cur"])

    # Prepare an array to hold the electron positions for each thread.
    # To start we set this to the initial electron positions, but once we've taken
    # samples we just keep each thread's most recent sample here.
      elec_pos_m = np.zeros([nthread, 2, nalp, 3])
      for i in range(nthread):
        elec_pos_m[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
        elec_pos_m[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])

      # prepare dictionary that holds the data arrays for the different accumulators
      acc_names_multipole = ["AccumulatorPosXi", "AccumulatorPosXiXj"] 
      #acc_names_multipole = ["AccumulatorPos"] 
      #acc_names_multipole = ["AccumulatorMonopole", "AccumulatorDipole", "AccumulatorQuad", "AccumulatorPos"] 
      acc_dict_multipole1 = make_accumulator_dict(acc_names_multipole, internal_options)
      acc_dict_multipole2 = make_accumulator_dict(acc_names_multipole, internal_options_s)

      # take the sample and accumulate the data we need
      vmc_take_relative_samples_multipole(internal_options, internal_options_s, acc_dict_multipole1, acc_dict_multipole2, elec_pos_m)

      # calculate multipoles for this batch of active electrons
      calculate_multipole_for_batch(j, internal_options,   acc_dict_multipole1)
      calculate_multipole_for_batch(j, internal_options_s, acc_dict_multipole2)

#      print("Position accumulator fpr geom. 1:\n", acc_dict_multipole1["AccumulatorPos"])
#      print("Position accumulator fpr geom. 2:\n", acc_dict_multipole2["AccumulatorPos"])
#      n = internal_options["num_active"]
#      s = internal_options["nsamp_per_block"]
#      b = internal_options["nblock"]
#      centroid = np.sum(acc_dict_multipole1["AccumulatorPos"], axis=0) / (n * s * b)
#      centroid_s = np.sum(acc_dict_multipole2["AccumulatorPos"], axis=0) / (n * s * b)
#      print("Centroid of charge distribution for geom. 1:\n", centroid)
#      print("Centroid of charge distribution for geom. 2:\n", centroid_s)
#      
#      internal_options["m_origin"][j] = centroid
#      internal_options_s["m_origin"][j] = centroid_s
#      #print("Monopole accumulator:\n", acc_dict_multipole["AccumulatorMonopole"])
#      #print("Dipole accumulator for geom. 1:\n", acc_dict_multipole1["AccumulatorDipole"])
#      #print("Dipole accumulator for geom. 2:\n", acc_dict_multipole2["AccumulatorDipole"])
#      #print("Quadrupole accumulator:\n", acc_dict_multipole["AccumulatorQuad"])
#
#      # Build multipole moments
#
#      # monopole
#      for i in range(n):
#        internal_options["monopole"][0][j] += -1.0
#        internal_options_s["monopole"][0][j] += -1.0
#
#      # dipole
#      for k in range(3):
#        for i in range(n*s*b):
#          internal_options["dipole"][k][j] += -1.0 * (acc_dict_multipole1["AccumulatorPos"][i][k] - internal_options["m_origin"][j][k]) / (n*s*b)
#          internal_options_s["dipole"][k][j] += -1.0 * (acc_dict_multipole2["AccumulatorPos"][i][k] - internal_options_s["m_origin"][j][k]) / (n*s*b)
#
#      # quadrupole
#      for i in range(n*s*b):
#        for k in range(3):
#          for l in range(3):
#            xk = acc_dict_multipole1["AccumulatorPos"][i][k] - internal_options["m_origin"][j][k]
#            xl = acc_dict_multipole1["AccumulatorPos"][i][l] - internal_options["m_origin"][j][l]
#            internal_options["quad"][k+3*l][j] += -1.0 * 3.0 * xk * xl / (n*s*b)
#            
#            xk_s = acc_dict_multipole2["AccumulatorPos"][i][k] - internal_options_s["m_origin"][j][k]
#            xl_s = acc_dict_multipole2["AccumulatorPos"][i][l] - internal_options_s["m_origin"][j][l]
#            internal_options_s["quad"][k+3*l][j] += -1.0 * 3.0 * xk_s * xl_s / (n*s*b)
#            if k == l:
#              x = acc_dict_multipole1["AccumulatorPos"][i][0] - internal_options["m_origin"][j][0]
#              y = acc_dict_multipole1["AccumulatorPos"][i][1] - internal_options["m_origin"][j][1]
#              z = acc_dict_multipole1["AccumulatorPos"][i][2] - internal_options["m_origin"][j][2]
#              r = np.sqrt(x**2.0 + y**2.0 + z**2.0)
#              internal_options["quad"][k+3*l][j] += r * r / (n*s*b)
#
#              x_s = acc_dict_multipole2["AccumulatorPos"][i][0] - internal_options_s["m_origin"][j][0]
#              y_s = acc_dict_multipole2["AccumulatorPos"][i][1] - internal_options_s["m_origin"][j][1]
#              z_s = acc_dict_multipole2["AccumulatorPos"][i][2] - internal_options_s["m_origin"][j][2]
#              r_s = np.sqrt(x_s**2.0 + y_s**2.0 + z_s**2.0)
#              internal_options_s["quad"][k+3*l][j] += r_s * r_s / (n*s*b)
#      print()

    internal_options["monopole"][:,[1,0]] = internal_options["monopole"][:, [0,1]]
    internal_options["dipole"][:,[1,0]] = internal_options["dipole"][:, [0,1]]
    internal_options["quad"][:,[1,0]] = internal_options["quad"][:, [0,1]]
    #print("Did transfer to Python dict work for geom.1:\n", internal_options["monopole"])
    #print("Did transfer to Python dict work for geom.1:\n", internal_options["dipole"])
    #print("Did transfer to Python dict work for geom.1:\n", internal_options["quad"])

    internal_options_s["monopole"][:,[1,0]] = internal_options_s["monopole"][:, [0,1]]
    internal_options_s["dipole"][:,[1,0]] = internal_options_s["dipole"][:, [0,1]]
    internal_options_s["quad"][:,[1,0]] = internal_options_s["quad"][:, [0,1]]
    #print("Did transfer to Python dict work for geom.2:\n", internal_options_s["monopole"])
    #print("Did transfer to Python dict work for geom.2:\n", internal_options_s["dipole"])
    #print("Did transfer to Python dict work for geom.2:\n", internal_options_s["quad"])
    # Switch multipole origins for actual sampling

    internal_options["m_origin"][[0,1]] = internal_options["m_origin"][[1,0]]
    #print("Did swapping m_origins work for geom.1:\n", internal_options["m_origin"])
 
    #nuc_o_0 = internal_options["nuc_origin"][0]
    #nuc_o_1 = internal_options["nuc_origin"][1]

    #internal_options["nuc_origin"][0] = 2
    #internal_options["nuc_origin"][1] = 0
    #print("Did swapping nuc_origins work:\n", internal_options["nuc_origin"])
    
    internal_options_s["m_origin"][[0,1]] = internal_options_s["m_origin"][[1,0]]
    #print("Did swapping m_origins work for geom.2:\n", internal_options_s["m_origin"])
 
    #nuc_o_0_s = internal_options_s["nuc_origin"][0]
    #nuc_o_1_s = internal_options_s["nuc_origin"][1]

    #internal_options_s["nuc_origin"][0] = 2
    #internal_options_s["nuc_origin"][1] = 0
    #print("Did swapping nuc_origins work:\n", internal_options_s["nuc_origin"])

    internal_options["active_batch"] = 0
    internal_options_s["active_batch"] = 0
    
    sum_rel_e = 0.0
    sum_square_unc = 0.0
    for j in range(nbat):

      total_e = 0
      total_e_s = 0
      
      internal_options["active_batch"] = j
      internal_options_s["active_batch"] = j
      #print("Current active electrons: ", internal_options["active_elec"])
      internal_options["m_origin_cur"] = internal_options["m_origin"][j]
      print("Current multipole origin for this batch:\n", internal_options["m_origin_cur"])
      internal_options_s["m_origin_cur"] = internal_options_s["m_origin"][j]
      print("Current multipole origin for this batch:\n", internal_options_s["m_origin_cur"])

    # Prepare an array to hold the electron positions for each thread.
    # To start we set this to the initial electron positions, but once we've taken
    # samples we just keep each thread's most recent sample here.
      elec_pos = np.zeros([nthread, 2, nalp, 3])
      for i in range(nthread):
          elec_pos[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
          elec_pos[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])

      # prepare dictionary that holds the data arrays for the different accumulators
      acc_names = ["AccumulatorVGR", "AccumulatorKE", "AccumulatorEEE", "AccumulatorENE", "AccumulatorNNE", "AccumulatorLocalE"] #, "AccumulatorPvP0"]
      acc_names2 = ["AccumulatorVGR", "AccumulatorKE", "AccumulatorEEE", "AccumulatorENE", "AccumulatorNNE", "AccumulatorLocalE", "AccumulatorLocalEPsiRatio", "AccumulatorPsiRatio"] #, "AccumulatorPvP0"]
      acc_dict1 = make_accumulator_dict(acc_names, internal_options)
      acc_dict2 = make_accumulator_dict(acc_names2, internal_options_s)
      
      # take the sample and accumulate the data we need
      vmc_take_relative_samples(internal_options, internal_options_s, acc_dict1, acc_dict2, elec_pos)

      total_e += np.mean(acc_dict1["AccumulatorLocalE"], axis=0)
      #total_e_s += np.mean(acc_dict2["AccumulatorLocalE"], axis=0)
   
      #print("AccumulatorPsiRatio for acc_dict2:\n", acc_dict2["AccumulatorPsiRatio"])
      #print("AccumulatorLocalEPsiRatio for acc_dict2:\n", acc_dict2["AccumulatorLocalEPsiRatio"])
      #one_over_psi_ratio = 1.0/np.mean(acc_dict2["AccumulatorPsiRatio"], axis=0)
      #print("One over psi ratio:\n", one_over_psi_ratio)
      #local_e_s = one_over_psi_ratio * acc_dict2["AccumulatorLocalEPsiRatio"] 

      #deltaE = local_e_s - acc_dict1["AccumulatorLocalE"]
      #print("Delta E array:\n", deltaE)
      #print("Average delta E:\n", np.mean(deltaE, axis=0))
      #uncertainty = np.std(deltaE, axis=0) / np.sqrt(24)
      #print("uncertainty:\n", uncertainty)
      

      #local_e_s = np.divide(acc_dict2["AccumulatorLocalEPsiRatio"], acc_dict2["AccumulatorPsiRatio"])
      #print("Weighted Local Energy for Geom. 2:\n", local_e_s)
      #print("Mean of weighted local energy for geom. 2:\n", np.mean(local_e_s, axis=0))
      #print("Uncertainty:\n", np.std(local_e_s, axis=0)/np.sqrt(32))
      #print("Local energy for geom. 1:\n", acc_dict1["AccumulatorLocalE"])
      #deltaE = local_e_s - acc_dict1["AccumulatorLocalE"]
      #print("Delta E for the stretch in correlated sampling:\n", deltaE)
      #print()
      #print("Mean delta:\n", np.mean(deltaE, axis=0))
      #print("Uncertainty:\n", np.std(deltaE, axis=0)/np.sqrt(24))

      # Do statistics for the weighted local energy estimator for the secondary geometry
      les_data = acc_dict2["AccumulatorLocalEPsiRatio"]
      psi_rat_data = acc_dict2["AccumulatorPsiRatio"]
      le_data = acc_dict1["AccumulatorLocalE"]
      print()
      print("AccumulatorLocalEPsiRatio array for geom. 2:\n", les_data)
      print()
      print("AccumulatorPsiRatio array for geom. 2:\n", psi_rat_data)
      print()
      print("AccumulatorLocalE array for geom. 1:\n", le_data)
      print()
      avg_les = np.mean(les_data, axis=0)
      #print("Average local e s for geom. 2:\n",avg_les) 
      print()
      std_les = np.std(les_data, axis=0)
      avg_psi_rat = np.mean(psi_rat_data, axis=0)
      #print("Average psi rat for geom. 2:\n",avg_psi_rat) 
      #print()
      std_psi_rat = np.std(psi_rat_data, axis=0)
      sqrtN = np.sqrt( 1.0 * psi_rat_data.size )
      avg_weighted_les = avg_les / avg_psi_rat
      print("Average weighted local energy of geom. 2:\n", avg_weighted_les)
      #cov_les_psi = np.mean(   ( les_data - np.reshape(avg_les, [1,-1]) ) * (psi_rat_data - np.reshape(avg_psi_rat, [-1,1])), axis=0)
      cov_les_psi = np.mean(   ( les_data - np.reshape(avg_les, [1,-1]) ) * np.reshape(psi_rat_data - avg_psi_rat, [-1,1]), axis=0)
      #print()
      #print("np.multiply(les_data, psi_rat_data):\n", np.multiply(les_data, psi_rat_data))
      #print()
      #print("np.reshape(avg_les,[1,-1]):\n", np.reshape(avg_les, [1,-1]))
      #print()  
      #print("les_data - np.reshape(avg_les, [1,-1]):\n", les_data - np.reshape(avg_les, [1,-1]))
      #print("Covariance for LES and Psi Rat:\n", cov_les_psi)
      cov_les_psi_3 = np.mean( np.multiply(les_data, psi_rat_data),axis=0) - avg_les * avg_psi_rat
      #print("Covariance another way:\n", cov_les_psi_3)
      cov_les_psi_2 = np.mean( ( les_data - avg_les ) * np.reshape(psi_rat_data - avg_psi_rat, [-1,1]), axis=0)
      std_les_over_psi = np.sqrt(    ( avg_les / avg_psi_rat ) * ( avg_les / avg_psi_rat ) * (   ( std_les / avg_les ) * ( std_les / avg_les )+ ( std_psi_rat / avg_psi_rat ) * ( std_psi_rat / avg_psi_rat ) - 2.0 * cov_les_psi_3 / ( avg_les * avg_psi_rat ) ) )
      weighted_les_unc = std_les_over_psi / sqrtN
      print()
      print("Uncertainty of weighted local energy of geom. 2:\n", weighted_les_unc) 

      # Do statistics for the relative energy between geom. 1 and geom.2 
      # Var(X/Y - Z) 
      avg_le = np.mean(le_data, axis=0)
      std_le = np.std(le_data, axis=0)
      rel_e = avg_weighted_les - avg_le
      sum_rel_e += rel_e
      #print()
      #print("Average delta:\n", rel_e)                 
     
      avg_X = avg_les
      avg_Y = avg_psi_rat
      avg_Z = avg_le
      std_X = std_les
      std_Y = std_psi_rat
      std_Z = std_le
      cov_XY = np.mean( np.multiply(les_data, psi_rat_data),axis=0) - avg_les * avg_psi_rat
      cov_XZ = np.mean( np.multiply(les_data, le_data),axis=0) - avg_les * avg_le
      cov_YZ = np.mean( np.multiply(le_data, psi_rat_data),axis=0) - avg_le * avg_psi_rat
      diff_avg = avg_X / avg_Y - avg_Z
      sqrt_1 = (std_X / avg_Y)**2.0 + (avg_X * std_Y / avg_Y**2.0)**2.0 + std_Z**2.0
      sqrt_2 = -2.0 * avg_X * cov_XY / avg_Y**3.0 - 2.0 * cov_XZ / avg_Y + 2.0 * avg_X * cov_YZ / avg_Y**2.0
      #print()
      #print("sqrt_1:\n", sqrt_1)
      #print()
      #print("sqrt_2:\n", sqrt_2)
      var_XYZ = np.sqrt( sqrt_1 + sqrt_2 )
      diff_unc = var_XYZ / sqrtN
      sum_square_unc += diff_unc * diff_unc
      #print()
      #print("Uncertainty of the difference:\n", diff_unc)

      # Do statistics for the relative energy between geom. 1 and geom.2 
      #avg_le = np.mean(le_data, axis=0)
      #std_le = np.std(le_data, axis=0)
      #rel_e = avg_weighted_les - avg_le
      #print("Average delta:\n", rel_e)                 

      #avg_A = avg_weighted_les
      #avg_B = avg_le
      #diff_avg = avg_A - avg_B
      #std_A = weighted_les_unc * sqrtN
      #std_B = std_le
      #cov_AB = np.mean( (np.divide(les_data, psi_rat_data) - np.reshape(avg_A, [1, -1])) * (le_data - np.reshape(avg_B, [1,-1])), axis=0)
      #cov_AB = np.mean( np.multiply( np.divide(les_data, avg_psi_rat), le_data), axis=0) - avg_A * avg_B
      #cov_AB_2 = np.mean(    np.multiply( (les_data / avg_psi_rat - avg_A), (le_data - avg_B))   , axis=0)
      #print("Covariance between <f>/<g> - <h>:\n", cov_AB)
      #print("Covariance between <f>/<g> - <h> another way:\n", cov_AB_2)
      #sqrt_1 = std_A * std_A + std_B * std_B
      #print("sqrt_1:\n", sqrt_1)
      #sqrt_2 = 2.0 * cov_AB
      #print("sqrt_2:\n", sqrt_2)
      #std_A_minus_B = np.sqrt( std_A * std_A + std_B * std_B - 2.0 * cov_AB )
      #diff_unc = std_A_minus_B / sqrtN
      #print()
      #print("Average delta again:\n", diff_avg)
      #print()
      #print("Uncertainty of delta:\n", diff_unc)

      #avg_A = np.mean(num_matrix, axis=0)
      #std_A = np.std(num_matrix, axis=0)
      #avg_B = np.mean(vgr_vec)
      #std_B = np.std(vgr_vec)
      #sqrtN = np.sqrt( 1.0 * vgr_vec.size )
      #ratio_avg = avg_A / avg_B
      #if do_ratio_unc:
      #cov_AB = np.mean(   ( num_matrix - np.reshape(avg_A, [1,-1]) ) * np.reshape(vgr_vec - avg_B, [-1,1]), axis=0)
      #std_A_over_B = np.sqrt(    ( avg_A / avg_B ) * ( avg_A / avg_B ) * (   ( std_A / avg_A ) * ( std_A / avg_A )+ ( std_B / avg_B ) * ( std_B / avg_B ) - 2.0 * cov_AB / ( avg_A * avg_B ) ) )
      #ratio_unc = std_A_over_B / sqrtN
      #else:
      #ratio_unc = np.zeros_like(ratio_avg)
      #vgr_avg = avg_B
      #vgr_unc = std_B / sqrtN
      #return ratio_avg, ratio_unc, vgr_avg, vgr_unc


      #def region_ratio_difference_stats(num_matrix_1, vgr_vec_1, num_matrix_2, vgr_vec_2, do_diff_unc=True, print_covAB=True):
      #sqrtN = np.sqrt( 1.0 * vgr_vec_1.size )
      #ratio_avg_1, ratio_unc_1, vgr_avg_1, vgr_unc_1 = region_ratio_stats(num_matrix_1, vgr_vec_1, do_diff_unc)
      #ratio_avg_2, ratio_unc_2, vgr_avg_2, vgr_unc_2 = region_ratio_stats(num_matrix_2, vgr_vec_2, do_diff_unc)
      #avg_A = ratio_avg_1
      #avg_B = ratio_avg_2
      #diff_avg = avg_A - avg_B
      #if do_diff_unc:
      #std_A = ratio_unc_1 * sqrtN
      #std_B = ratio_unc_2 * sqrtN
      #cov_AB = np.mean(   ( num_matrix_1 / vgr_avg_1 - np.reshape(avg_A, [1,-1]) ) * ( num_matrix_2 / vgr_avg_2 - np.reshape(avg_B, [1,-1]) ), axis=0)
      #if print_covAB:
      #print("In A_minus_B, cov_AB is: ", end="")
      #printmat(np.reshape(cov_AB, [1,-1]))
      #print("")
      #sqrt_arg1 = np.reshape(std_A * std_A + std_B * std_B, [-1])
      #sqrt_arg2 = np.reshape(2.0 * cov_AB, [-1])
    #for i in range(sqrt_arg1.shape[0]):
    #  if sqrt_arg1[i] - sqrt_arg2[i] < 0.0:
    #    print("In region_ratio_difference_stats, sqrt argument is less than zero: %14.6e vs %14.6e" % (sqrt_arg1[i] - sqrt_arg2[i], sqrt_arg1[i]) )
    #  if np.abs(sqrt_arg1[i]-sqrt_arg2[i]) / sqrt_arg1[i] < 1.0e-10: # avoid nan on negative numbers that come from roundoff error
    #    sqrt_arg1[i] = 0.0
    #    sqrt_arg2[i] = 0.0
      #std_A_minus_B = np.reshape( np.sqrt( sqrt_arg1 - sqrt_arg2 ), std_A.shape )
      #diff_unc = std_A_minus_B / sqrtN
      #else:
      #diff_unc = np.zeros_like(diff_avg)
      #return diff_avg, diff_unc

    # save the value guiding ratio averages (they are the same for all regions, so we have one per sampling block)
      # process the data we accumulated during sampling and print the results
      stats_processors = []
      for ad in [ acc_dict1, acc_dict2 ]:
        acc_data_total_e = ad["AccumulatorKE"] + ad["AccumulatorEEE"] + ad["AccumulatorENE"] + ad["AccumulatorNNE"]
        stats_processors.append( [
          RegionStatsProcessor(     'kinetic', ad['AccumulatorVGR'], ad[ "AccumulatorKE"],  True),
          RegionStatsProcessor(   'elec-elec', ad['AccumulatorVGR'], ad["AccumulatorEEE"],  True),
          RegionStatsProcessor(    'elec-nuc', ad['AccumulatorVGR'], ad["AccumulatorENE"],  True),
          RegionStatsProcessor(     'nuc-nuc', ad['AccumulatorVGR'], ad["AccumulatorNNE"], False),
          RegionStatsProcessor('total energy', ad['AccumulatorVGR'],     acc_data_total_e,  True),
        ] )
        print("")
        print("absolute quantities for batch", j)
        print("")
        print(stats_processors[-1][0].table_header())
        print(stats_processors[-1][0].table_dashes())
        for rsp in stats_processors[-1]:
          print(rsp.table_row())

      total_total_e += total_e
      total_total_e_s += total_e_s
    print("Total E geom.1: ", total_total_e)
    #print("Total E geom.2: ", total_total_e_s)

    print("")
    print("delta-and-uncertainty: %20.12f +/- %20.12f" % (sum_rel_e, np.sqrt(sum_square_unc)))
    print("")


    #print("----------------------------------")
    #print("AccumulatorPsiRatio for geom.2:\n", acc_dict2["AccumulatorPsiRatio"])
    #print("AccumulatorLocalEPsiRatio for geom.2:\n", acc_dict2["AccumulatorLocalEPsiRatio"])

#def do_relative_energy(internal_options):
#
#  # get number of nuclei
#  nnuc = internal_options["nuclei"].size // 3
#
#  # get number of alpha electrons
#  nalp = internal_options["apos"].size // 3
#
#  # get number of threads
#  nthread = get_number_of_threads()
#  print("")
#  print("have %i threads for do_relative_energy" % nthread)
#  print("", flush=True)
#
#  # Prepare an array to hold the electron positions for each thread.
#  # To start we set this to the initial electron positions, but once we've taken
#  # samples we just keep each thread's most recent sample here.
#  elec_pos = np.zeros([nthread, 2, nalp, 3])
#  for i in range(nthread):
#    elec_pos[i, 0, :, :] = np.reshape(internal_options["apos"], [nalp,3])
#    elec_pos[i, 1, :, :] = np.reshape(internal_options["bpos"], [nalp,3])
#
#  # prepare dictionaries that hold the data arrays for the different accumulators
#  acc_names = ["AccumulatorVGR", "AccumulatorKE", "AccumulatorEEE", "AccumulatorENE", "AccumulatorNNE"]
#  acc_dict1 = make_accumulator_dict(acc_names, internal_options)
#  acc_dict2 = make_accumulator_dict(acc_names, internal_options)
#
#  # take the sample and accumulate the data we need
#  vmc_take_relative_samples(internal_options, acc_dict1, acc_dict2, elec_pos)
#
#  # process the data we accumulated during sampling and print the results
#  stats_processors = []
#  for ad in [ acc_dict1, acc_dict2 ]:
#    acc_data_total_e = ad["AccumulatorKE"] + ad["AccumulatorEEE"] + ad["AccumulatorENE"] + ad["AccumulatorNNE"]
#    stats_processors.append( [
#      RegionStatsProcessor(     'kinetic', ad['AccumulatorVGR'], ad[ "AccumulatorKE"],  True),
#      RegionStatsProcessor(   'elec-elec', ad['AccumulatorVGR'], ad["AccumulatorEEE"],  True),
#      RegionStatsProcessor(    'elec-nuc', ad['AccumulatorVGR'], ad["AccumulatorENE"],  True),
#      RegionStatsProcessor(     'nuc-nuc', ad['AccumulatorVGR'], ad["AccumulatorNNE"], False),
#      RegionStatsProcessor('total energy', ad['AccumulatorVGR'],     acc_data_total_e,  True),
#    ] )
#    print("")
#    print("absolute quantities at geometry %i:" % len(stats_processors))
#    print("")
#    print(stats_processors[-1][0].table_header())
#    print(stats_processors[-1][0].table_dashes())
#    for rsp in stats_processors[-1]:
#      print(rsp.table_row())
#  print("")
#  print("differences (geomtry 1 minus geometry 2):")
#  print("")
#  print(stats_processors[0][0].table_header())
#  print(stats_processors[0][0].table_dashes())
#  for i in range(len(stats_processors[0])):
#    print(stats_processors[0][i].table_diff_row(stats_processors[1][i]))
#  print("")

def cusp_a0_scf(options_input):
  """ self-consistently optimize cusp_a0 and cusp_coeff_matrix """

  options = options_input.copy()

  print("Before SCF update\n", options["cusp_a0"])
  update_cusp_a0 = np.zeros_like(options["cusp_a0"])
  update_cusp_coeff = np.zeros_like(options["cusp_coeff_matrix"])

  scf_cycle = 1
  do_scf = True

  scf_dict = {}

  while do_scf == True:
    print("SCF cycle: ", scf_cycle, flush=True)
    for nuc_ind in np.arange(options["cusp_a0"].shape[0]):  # columns
      for ao_ind in np.arange(options["cusp_a0"].shape[1]):  # rows 
        a0 = options["cusp_a0"][nuc_ind, ao_ind]
        if a0 != 0.0:
          ij, all_nuc_xyz, nuc_xyz, ao_type, basis_center_ind, on_center, basis_center_xyz, alpha, d, r_cusp, orth_orb_shifted, core_ind, alpha_core, d_core, proj, pyscf_s, pyscf_en = cusp_orbitals.gaussian_info_and_eval(nuc_ind, ao_ind, False, options)

          delta = nuc_xyz - basis_center_xyz
          delta_norm = delta if np.linalg.norm(delta) == 1. else delta / np.linalg.norm(delta)
          r_eval = cusp_orbitals.r_from_xyz(basis_center_xyz,nuc_xyz)[0][0] + 0.000001 #SET TO CUSP Peek 

          xyz_orb_thru_nuc = (basis_center_xyz) if on_center == True else (basis_center_xyz + r_eval.reshape(-1,1) * delta_norm)    #  [m,3], xyz coord of line through the two nuclei from ao to nuc

          Z = options["Z"][nuc_ind]
          zeta = Z/2 if ao_type > 1 and on_center is True else Z    # only Z/2 for atom centered p orbitals

          input_info = [[r_eval, xyz_orb_thru_nuc, ao_type, alpha_core, d_core, basis_center_xyz, nuc_xyz, alpha, d, on_center, orth_orb_shifted, proj]]
          cusp_stuff = [[a0, r_cusp, zeta, r_eval]]
          lc_cusp_stuff = [[options["cusp_coeff_matrix"][ij[0],ij[1],:], options["order_n_list"], options["cusp_type"]]]

          orb_evals, cusp_orb_evals, slater_orb_evals, lc_slater_cusp_orb_evals = cusp_orbitals.gaussian_and_cusp_r_val(input_info, cusp_info=cusp_stuff, lc_cusp_info=lc_cusp_stuff, get_max=False)
          update_cusp_a0[nuc_ind, ao_ind] = lc_slater_cusp_orb_evals[0][0]

          #print(nuc_ind, ao_ind,"should match(", a0, ") == ", slater_orb_evals[0][0], "new_a0 = ", lc_slater_cusp_orb_evals[0][0], "\t difference: ", lc_slater_cusp_orb_evals[0][0] - a0)

    print("\nNew a0 matrix\n", update_cusp_a0, flush=True) 
    print("\nDifference in a0 matrix\n", update_cusp_a0 - options["cusp_a0"], flush=True) 

    if np.array_equal(np.around(update_cusp_a0 - options["cusp_a0"], decimals=4), np.zeros_like(update_cusp_a0)):  # if no cusp a0 changes more than 1e-4
      print("\nCONVERGED --- After SCF update\n", update_cusp_a0, flush=True) 
      do_scf = False
    else:
      print("\nAfter SCF update\n", update_cusp_a0, flush=True) 
      scf_cycle += 1

    options["cusp_a0"] = scf_dict['cusp_a0_scf'+str(scf_cycle)] = update_cusp_a0

    print("\n\n*****\nCalc new cusp_coeff_matrix", flush=True)
    update_cusp_coeff = cusp_orbitals.get_cusp_coeff_matrix(options, stretched=False)

    print("\nDifference in cusp_coeff_matrix\n", update_cusp_coeff - options["cusp_coeff_matrix"], flush=True) 
    options['cusp_coeff_matrix'] = scf_dict['cusp_coeff_matrix_scf'+str(scf_cycle)] = update_cusp_coeff
   
  scf_dict["final_iter"] = scf_cycle 
  return scf_dict #options["cusp_a0"], options["cusp_coeff_matrix"]

def batch_cusp_test(internal_options_pkl,C,name):
  with open(internal_options_pkl, 'rb') as fp:
    internal_options = pickle.load(fp)
  internal_options["file_name"] = 'cusp_test/cusp_test_'+name 
  internal_options["mocoeff"] = C 
  do_1e_cusp_test(internal_options, mo_list=np.arange(0,internal_options["mocoeff"].shape[-1]), acc_names=["AccumulatorCuspTestingLE","AccumulatorCuspTestingKE",], which_orb_all=['cusp', ])
  print()

def vanilla_pickup(internal_options_pkl,name,nspb=None,num_iters=21):
  with open(internal_options_pkl, 'rb') as fp:
    internal_options = pickle.load(fp)
  
  # have intermediate step to tunr on all bfs on "on" atoms
  #   then grow to add neighbooring atoms
  internal_options["file_name"] = name 
  internal_options["constrained_opt"] = True 
  internal_options["selected_LCAO"] = True 
  internal_options["epsilon"] = 0.0000 
  internal_options["zeros_fixed"] = False

  # update fixed_param_ind to include whole mat, expect normalization
  internal_options["fixed_param_ind"] = linear_method.set_param_ind(internal_options["mocoeff"], 
                                                                    internal_options["Z"], 
                                                                    internal_options["basis_centers"], 
                                                                    internal_options["nearest_neighboors"], 
                                                                    internal_options["constrained_opt"],
                                                                    internal_options["zeros_fixed"],
                                                                    internal_options["selected_LCAO"],
                                                                    internal_options["opt_these_orbs"],
                                                                    add_neighboors=False) # add_neighboors applies only if selection is turned on



  if nspb != None: internal_options["nsamp_per_block"] = nspb 
  internal_options['iter'] = num_iters 
  #print("internal_options")
  #print(internal_options)
  do_absolute_energy(internal_options)

def repeat_LM1_step(internal_options,outpath,nrepeat):
  """
  outpath - where to save all VMC output files and generated LM matrix
  nrepeat - number of time to run 1 VMC calc and LM step, saving output energy and C mat
  """
  initC = np.copy(internal_options['mocoeff']) 

  initfpi = np.copy(internal_options["fixed_param_ind"])

  initapos = np.copy(internal_options['apos']) 
  initbpos = np.copy(internal_options['bpos']) 
  elecshape = initapos.shape

  #internal_options["LMParams"] = ["AlphaDet", "BetaDet",] 		
  internal_options['iter'] = 1
  internal_options["constrained_opt"] = False
  alloptC=[]

  for i in range(int(nrepeat)): 
    internal_options['mocoeff'] = initC
    internal_options["fixed_param_ind"]=initfpi
    internal_options["seed"] += 378
    internal_options['file_name']=outpath+'_trial'+str(i)
    internal_options['iter_E'] = 0.0 # initialize iteration energy
    internal_options['iter_E_std_err'] = 0.0 # initialize iteration energy

    e_rand = np.random.normal(loc=0.0, scale=0.1, size=np.prod(elecshape)).reshape(elecshape)
    internal_options['apos'] = e_rand + initapos
    internal_options['bpos'] = e_rand + initbpos

    if i < 2: # saving to compare what exactly changed between repeat runs
      with open(internal_options['file_name']+'_internal_options_dict.txt', 'w') as f:  
        for key, value in internal_options.items():  
          f.write('%s:%s\n' % (key, value))
          pickle.dump(internal_options, open(internal_options['file_name']+'_internal_options_dict.pkl', "wb"))

    do_absolute_energy(internal_options,savefiles=False)

    outC = internal_options['mocoeff'] 
    print("Optimized C\n",np.round(outC,4))
    alloptC.append(outC)
    np.savetxt(outpath+"_trial"+str(i)+'.mat',outC)

  #print("\n************ MO cusp test **********\n")
  ## cusp test on optimized C
  #for ind, cmat in enumerate(alloptC:
  #  print("Cmat cusp test:")
  #  do_1e_cusp_test(internal_options, mo_list=np.arange(0,internal_options["mocoeff"].shape[-1]), acc_names=["AccumulatorCuspTestingLE","AccumulatorCuspTestingKE",], which_orb_all=['cusp', ],savefiles=False)



def vmc_take_samples(options,test=False):

  if test:
    """ internal options pkl dict provided, go straight to calc"""
    itera = int(options["iter_start"]) if "iter_start" in options else 0

    do_absolute_energy(options, itera)
    sys.exit()

  options = make_full_input(options)
  print("Done making input",flush=True)

  with open(options['name']+'_initial_options_dict.txt', 'w') as f:  
      for key, value in options.items():  
          f.write('%s:%s\n' % (key, value))
  pickle.dump(options, open(options['name']+'_initial_options_dict.pkl', "wb"))

  print("", flush=True, end="")
  print("")
  print("############################################################################")
  print("###############  Entering vmc_take_samples in simple_vmc.py  ###############")
  print("############################################################################")
  print("")

  # get nuclear positions as a one-dimensional numpy array
  nucpos = process_positions_in_options(options, "nuclei", "nuclei")

  # print secondary nuclear coordinates if doing correlated sampling
  if options["do_corr_samp"]:
    # get nuclear positions as a one-dimensional numpy array
    nucpos_s = process_positions_in_options(options, "nuclei_s", "nuclei_s")
 
  # randomly normally distribute each atoms electrons over the bond center (or nuclei if core e-) 
  if ("apos" not in options) or ("bpos" not in options): 
    print("calculating apos and bpos")
    options["apos"], options["bpos"] = get_elec_pos_near_bond(options['Z'].flatten(), cusp_orbitals.get_mol_xyz(options['nuclei']), options['bonding_neighboors'])
    #options["apos"], options["bpos"] = get_elec_pos_near_bond(options['Z'].flatten(), cusp_orbitals.get_mol_xyz(options['nuclei']), options['nearest_neighboors'])
  else:  
    options["apos"] = options["apos"].reshape(-1)
    options["bpos"] = options["bpos"].reshape(-1)
    print("defaulting to input apos and bpos")

  # get alpha electron starting positions as a one-dimensional numpy array
  apos = process_positions_in_options(options, "apos", "alpha electrons", "starting ")

  # get alpha electron starting positions as a one-dimensional numpy array
  bpos = process_positions_in_options(options, "bpos", "beta electrons", "starting ")

  #print("apos\n",apos)
  #print("bpos\n",bpos)

  # get multipole origin positions as a one-dimensional numpy array
  #mulpos = process_positions_in_options(options, "m_origin", "multipole origins")
  
  # get move center as a one-dimensional numpy array
  move_group_center = np.zeros([3])
  if "move_group_center" in options:
    move_group_center = process_positions_in_options(options, "move_group_center", "center used to organize electrons into move groups")

  # get number of nuclei
  nnuc = nucpos.size // 3

  # get number of alpha electrons
  nalp = apos.size // 3

  # recalc C F S if not all in input
  if "mocoeff" in options:
    if "fock_mat" not in options or "pyscf_S" not in options:
      print("C included in options but F and S not -- to pyscf HF S, F matrices at chosen basis")
      pyscf_s, pyscf_1e_energy, pyscf_fock_mat, pyscf_orbs = cusp_orbitals.pyscf_result_loc(options["Z"], cusp_orbitals.get_mol_xyz(options['nuclei']), options["basis_type"], loc=True)
      #del options["mocoeff"]
      options["fock_mat"] = pyscf_fock_mat
      #options["pyscf_1e_energy"] = pyscf_1e_energy
      options["pyscf_S"] = pyscf_s
    else:
      print("C, F, and S all included in options ")

  if "mocoeff" not in options:
    # check for presence of MO coefficient matrix
    if "HF_mocoeff" in options:
      if options["HF_mocoeff"] == True:
        print()
        print("mocoeff defaulting to pyscf HF C, S, F matrices at chosen basis")
        pyscf_s, pyscf_1e_energy, pyscf_fock_mat, pyscf_orbs = cusp_orbitals.pyscf_result_loc(options["Z"], cusp_orbitals.get_mol_xyz(options['nuclei']), options["basis_type"], loc=True)

        # Run coupled cluster calc
        cusp_orbitals.run_cc(options["Z"], cusp_orbitals.get_mol_xyz(options['nuclei']), options["basis_type"])

        options["mocoeff"] = pyscf_orbs
        options["fock_mat"] = pyscf_fock_mat
        #options["pyscf_1e_energy"] = pyscf_1e_energy
        options["pyscf_S"] = pyscf_s

        print("\nfock matrix shape before orthognalization",  options["fock_mat"].shape, flush=True)
        print("fock_mat before orthogonalization\n",np.array2string(options["fock_mat"],separator=','),flush=True)
        print("")
        print("\noverlap matrix shape before orthognalization", options["pyscf_S"].shape, flush=True)
        print("overlap before orthogonalization\n",np.array2string(options["pyscf_S"],separator=','),flush=True)
      else:
        raise RuntimeError('you need to specify the MO coefficient matrix via options["mocoeff"] or set "HF_mocoeff" to True to default to PYSCF C matrix')
    else:
      raise RuntimeError('you need to specify the MO coefficient matrix via options["mocoeff"] or set "HF_mocoeff" to True to default to PYSCF C matrix')
  else:	# mocoeff in options but want to override with pyscf C matrix, else use input C mat
    if "HF_mocoeff" in options:
      if options["HF_mocoeff"] == True:
        pyscf_s, pyscf_1e_energy, fock_mat, pyscf_orbs = cusp_orbitals.pyscf_result_loc(options["Z"], cusp_orbitals.get_mol_xyz(options['nuclei']), options["basis_type"], loc=True)
        if 'fock_mat' in options and 'pycsf_S' in options:
          print("mocoeff input overrided - defaulting to pyscf HF C (only); S, F matrices already in options")
          options["mocoeff"] = pyscf_orbs
        else:
          print("mocoeff input overrided and F and S not in options - defaulting to pyscf HF C, S, F matrices at chosen basis")
          options["mocoeff"] = pyscf_orbs
          options["fock_mat"] = fock_mat
          #options["pyscf_1e_energy"] = pyscf_1e_energy
          options["pyscf_S"] = pyscf_s

        print("\nfock matrix shape before orthognalization",  options["fock_mat"].shape, flush=True)
        print("fock_mat before orthogonalization\n",np.array2string(options["fock_mat"],separator=','),flush=True)
        print("")
        print("\noverlap matrix shape before orthognalization", options["pyscf_S"].shape, flush=True)
        print("overlap_mat before orthogonalization\n",np.array2string(options["pyscf_S"],separator=','),flush=True)
        print("")
  #sys.exit()
  if type(options["mocoeff"]) != type(np.array([1.0])):
    raise RuntimeError('options["mocoeff"] should be a numpy array')

  # prepare default options to pass to the c++ code
  internal_options = {
              "nuclei": nucpos,
                "apos": apos,
                "bpos": bpos,
               "nelec": 2*nalp,
                "nnuc": nnuc,
                   "Z": 1.0 * options["Z"],
             "mocoeff": 1.0 * options["mocoeff"],
         "batch_mat_a": 1.0 * options["batch_mat_a"],
         "batch_mat_b": 1.0 * options["batch_mat_b"],
         "batch_mat_n": 1.0 * options["batch_mat_n"],
             "batches": options["batches"],
          "num_active": 0,
        "active_batch": 0,
         "num_batches": len(options["batches"]),
             "move_sd": 0.3,
                "jAss": 1.0 * options["jAss"],
                "jAos": 1.0 * options["jAos"],
          #"useJastrow": False,
                "apin": [],
                "bpin": [],
              "nblock": 128,
     "nsamp_per_block": 1000,
             "nwarmup": 100,
           "nsubsteps": 2,
                "seed": 384715,#384713,
              "do_rel": True,
        "qq_plot_name": 'qqplot',
       "rot_tap_start": 0.0,
         "rot_tap_end": 0.0,
           "rot_angle": 0.0,
          "rot_center": np.array([0.0, 0.0, 0.0]),
         "atom_groups": [],
         "move_groups": [],
   "move_group_center": move_group_center,
           "WFNPieces": [
                           # "RMPJastrowAA",
                           # "RMPJastrowBB",
                           # "RMPJastrowAB",
                            "RMPJastrowAA",
                            "RMPJastrowBB",
                            "RMPJastrowAB",
                            "AlphaDet",
                            "BetaDet",
                        ],
              "useHAO": False,
              "useSTO": False,
              "useGTO": False,
          "useCuspGTO": False,
          "basis_type": options['basis_type'],
                  "ng": options["ng"],
           "basis_exp": 1.0 * options["basis_exp"],
         "basis_coeff": 1.0 * options["basis_coeff"],
       "basis_centers": 1.0 * options["basis_centers"], # must be float to work in cpp
      "basis_orb_type": 1.0 * options["basis_orb_type"], # must be float to work in cpp 
                 "nbf": len(options["basis_exp"]),
            "m_origin": 1.0 * options["m_origin"],
        "m_origin_cur": np.array([0.0, 0.0, 0.0]),
            "monopole": 1.0 * options["monopole"],
              "dipole": 1.0 * options["dipole"],
		"quad": 1.0 * options["quad"],
	  "nuc_origin": 1.0 * options["nuc_origin"],
	    "do_batch": options["do_batch"],
	"do_corr_samp": options["do_corr_samp"],
 "do_sherman_morrison": False,
                "iter": 1,
                 "cI" : 0.0,
            "LMParams": options["LMParams"],
              "dim_vp": [],
             "name_vp": [],
                 "nvp": 0,
     "constrained_opt": False,
         "zeros_fixed": False,
       "selected_LCAO": False,
      "opt_these_orbs": np.array([]), 
     "fixed_param_ind": np.array([]), 
             "epsilon": 0.0,
        "max_delta_p" : 0.0, 
             "cusp_a0": np.array([]),
         "cusp_radius": None, 
      "cusp_radii_mat": np.array([]),
   "cusp_coeff_matrix": np.array([]),
      "orth_orb_array": np.array([]), # must be float to work in cpp
            "proj_mat": np.array([]),
           "file_name": options['name'],
"get_slater_derivs_cusp": False,
        "order_n_list": np.array([]),
          "num_p_func": 0, # 0.1 ensures rounding to 0 in c++
         "dE_schedule": False
  }
 
  if options["do_corr_samp"]:

    internal_options_s = {
                "nuclei": nucpos_s,
                  "apos": apos,
                  "bpos": bpos,
                 "nelec": 2*nalp,
                  "nnuc": nnuc,
                     "Z": 1.0 * options["Z"],
               "mocoeff": 1.0 * options["mocoeff_s"],
           "batch_mat_a": 1.0 * options["batch_mat_a"],
           "batch_mat_b": 1.0 * options["batch_mat_b"],
           "batch_mat_n": 1.0 * options["batch_mat_n"],
               "batches": options["batches"],
            "num_active": 0,
          "active_batch": 0,
           "num_batches": len(options["batches"]),
               "move_sd": 0.3,
                  "jAss": 1.0 * options["jAss"],
                  "jAos": 1.0 * options["jAos"],
            #"useJastrow": False,
                  "apin": [],
                  "bpin": [],
                "nblock": options["nblock"],
       "nsamp_per_block": options["nsamp_per_block"],
               "nwarmup": options["nwarmup"],
             "nsubsteps": 2,
                  "seed": 384715,#384713,
                "do_rel": True,
          "qq_plot_name": 'qqplot',
         "rot_tap_start": 0.0,
           "rot_tap_end": 0.0,
             "rot_angle": 0.0,
            "rot_center": np.array([0.0, 0.0, 0.0]),
           "atom_groups": [],
           "move_groups": [],
     "move_group_center": move_group_center,
             "WFNPieces": [
                              "RMPJastrowAA",
                              "RMPJastrowBB",
                              "RMPJastrowAB",
                              "AlphaDet",
                              "BetaDet",
                              #"AlphaPinDet",
                              #"BetaPinDet",
                          ],
                "useSTO": True,
                #"useHAO": True,
                 "basis": [
                              "HAO",
                          ],
                    "ng": 1,
             "basis_exp": 1.0 * options["basis_exp"],
           #"basis_coeff": 1.0 * options["basis_coeff"],
         "basis_centers": 1.0 * options["basis_centers"],
        "basis_orb_type": 1.0 * options["basis_orb_type"],
                   "nbf": len(options["basis_exp"]),
              "m_origin": 1.0 * options["m_origin"],
          "m_origin_cur": np.array([0.0, 0.0, 0.0]),
              "monopole": 1.0 * options["monopole_s"],
                "dipole": 1.0 * options["dipole_s"],
  		  "quad": 1.0 * options["quad_s"],
       	    "nuc_origin": 1.0 * options["nuc_origin"],
	      "do_batch": options["do_batch"],
	  "do_corr_samp": options["do_corr_samp"],
   "do_sherman_morrison": False,
     "cusp_coeff_matrix": np.array([]),
          "order_n_list": np.array([]),
            "num_p_func": 0 # 0.1 ensures rounding to 0 in c++
    }
  if "WFNPieces" in options:
    internal_options["WFNPieces"] = options["WFNPieces"]

  # standard deviation for electron moves
  if "move_sd" in options:
    if type(options["move_sd"]) == type(1.0):
      internal_options["move_sd"] = options["move_sd"]
    else:
      raise RuntimeError('options["move_sd"] should be a floating point number')
#
#  # same-spin Jastrow variable
#  if "jAss" in options:
#    print(options['jAss'][0], type(options['jAss'][0]), flush=True)
#    if type(options["jAss"][0]) == type(1.0):
#      internal_options["jAss"] = options["jAss"]
#    else:
#      raise RuntimeError('options["jAss"] should be a floating point number')
#
#  # opposite-spin Jastrow variable
#  if "jAos" in options:
#    if type(options["jAos"][0]) == type(1.0):
#      internal_options["jAos"] = options["jAos"]
#    else:
#      raise RuntimeError('options["jAos"] should be a floating point number')
#
#  # whether to use the jastrow factor
#  if "useJastrow" in options:
#    if type(options["useJastrow"]) == type(True):
#      internal_options["useJastrow"] = options["useJastrow"]
#    else:
#      raise RuntimeError('options["useJastrow"] should be either True or False')

  # list of pinned alpha electrons
  if "apin" in options:
    if type(options["apin"]) != type([]):
      raise RuntimeError('options["apin"] should be a list of integers')
    if len(options["apin"]) > 0:
      print('pinning alpha electrons at positions:')
      print('')
    for a in options["apin"]:
      if type(a) != type(1):
        raise RuntimeError('options["apin"] should be a list of integers')
      if a < 0 or a >= nalp:
        raise RuntimeError('options["apin"] elements should be >= 0 and < %i' % (nalp))
      print(' %20.12f %20.12f %20.12f' % ( apos[3*a+0], apos[3*a+1], apos[3*a+2] ) )
    if len(options["apin"]) > 0:
      print('')
    internal_options["apin"] = options["apin"]

  # list of pinned beta electrons
  if "bpin" in options:
    if type(options["bpin"]) != type([]):
      raise RuntimeError('options["bpin"] should be a list of integers')
    if len(options["bpin"]) > 0:
      print('pinning  beta electrons at positions:')
      print('')
    for a in options["bpin"]:
      if type(a) != type(1):
        raise RuntimeError('options["bpin"] should be a list of integers')
      if a < 0 or a >= nalp:
        raise RuntimeError('options["bpin"] elements should be >= 0 and < %i' % (nalp))
      print(' %20.12f %20.12f %20.12f' % ( bpos[3*a+0], bpos[3*a+1], bpos[3*a+2] ) )
    if len(options["bpin"]) > 0:
      print('')
    internal_options["bpin"] = options["bpin"]

  # number of total blocks over all threads
  if "nblock" in options:
    if type(options["nblock"]) != type(1):
      raise RuntimeError('options["nblock"] should be an integer')
    internal_options["nblock"] = options["nblock"]

  # number of samples per block
  if "nsamp_per_block" in options:
    if type(options["nsamp_per_block"]) != type(1):
      raise RuntimeError('options["nsamp_per_block"] should be an integer')
    internal_options["nsamp_per_block"] = options["nsamp_per_block"]

  # number of warmup steps
  if "nwarmup" in options:
    if type(options["nwarmup"]) != type(1):
      raise RuntimeError('options["nwarmup"] should be an integer')
    internal_options["nwarmup"] = options["nwarmup"]

  # number of steps between samples
  if "nsubsteps" in options:
    if type(options["nsubsteps"]) != type(1):
      raise RuntimeError('options["nsubsteps"] should be an integer')
    internal_options["nsubsteps"] = options["nsubsteps"]

  # info for how many electrons are in each sampling group and how many times we move electrons in each group during a sampling block
  if "move_groups" in options:
    if type(options["move_groups"]) != type([]):
      raise RuntimeError('options["move_groups"] should be list of lists, with each inner list containing two integers')
    internal_options["move_groups"] = []
    for mg in options["move_groups"]:
      if type(mg) != type([]) or len(mg) != 2:
        raise RuntimeError('options["move_groups"] should be list of lists, with each inner list containing two integers')
      if type(mg[0]) != type(1) or type(mg[1]) != type(1):
        raise RuntimeError('options["move_groups"] should be list of lists, with each inner list containing two integers')
      internal_options["move_groups"].append( [ x + 0 for x in mg ] )

  # seed for random number generator
  if "seed" in options:
    if type(options["seed"]) != type(1):
      raise RuntimeError('options["seed"] should be an integer')
    internal_options["seed"] = options["seed"]

  # flag for whether we are doing a relative energy estimation
  if "do_rel" in options:
    if type(options["do_rel"]) != type(True):
      raise RuntimeError('options["do_rel"] should be True or False')
    internal_options["do_rel"] = options["do_rel"]

  # name for qq plots
  if "qq_plot_name" in options:
    if type(options["qq_plot_name"]) != type("hi"):
      raise RuntimeError('options["qq_plot_name"] should be a string')
    internal_options["qq_plot_name"] = options["qq_plot_name"]

  # start of the rotation taper
  if "rot_tap_start" in options:
    if type(options["rot_tap_start"]) != type(1.0):
      raise RuntimeError('options["rot_tap_start"] should be a floating point number')
    internal_options["rot_tap_start"] = options["rot_tap_start"]

  # end of the rotation taper
  if "rot_tap_end" in options:
    if type(options["rot_tap_end"]) != type(1.0):
      raise RuntimeError('options["rot_tap_end"] should be a floating point number')
    internal_options["rot_tap_end"] = options["rot_tap_end"]

  # tapered rotation base rotation angle
  if "rot_angle" in options:
    if type(options["rot_angle"]) != type(1.0):
      raise RuntimeError('options["rot_angle"] should be a floating point number')
    internal_options["rot_angle"] = options["rot_angle"]

  # tapered rotation center
  if "rot_center" in options:
    if type(options["rot_center"]) != type(np.array([1.0])):
      raise RuntimeError('options["rot_center"] should be a numpy array')
    internal_options["rot_center"] = options["rot_center"]

  # atom groups (we add their energies together before doing statistics)
  if "atom_groups" in options:
    if type(options["atom_groups"]) != type([]):
      raise RuntimeError('options["atom_groups"] should be a list of lists of integers')
    for ag in options["atom_groups"]:
      if type(ag) != type([]):
        raise RuntimeError('options["atom_groups"] should be a list of lists of integers')
      for i in ag:
        if type(i) != type(1):
          raise RuntimeError('options["atom_groups"] should be a list of lists of integers')
        if i < 0:
          raise RuntimeError('indices inside options["atom_groups"] must not be negative')
        if i >= nnuc:
          raise RuntimeError('indices inside options["atom_groups"] must be less than the number of nuclei (%i)' % nnuc)
    internal_options["atom_groups"] = options["atom_groups"]

  # if doing Sherman Morrison math
  if "do_sherman_morrison" in options:
    if internal_options["num_batches"] > 1 and options["do_sherman_morrison"] == True:
      raise RuntimeError('Sherman Morrison math only compatible with a vanilla calculation (make sure there is only one batch of electrons)')
    #print("ensuring do_sherman_morrison is TRUE in simple_vmc.py vmc_take_samples()")
    internal_options["do_sherman_morrison"] = options["do_sherman_morrison"]

  ### BASIS TYPE ###
  if 'basis_type' in options:
    internal_options["basis_type"] = options["basis_type"]
    if internal_options["basis_type"] in {"STO-3G", "6-31G", "6-31G*", "cc-pcvdz"}:
      internal_options["useGTO"] = True
      print("useGTO is TRUE")
      if 'ng' in options:
        internal_options['ng'] = options["ng"]
      else:
        RuntimeError("Define number of gaussians in the STO-nG basis via 'ng'") 
    elif internal_options["basis_type"] == "STO":
      internal_options["useSTO"] = True
    elif internal_options["basis_type"] == "HAO":
      internal_options["useHAO"] = True
    else:
      RuntimeError("Cannot recognize basis: ", options["basis_type"])
  else:
    RuntimeError("must define basis_type")

  ### CUSPED BASIS ###
  if "useCuspGTO" in options:  # default False 

    internal_options["useCuspGTO"] = options["useCuspGTO"]
    
    if internal_options["useCuspGTO"] == True:  # move C, F, and S to orthogonalized basis (ie B.T @ C @ B)
    
      all_nuc_xyz = cusp_orbitals.get_mol_xyz(options['nuclei']) 
      global_cusp = options["cusp_radius"] if "cusp_radius" in options else None
      print("set orb info", flush=True)
      r_cusp_matrix, orth_orb_array = cusp_orbitals.set_basis_orb_info(global_cusp, options['basis_type'], options['basis_orb_type'], options['basis_centers'], options['Z'])

      if 'cusp_radii_mat' in options:
        print("cusp_radii_mat from input")
        internal_options['cusp_radii_mat'] = options['cusp_radii_mat']
      else:
        print("cusp_radii_mat generated")
        print(internal_options['cusp_radii_mat'], flush=True)
        internal_options['cusp_radii_mat'] = options['cusp_radii_mat'] = r_cusp_matrix
        #raise RuntimeError('cusp radius must be defined in some way, set options["cusp_radius"] or options["cusp_radii_mat"]')
        #print(internal_options['cusp_radii_mat'],flush=True)
        #sys.exit()

      if "orth_orb_array" in options: # must be float to work in cpp
        print("orth_orb_array from input")
        internal_options['orth_orb_array'] = 1.0 * options['orth_orb_array']
      else: 
        print("orth_orb_array generated")
        options['orth_orb_array'] = orth_orb_array
        internal_options['orth_orb_array'] = 1.0 * orth_orb_array 
      #print("orth_orb in internal_options", internal_options["orth_orb_array"])

      if "proj_mat" in options:
        print("proj_mat from input")
        internal_options['proj_mat'] = options['proj_mat']
      else:
        print("proj_mat generated", flush=True)
        internal_options['proj_mat'] = options['proj_mat'] = cusp_orbitals.get_proj_mat(options['basis_centers'].astype(int), options['orth_orb_array'], all_nuc_xyz, options['Z'], options['basis_orb_type'], options['basis_exp'], options['basis_coeff'])


      #print("\nproj mat:\n", internal_options['proj_mat'] )
      #print("\ncusp radii mat:\n", internal_options['cusp_radii_mat'] )
      #print("\north orb array:\n", internal_options['orth_orb_array'] )
      #print("\nbasis orb type:\n", internal_options['basis_orb_type'] )
      #print("\nbasis orb centers:\n", internal_options['basis_centers'] )

      ################################
      ##### ORTHOGONALIZE BASIS ######
      ################################
      print("====================================================", flush=True)
      print("  Changing C, S, and F to orthgonalized cusped basis", flush=True)
      print("====================================================", flush=True)

      B_mat = cusp_orbitals.orth_transform(internal_options["orth_orb_array"], internal_options['proj_mat'], nnuc, internal_options["basis_centers"].astype(int), internal_options["nbf"])
      B_inv = np.linalg.inv(B_mat) 
      internal_options["B"] = B_mat # change of basis matrix
      #print("B_mat",B_mat)
      print("\nB shape\n", B_mat.shape, flush=True)

      mo_orth=True
      if "mocoeff_orth" in options:
        if options["mocoeff_orth"] == False:
          mo_orth=False
          print("")
          print("mocoeff WILL NOT get orthogonalized\n",np.array2string(internal_options["mocoeff"],separator=','),flush=True)
          print("")

      if mo_orth:
        print("")
        print("mocoeff before orthogonalization\n",np.array2string(internal_options["mocoeff"],separator=','),flush=True)
        print("")

        # Update C matrix to orthogonalized basis and normalize
        #      X' C' = (X B) (B_inv C), defaults to identity 
        internal_options["mocoeff"] = B_inv @ internal_options["mocoeff"]
        internal_options["mocoeff"] = linear_method.norm_each_MO(internal_options["mocoeff"])

      # HELP - does this need to correspond to how I generate or input the mocoeff mat?
      if "pyscf_S" in options and "fock_mat" in options:

        #print("\nfock matrix shape before orthognalization",  options["fock_mat"].shape, flush=True)
        #print("fock_mat before orthogonalization\n",np.array2string(options["fock_mat"],separator=','),flush=True)
        #print("")
        #print("\noverlap matrix shape before orthognalization", options["pyscf_S"].shape, flush=True)
        #print("overlap_mat before orthogonalization\n",np.array2string(options["pyscf_S"],separator=','),flush=True)
        #print("")

        #options["pyscf_1e_energy"] = B_mat.T @ options["pyscf_1e_energy"] @ B_mat
        options["pyscf_S"] = B_mat.T @ options["pyscf_S"] @ B_mat
        options["fock_mat"] = B_mat.T @ options["fock_mat"] @ B_mat
        print("pyscf_S and pyscf_F orthogonalized") #\n", pyscf_s)
        #print("\npyscf_S\n", pyscf_s)
      else:
        pyscf_s, pyscf_1e_energy, fock_mat, pyscf_orbs = cusp_orbitals.pyscf_result_loc(options["Z"], all_nuc_xyz, options["basis_type"], loc=True)

        print("\nfock matrix shape before orthognalization",  options["fock_mat"].shape, flush=True)
        print("fock_mat before orthogonalization\n",np.array2string(options["fock_mat"],separator=','),flush=True)
        print("")
        print("\noverlap matrix shape before orthognalization", options["pyscf_S"].shape, flush=True)
        print("overlap_mat before orthogonalization\n",np.array2string(options["pyscf_S"],separator=','),flush=True)
        print("")

        #options["pyscf_1e_energy"] = B_mat.T @ pyscf_1e_energy @ B_mat
        options["pyscf_S"] = B_mat.T @ pyscf_s @ B_mat
        options["fock_mat"] = B_mat.T @ fock_mat @ B_mat
        print("pyscf_S and pyscf_F generated and orthogonalized") #\n", pyscf_s)
      #print("pyscf_1e\n", pyscf_1e_energy)

      #print("====================================================", flush=True)
      #print(" Changing orthogonalized S and F to localized basis C is in    ", flush=True)
      #print("====================================================", flush=True)
      ## HERE should this mocoeff be orthed already?? 
      #options["pyscf_S"] = options["mocoeff"].T @ pyscf_s @ options["mocoeff"]
      #options["fock_mat"] = options["mocoeff"].T @ fock_mat @ options["mocoeff"]

      if 'nuclei_s' in options:
        pyscf_s_s, pyscf_1e_energy_s, fock_mat_s, pyscf_orbs_s = cusp_orbitals.pyscf_result_loc(options["Z"], cusp_orbitals.get_mol_xyz(options['nuclei_s']) , options["basis_type"], loc=True)
        #options["pyscf_1e_energy_s"] = B_mat.T @ pyscf_1e_energy_s @ B_mat
        options["pyscf_S_s"] = B_mat.T @ pyscf_s_s @ B_mat

        if "HF_mocoeff_s" in options:
          if options["HF_mocoeff_s"] == True:
             print("mocoeff_s defaulting to pyscf HF C matrix at chosen basis")
             options["mocoeff_s"] = pyscf_orbs_s

        if "mocoeff_s" in options:
          internal_options["mocoeff_s"] = B_inv @ options["mocoeff_s"]
        else:
          raise RuntimeError('nuclei_s is in options but no mocoeff_s provided OR HF_mocoeff_s bool assigned in options')

      # initial cusp_a0 - hardcoded to 1's or 0
      internal_options['cusp_a0'] = options["cusp_a0"] = cusp_orbitals.get_a0_matrix(options)
      #if 'cusp_a0' in options:  # default is input
      #    internal_options["cusp_a0"] = options["cusp_a0"] 
      ##else:
      #elif 'get_a0_mat' in options and options['get_a0_mat'] == True: 
      #    print("Calculating new a0_mat")
      #    internal_options['cusp_a0'] = options["cusp_a0"] = cusp_orbitals.get_a0_matrix(options)
      #else:
      #    raise RuntimeError('cusped basis is chosen, must define get_a0_mat (bool) or provide a cusp_a0 in options dictionary')

      ## get list of r orders for the P functions
      #if 'order_n_list' in options:
      #    print("order_n_list:", internal_options["order_n_list"], flush=True)
      #    internal_options["order_n_list"] = np.array(options["order_n_list"]) 
      #else:
      #    raise RuntimeError('cusped basis is chosen, must define order_n_list in options dictionary')

      # get list of r orders for the P functions
      if 'order_n_list' in options:
          n_list = np.zeros(len(options["order_n_list"]))
          for i in range(len(n_list)):
              n_list[i] = float(options["order_n_list"][i])
          internal_options["order_n_list"] = n_list
          print("order_n_list:", internal_options["order_n_list"], flush=True)
          internal_options["num_p_func"] = len(internal_options["order_n_list"])
          print("num_p_func:", internal_options["num_p_func"], flush=True)
      else:
          raise RuntimeError('cusped basis is chosen, must define order_n_list in options dictionary')
      
      if 'cusp_coeff_matrix' in options:
          print("using input cusp_coeff_mat", flush=True)
          internal_options["cusp_coeff_matrix"] = options["cusp_coeff_matrix"]
      else:
          print("calculating cusp_coeff_mat", flush=True)
          if 'cusp_type' not in options:
              raise RuntimeError('need to specify cusp_type in input')
          else:
              internal_options["cusp_type"] = options["cusp_type"]
          internal_options['cusp_coeff_matrix'] = options["cusp_coeff_matrix"] = cusp_orbitals.get_cusp_coeff_matrix(options, stretched=False)

          #if 'scf_a0' in options:
          #  if options['scf_a0']:
          #    ################## TEST a0 SCF CYCLE ###################
          #    scf_dict = cusp_a0_scf(options)
          #    scf_cusp_a0 = scf_dict["cusp_a0_scf"+str(scf_dict["final_iter"])] 
          #    scf_cusp_coeffs = scf_dict["cusp_coeff_matrix_scf"+str(scf_dict["final_iter"])]

          #    internal_options['cusp_coeff_matrix'] = options["cusp_coeff_matrix"] = scf_cusp_coeffs
          #    internal_options['cusp_a0'] = options["cusp_a0"] = scf_cusp_a0
          #    #########################################################

      #print("\nGet info to compare cusp types here", flush=True)
      #options['slater_poly_cusp_coeff_matrix'], options['slater_poly_plus_ratio_cusp_coeff_matrix'] = do_cusp_comp_test(options, internal_options["order_n_list"])

      #########################################################

      #print("\n",options['slater_poly_cusp_coeff_matrix'], "\n", options['slater_poly_plus_ratio_cusp_coeff_matrix'],flush=True)
      #internal_options['cusp_coeff_matrix'][0][0][:] = 0.0
      #internal_options['cusp_coeff_matrix'][0][0][0] = 1.0
      #print("cusp_coeff shape", internal_options["cusp_coeff_matrix"].shape)
      #for i in range(1):
      #    for j in range(9):
      #        for k in range(3):
      #            internal_options["cusp_coeff_matrix"][i,j,k] += i*1 + j*2 + k*3
      #print("printing cusp_coeff_matrix:\n", internal_options["cusp_coeff_matrix"], flush=True)
      #print("printing 012 element of cusp_coeff_matrix:\n", internal_options["cusp_coeff_matrix"][0,1,2], flush=True)
      #sys.exit()

      #print("order_n_list:", internal_options["order_n_list"], flush=True)
      #n_list = np.zeros((1,len(internal_options["order_n_list"]))).reshape([-1])
      #for i in range(len(n_list)):
      #    n_list[i] = float(internal_options["order_n_list"][i]+0.1)

      #internal_options["order_n_list"] = n_list #np.array(internal_options["order_n_list"])
      #print("order_n_list:", internal_options["order_n_list"], flush=True)
      #internal_options["order_n_list"] = np.array([0.0, 2.0, 3.0]) #np.array(internal_options["order_n_list"])
      #print("order_n_list after conversion to array:\n", internal_options["order_n_list"], flush=True)
      #internal_options["num_p_func"] = len(internal_options["order_n_list"]) # add 0.1 to confirm it rounds to correct int on c++ side
      #print("num_p_func:", internal_options["num_p_func"], flush=True)

      if 'cusp_a0_s' in options:
          internal_options['cusp_a0_s'] = options['cusp_a0_s']
      elif 'get_a0_s_mat' in options:
          if options['get_a0_s_mat'] == True:
              internal_options['cusp_a0_s'] = cusp_orbitals.get_a0_matrix(options, stretched=True)
      else:
          raise RuntimeError('cusped basis is chosen, must define get_a0_s_mat (bool) or provide a cusp_a0_s in options dictionary')

      if global_cusp != 0.0:
        ### add cusp test here
        ########################################################################################
        # hack to run the cusp tester
        ########################################################################################
        #
        #  Hi Sonja.  Here is the current hack into
        #  the C++ side.  For now, I'm editing the
        #  internal_options directly, but rather
        #  than mess that up you can create copies
        #  as we discussed and update those copies'
        #  nsamp_per_block, nblock, and cusp_testing_positions.
        #
        #  The values for these are:
        #    nsamp_per_block = 4 * number of (original) nuclei
        #    nblock = 10 * number of threads
        #             (this avoids error traps;  its not actually using threads or blocks)
        #    cusp_testing_positions = 4 positions near each nucleus for use in a regression vs 1/r
        #
        #  Once the C++ side has run, the acc_dict["AccumulatorCuspTestingKE"] array will hold
        #  the kinetic energy of the 0th electron at each of the positions in cusp_testing_positions.
        #  From this, you can do one regression per nucleus (KE vs 1/r using the 4 points)
        #  to check how closely the KE matches Z/r (so, KE = a(1/r) + b, and we want a = Z).
        #  From that, you can figure out how to adjust that nucleus's 1s orbital in that MO so
        #  that the KE diverges as exactly Z/r as it should.
        #  One way to do that would be to repeat this calculation for a few different values of the
        #  1s coeff on the nucleus in question, fit the a values (slopes vs 1/r) to a smooth function,
        #  of that 1s coeff, and interpolate to figure out the 1s coeff value at which a = Z.
        #
        #  From the initial data I've gotten from this, it looks like our hydrogen cusps
        #  might be more off than the oxygen cusps.  So, might be good to fix those too.
        #  Maybe doing this in a sort of self-consistent cycle would work?  
        #     1. Update the 1s MO coeffs for each atom center in each MO.
        #     2. Recalculate the cusp KE vs 1/r regressions
        #     3. Check how close they are to correct
        #     4. If not that close, go back to step 1 and repeat as needed.
        #
        ########################################################################################

        with open(internal_options['file_name']+'_options_dict.txt', 'w') as f:  
            for key, value in options.items():  
                f.write('%s:%s\n' % (key, value))
        pickle.dump(options, open(internal_options['file_name']+'_options_dict.pkl', "wb"))

        print("cusp test func here")
        #do_thru_nuc_test(internal_options, acc_names=["AccumulatorCuspTestingLEAllE",], which_orb_all={"cusp","gaussian"}) #,"slater"}):
        #do_thru_nuc_test(internal_options, acc_names=["AccumulatorCuspTestingLE", "AccumulatorCuspTestingKE", "AccumulatorCuspTestingPsiSquared"], which_orb_all={"cusp","gaussian"}) #,"slater"}):
        #do_1e_cusp_test(internal_options, mo_list=np.arange(0,internal_options["mocoeff"].shape[-1]), acc_names=["AccumulatorCuspTestingKE",], which_orb_all=['cusp', ],savefiles=False)
        #sys.exit()


  #cusp_orbitals.check_cpp_xmat('/home/trine/Desktop/essqc_vmc/tfes-1/essqc/src/essqc/vmc/test_cpp_orbEval_CH4-631G.csv', options)
  ##cusp_orbitals.check_cpp_xmat('/global/scratch/users/trinequady/containerized_tfes/essqc/src/essqc/vmc/CH4/analyze_LM/cpp_Xmat_testVals_v2a_03042024.csv', options)

  ## LINEAR METHOD ###
  if "LMParams" in options and len(options["LMParams"]) > 0:
    name_vp = [] 	# store names that match the parameter key in internal_options
    dim_vp = []		# store dimension of each parameter type

    dim_flag = False
    jss_flag = False
    for i in internal_options["LMParams"]:
      #print("loop through opt params: ", i)
      if dim_flag == False:
        #print("in dim_flag")          
        if i == 'AlphaDet' or i == 'BetaDet' or i == 'AlphaPinDet' or i == 'BetaPinDet':
          #print("\tDeterminant opt!",flush=True)          
          name_vp.append('mocoeff') 
          dim_vp.append(np.shape(options['mocoeff'])) 
          dim_flag = True
      if jss_flag == False:
        #print("in jas_flag")          
        if i == 'RMPJastrowAA' or i == 'RMPJastrowBB':
          #print("\tJastrow same spin opt!",flush=True)          
          name_vp.append('jAss') 
          dim_vp.append(np.shape([internal_options['jAss']]))
          jss_flag = True
      elif i == 'RMPJastrowAB':
          #print("\tJastrow opposite spin opt!",flush=True)          
          name_vp.append('jAos') 
          dim_vp.append(np.shape([internal_options['jAos']]))

    if len(dim_vp) == 0:
      raise RuntimeError("LMParams indicated to optimize, but no dimension of params listed")

    if "dE_schedule" in options:  # else defaults to false and max_delta_p stays fixed
      internal_options["dE_schedule"] = options["dE_schedule"] 
      if internal_options["dE_schedule"]:
        print("dE_schedule chosen in options, using the scheduled set max_delta_p in do_absolute_energy instead of options input")

    if "fock_mat" in options and "pyscf_S" in options:  # orthgonalized 
      #print("\nFock matrix in options",flush=True)
      internal_options["fock_mat"] = options["fock_mat"] 
      internal_options["overlap_mat"] = options["pyscf_S"] 
    else: #if "fock_mat" not in internal_options:
      raise RuntimeError("LMParams indicated to optimize, but no Fock and/or overlap matrix defined in options")
    #  print("\ngenerating Fock matrix",flush=True)
    #  pyscf_s, pyscf_1e_energy, fock_mat, pyscf_orbs = cusp_orbitals.pyscf_result(options["Z"], cusp_orbitals.get_mol_xyz(options['nuclei']), options["basis_type"])
    #  internal_options["fock_mat"] = fock_mat 
    #print("\nFock matrix", internal_options["fock_mat"].shape, "\n", internal_options["fock_mat"], flush=True)
    #print("\nOverlap matrix", internal_options["overlap_mat"].shape, "\n", internal_options["overlap_mat"], flush=True)

    internal_options['name_vp'] = np.array(name_vp)
    internal_options['dim_vp'] = np.array(dim_vp, dtype=object)
    print("")
    print("Optimizable parameters and dimensions: ", name_vp, dim_vp) 
    internal_options['nvp'] = sum(np.prod(dim) for dim in internal_options['dim_vp']) 

    internal_options["iter"] = options["iterations"]
    print("")
    print("Number of iterations: ", internal_options["iter"])

    internal_options['cI'] = options["hamiltonian_shift"]
    print("")
    print("Hamiltonian shift (cI): ", internal_options['cI'])

    if 'max_delta_p' in options:
      internal_options['max_delta_p'] = options['max_delta_p']
    else:
      internal_options['max_delta_p'] = 0.1 # default
    print("")
    print("Max element of proposed parameter update:", internal_options['max_delta_p']) #, " (Implemented if energy diverges, iniitial 1.0 element catch automatically rediagonalizes).")
    print("")

    # if constraining optimization - add more fixed params
    if "do_constrained_opt" in options and options['do_constrained_opt'] == True:
      internal_options["constrained_opt"] = options["do_constrained_opt"]
      print("")
      print("Constrained Optimization: ", internal_options["constrained_opt"])

      if "zeros_fixed" in options:
        internal_options["zeros_fixed"] = options["zeros_fixed"]
        print("")
        print("Zero's fixed in optimization?", internal_options['zeros_fixed'])

        if internal_options['zeros_fixed'] == False :

          if "opt_these_orbs" in options:
            internal_options["opt_these_orbs"] = options["opt_these_orbs"]

          if 'selected_LCAO' in options:
            internal_options["selected_LCAO"] = options["selected_LCAO"] 

            if "epsilon" in options:
              internal_options["epsilon"] = options["epsilon"]
            else:
              raise RuntimeError('LM optimization with selection algorithm chosen, must define epsilon to filter which params to add')

          if len(internal_options["opt_these_orbs"]) == 0 and internal_options["selected_LCAO"] == False:
            raise RuntimeError('LM constrained optimization choosen, must define either zeros_fixed, selected_LCAO, or opt_these_orbs')

    if 'nearest_neighboors' in options:
      internal_options["nearest_neighboors"] = options["nearest_neighboors"]
    else:
      raise RuntimeError('must define nearest_neighboors for LM optimization with selection algorithm chosen OR plotting molecule with bonds')

    if internal_options['selected_LCAO'] == True:

      internal_options["mocoeff"] = linear_method.delta_E_filter_C_mat_unsorted(internal_options["mocoeff"], 
                                                                                internal_options["fock_mat"], 
                                                                                internal_options["overlap_mat"], 
                                                                                options["epsilon"]) 
      print("")
      print("selectedLCAO = True, \nFiltering initial C_mat by epsilon (was eta) =", options["epsilon"], ", non-zero = ", linear_method.print_percent_of_mat_greater(internal_options["mocoeff"], options["epsilon"], print_val=False), flush=True)
    
    # dont include test yet 
    print("-----------------------------------------------")
    print("\n\tOPTIMIZATION of ", internal_options['nvp'], " parameters\n")

    internal_options["fixed_param_ind"] = linear_method.set_param_ind(internal_options["mocoeff"], 
                                                                      internal_options["Z"], 
                                                                      internal_options["basis_centers"], 
                                                                      internal_options["nearest_neighboors"], 
                                                                      internal_options["constrained_opt"],
                                                                      internal_options["zeros_fixed"],
                                                                      internal_options["selected_LCAO"],
                                                                      internal_options["opt_these_orbs"],
                                                                      add_neighboors=False) # add_neighboors applies only if selection is turned on

  # save BOTH dictionaries to file and pkl
  with open(internal_options['file_name']+'_options_dict.txt', 'w') as f:  
      for key, value in options.items():  
          f.write('%s:%s\n' % (key, value))
  pickle.dump(options, open(internal_options['file_name']+'_options_dict.pkl', "wb"))

  with open(internal_options['file_name']+'_internal_options_dict.txt', 'w') as f:  
      for key, value in internal_options.items():  
          f.write('%s:%s\n' % (key, value))
  pickle.dump(internal_options, open(internal_options['file_name']+'_internal_options_dict.pkl', "wb"))

  if "LMParams" in options and len(options["LMParams"]) > 0:
    print("Run LM grad test here", flush=True)
    #do_orb_deriv_test(internal_options['file_name']+'_internal_options_dict.pkl')
    print()
    do_LM_grad_test(internal_options['file_name']+'_internal_options_dict.pkl')
    #sys.exit()

  # get the batch size (the number of samples per block)
  batch_size = internal_options["nsamp_per_block"]

  # get the total number of samples over all blocks
  nsamp = internal_options["nblock"] * internal_options["nsamp_per_block"]

  # check that the batch size divides the number of samples evenly
  if nsamp % batch_size != 0:
    raise RuntimeError('sample size should be divisible by batch size')

  # get number of batches
  nbatch = internal_options["nblock"]
  #nbatch = nsamp // batch_size
  if nbatch < 10:
    raise RuntimeError('you need to have at least 10 batches (nblock) so we can do statistics')

  # we might be doing a relative energy (correlated sampling) calculation
  if internal_options["do_corr_samp"]:

    # if we are doing a batched calculation
    if internal_options["do_batch"]:
      do_relative_energy(internal_options, internal_options_s)
    # if we are doing a vanilla calculation
    else:
      #if internal_options["do_sherman_morrison"]: need to add correlated sampling functionality to SM math
      do_relative_energy_vanilla(internal_options, internal_options_s)

  # or we might be doing an absolute energy calculation
  else:
    # if we are doing a batched calculation
    if internal_options["do_batch"]:			## HELP SHOULDNT THIS BE IF WE ARE DOIN SM, BATCHING, OR CORRELATED SAMPLING???
      do_absolute_energy_multipole(internal_options)
    # if we are doing vanilla calculation
    else:
      do_absolute_energy(internal_options)
	      #print("check internal options being passed to do_absolute_energy: \n", internal_options["do_corr_samp"], "\n", internal_options["do_batch"], "\n", internal_options["do_sherman_morrison"])

