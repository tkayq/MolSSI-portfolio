#ifndef ESSQC_VMC_WFF_DET_H
#define ESSQC_VMC_WFF_DET_H

#include "wfn_factor_base.hpp"
#include <iomanip>

namespace essqc
{

  namespace qmc
  {

    /**
     * @brief A child class for the determinant pieces and their LM derivatives (will be used for both alpha and beta)
     *
     */
    class ABDeterminant : public WnfFactorBase
    {

    // protected member data
    protected:

      // spin of first electrons
      int m_e1s;

      // position of the first electrons
      const PosVec &m_e1pos;

      // position of nuclei
      const PosVec &m_nuc;

      // import coefficient matrix
      std::vector<double> m_C;

      // number of alpha elec
      int m_na;

      // number of orbies
      int m_no;

      // whether we are using slater AOs
      bool m_use_STO;

      // whether we are using hydrogenic AOs
      bool m_use_HAO;

      // whether we are using gaussian AOs
      bool m_use_GTO;

      // whether we are using orbital cusps
      bool m_use_cusp;

      // whether we are only collecting slater (from cusp) func derivs
      bool m_get_slater_derivs;

      // stores which basis we are using
      std::string m_basis_type;

      // number of GTOs per bf
      int m_ng;

      // number of nuclei
      int m_num_nuc;

      // Z array
      std::vector<double> m_Z;

      // matrix of gaussian cusp cutoff radii (# nuc x # AO)
      std::vector<double> m_cusp_radius_mat;

      // matrix of gaussian cusp parameters: a_0
      std::vector<double> m_cusp_a0_mat;

      // tensor of P basis function coefficients (# nuc x # AO x # nbf)
      std::vector<double> m_cusp_coeff_mat;

      // vector of orders of n for P basis functions
      std::vector<double> m_n_vec;

      // matrix of projection values for orthogonalized orbs
      std::vector<double> m_proj_mat;

      // array of basis set coefficients
      std::vector<double> m_bf_coeff;

      // array of basis set exponents
      std::vector<double> m_bf_exp;

      // array of basis set center
      std::vector<int> m_bf_cen;

      // array of basis orbital types
      std::vector<int> m_bf_orbs;

      // vector of assigning which orbs get orthogonalized (val based on orb_ind+1 they get orth against)
      std::vector<int> m_orth_orb;

      // current X matrix
      std::vector<double> m_x;

      // current XC
      std::vector<double> m_xc;

      // current XC inverse
      std::vector<double> m_xci;

      // proposed new row for X matrix
      std::vector<double> m_pxrow;

      // proposed new row for XC matrix
      std::vector<double> m_pxcrow;

      // v^T (XC)^-1u thing
      std::vector<double> m_vtxci;

      // workspace to hold temporary things
      std::vector<double> m_workspace;    // used for compute_ke_pieces
      std::vector<double> m_workspace_2;  // same use as m_workspace but for compute_grad_E
      std::vector<double> m_ddvworkspace; // for compute_grad_E - not sure if needed or can use m_workspace above

      // log of determinant
      double m_logdet;

      // determinant sign
      double m_detsign;

      // vector to hold indices of active electrons in a batch (just putting it here for now)
      std::vector<int> m_active_e;

      // gaussian cusp cutoff radius
      double m_cusp_radius;

      // protected member functions
    protected:

      // public member functions
    public:

      /**
       * @brief Test constructor - bare minimum
       */
      ABDeterminant(const int nelec, const SingleElectronInfo &mei, const int e1s, const PosVec &nucpos, const PosVec &e1pos)
          : WnfFactorBase(nelec, 0, mei),
            m_e1s(e1s),
            m_nuc(nucpos),
            m_e1pos(e1pos){}

      /**
       * @brief Construct the determinant (alpha or beta)
       *
       * @param nelec total number of electrons (alpha + beta, at least for now)
       * @param nvp = number of basis functions (orbitals) * number of alpha electrons, dimensions of C matrix
       * @param mei reference to an object containing the spin, index, and position of the moved electron
       */
      ABDeterminant(const int nelec, const SingleElectronInfo &mei, const int e1s, const PosVec &nucpos, const PosVec &e1pos, py::dict &options, py::dict &acc_dict)
          : WnfFactorBase(nelec, py::extract<int>(options["nbf"]) * e1pos.nparticles(), mei),
            m_logdet(0.0),
            m_detsign(0.0),
            m_e1s(e1s),
            m_nuc(nucpos),
            m_e1pos(e1pos)
      {

        int no = py::extract<int>(options["nbf"]);
        m_no = no;
        m_na = m_e1pos.nparticles();
        m_num_nuc = m_nuc.nparticles();

        // wavefunction pieces in the determinant: psi = e^J det(XC)
        m_x.assign(m_na * m_no, 0.0);
        m_xc.assign(m_na * m_na, 0.0);
        m_xci.assign(m_na * m_na, 0.0);
        m_pxrow.assign(m_no, 0.0);
        m_pxcrow.assign(m_na, 0.0);
        m_vtxci.assign(m_na, 0.0);

        // NEXT: extract the input information provided in the python dictionary input
        //       convert all numpy arrays as column-major vectors
        //       arrays involving cusps or the linear method (LM) pieces will be len 0 
        //       if cusps or LM optimization is turned off


        //////////////////////////////
        ///////// Basis Info /////////
        ////////////////////////////// 

        const int ng = py::extract<int>(options["ng"]);
        m_ng = ng;

        m_basis_type = py::extract<std::string>(options["basis_type"]);

        m_use_HAO = py::extract<bool>(options["useHAO"]);
        m_use_STO = py::extract<bool>(options["useSTO"]);
        m_use_GTO = py::extract<bool>(options["useGTO"]);
        m_use_cusp = py::extract<bool>(options["useCuspGTO"]);

        m_get_slater_derivs = py::extract<bool>(options["get_slater_derivs_cusp"]);

        np::ndarray bf_exp = py::extract<np::ndarray>(options["basis_exp"]);
        const double *const bf_exp_ptr = reinterpret_cast<const double *>(bf_exp.get_data());
        m_bf_exp.assign(m_no * m_ng, 0.0);
        for (int p = 0; p < m_no; p++)
          for (int i = 0; i < m_ng; i++)
            m_bf_exp[p + i * m_no] = bf_exp_ptr[p * m_ng + i];

        np::ndarray bf_coeff = py::extract<np::ndarray>(options["basis_coeff"]);
        const double *const bf_coeff_ptr = reinterpret_cast<const double *>(bf_coeff.get_data());
        m_bf_coeff.assign(m_no * m_ng, 0.0);
        for (int p = 0; p < m_no; p++)
          for (int i = 0; i < m_ng; i++)
            m_bf_coeff[p + i * m_no] = bf_coeff_ptr[p * m_ng + i];

        np::ndarray bf_cen = py::extract<np::ndarray>(options["basis_centers"]);
        const double *const bf_cen_ptr = reinterpret_cast<const double *>(bf_cen.get_data());
        m_bf_cen.assign(m_no, 0.0);
        for (int p = 0; p < m_no; p++)
        {
          for (int i = 0; i < 1; i++)
          {
            m_bf_cen[p + i * m_no] = bf_cen_ptr[p * 1 + i];
          }
        }

        np::ndarray bf_orbs = py::extract<np::ndarray>(options["basis_orb_type"]);
        const double *const bf_orbs_ptr = reinterpret_cast<const double *>(bf_orbs.get_data());
        m_bf_orbs.assign(m_no, 0.0);
        for (int p = 0; p < m_no; p++)
        {
          for (int i = 0; i < 1; i++)
          {
            m_bf_orbs[p + i * m_no] = bf_orbs_ptr[p * 1 + i];
          }
        }

        // get MO coefficients
        np::ndarray C = py::extract<np::ndarray>(options["mocoeff"]);
        const double *const C_ptr = reinterpret_cast<const double *>(C.get_data());
        m_C.assign(m_na * m_no, 0.0);
        for (int p = 0; p < m_no; p++)
        {
          for (int i = 0; i < m_na; i++)
          {
            m_C[p + i * m_no] = C_ptr[p * m_na + i]; // store from row major to column major
          }
        }

        // get Z array - array of nuclear charges
        np::ndarray Z = py::extract<np::ndarray>(options["Z"]);
        const double *const Z_ptr = reinterpret_cast<const double *>(Z.get_data());
        m_Z.assign(m_num_nuc, 0.0);
        for (int p = 0; p < m_num_nuc; p++)
        {
          for (int i = 0; i < 1; i++)
          {
            m_Z[p] = Z_ptr[p * 1 + i];
          }
        }

        //////////////////////////////
        ///// Cusp Specific Info /////
        ////////////////////////////// 

        // the number of linear combination of slaters that make the cusped orbital
        int num_orb_cusp = 0;
        int num_nuc_cusp = 0;
        int npbf = 0;
        if (m_use_cusp)
        {
          num_orb_cusp = m_no;
          num_nuc_cusp = m_num_nuc;
          npbf = py::extract<int>(options["num_p_func"]);
        }

        // provide indices for which orbitals are orthogonalized (and what index they are orthogonalized with respect to)
        //   only needed for cusping to get uniform curvature of the n>1 s-orbitals for a transferable cusp radius
        np::ndarray orth_orb = py::extract<np::ndarray>(options["orth_orb_array"]);
        const double *const orth_orb_ptr = reinterpret_cast<const double *>(orth_orb.get_data());
        m_orth_orb.assign(m_no, 0.0);
        for (int p = 0; p < m_no; p++)
        {
          for (int i = 0; i < 1; i++)
          {
            m_orth_orb[p + i * m_no] = orth_orb_ptr[p * 1 + i];
          }
        }

        // nucleus x orbital matrix of the cusp radius for each pair
        np::ndarray cusp_radii_mat = py::extract<np::ndarray>(options["cusp_radii_mat"]);
        const double *const cusp_radii_mat_ptr = reinterpret_cast<const double *>(cusp_radii_mat.get_data());
        m_cusp_radius_mat.assign(num_nuc_cusp * num_orb_cusp, 0.0);
        for (int p = 0; p < num_nuc_cusp; p++)
          for (int i = 0; i < num_orb_cusp; i++)
            m_cusp_radius_mat[p + i * num_nuc_cusp] = cusp_radii_mat_ptr[p * num_orb_cusp + i];

        // nucleus x orbital matrix of 0s or 1s indicating if the pair has a cusp correction - 0 for if far away or on a p-orbital node 
        np::ndarray cusp_a0 = py::extract<np::ndarray>(options["cusp_a0"]);
        const double *const cusp_a0_ptr = reinterpret_cast<const double *>(cusp_a0.get_data());
        m_cusp_a0_mat.assign(num_nuc_cusp * num_orb_cusp, 0.0);

        for (int p = 0; p < num_nuc_cusp; p++)
        {
          for (int i = 0; i < num_orb_cusp; i++)
          {
            m_cusp_a0_mat[p + i * num_nuc_cusp] = cusp_a0_ptr[p * num_orb_cusp + i];
          }
        }

        // npbf cusping parameters in the linear combo of slaters optimized on python side for each nucleus/orbital pair 
        np::ndarray cusp_coeff = py::extract<np::ndarray>(options["cusp_coeff_matrix"]);
        const double *const cusp_coeff_ptr = reinterpret_cast<const double *>(cusp_coeff.get_data());
        const int dim = num_nuc_cusp * num_orb_cusp * npbf;
        m_cusp_coeff_mat.assign(dim, 0.0);
        for (int p = 0; p < num_nuc_cusp; p++)
        {
          for (int i = 0; i < num_orb_cusp; i++)
          {
            for (int j = 0; j < npbf; j++)
            {
              m_cusp_coeff_mat[p * npbf + i * num_nuc_cusp * npbf + j] = cusp_coeff_ptr[j + p * num_orb_cusp * npbf + i * npbf];
            }
          }
        }

        // slater polynomial degrees included in the linear combination of slater function that make up the cusp
        np::ndarray n_vec_list = py::extract<np::ndarray>(options["order_n_list"]);
        const double *const n_vec_list_ptr = reinterpret_cast<const double *>(n_vec_list.get_data());
        m_n_vec.assign(npbf, 0.0);
        for (int i = 0; i < npbf; i++)
        {
          m_n_vec[i] = n_vec_list_ptr[i];
        }

        // get matrix of projection elements to evaluate the orthogonalized MO basis if cusping
        np::ndarray proj_mat = py::extract<np::ndarray>(options["proj_mat"]);
        const double *const proj_mat_ptr = reinterpret_cast<const double *>(proj_mat.get_data());
        m_proj_mat.assign(num_orb_cusp * num_orb_cusp, 0.0);
        for (int p = 0; p < num_orb_cusp; p++)
        {
          for (int i = 0; i < num_orb_cusp; i++)
          {
            m_proj_mat[p + i * num_orb_cusp] = proj_mat_ptr[p * num_orb_cusp + i];
          }
        }

      } // end constructor

      ///////////////////////////////////////////////////////
      //// Wavefunction evaluation and MC step functions ////
      ///////////////////////////////////////////////////////

      // number of variational parameters for LM - num_num * num_aos if optimizing orbitals is turned on
      int get_num_of_vp() const;

      // evaluate electron x orbital matrix inside wfn determinant
      void get_full_X_mat(const PosVec &e_pos, const PosVec &n_pos, std::vector<double> &xmat);

      // initialize internal data (if any) at the particle positions
      void initialize_internal_data();

      int print_spin();

      void copy_m_active(std::vector<int> &m_act_e);

      // evaluate ratio of new to old value for the proposed move
      double get_new_to_old_ratio();

      // update internal data after the one electron move is accepted
      void accept_poposed_move();

      // function to calculate KE pieces
      //  REMINDER OF HOW WE DO K.E.                                                          //
      //                                                                                      //
      //     __2                            /  __          \     /  __          \             //
      //     \/  Psi       __2             |   \/ log(Psi)  | . |   \/ log(Psi)  |            //
      //    ---------  =   \/  log(Psi) +  |                |   |                |            //
      //       Psi                          \              /     \              /             //
      //                                                                                      //

      // function for orbital derivatives - DOES NOT POPULATE orbs
      void get_orbs_and_derivs(const PosVec &epos, const PosVec &npos, double *orbs, double **der1, double **der2);

      void compute_psi(double &psi);
      void compute_psi_piece_for_ratio(double &psi);

      // evaluate this factor's contribution to nabla(log(Psi)) and grad(log(Psi)) and add them to the supplied totals
      void compute_ke_pieces(double *const d1logx, double *const d2logx);

      // evaluate and populate orbs (column major nelec x norb, alpha then beta), der1, der2 (for derivs: column major 3*[nelec x norb, alpha then beta] for x y and z)
      void orbs_and_derivs_test(double *orbs, double **der1, double **der2, const double fd_delta);

      ////////////////////////////////////////////
      //// Evaluation of LINEAR METHOD pieces ////
      ////////////////////////////////////////////

      // linear method: function to calculate Psi_v/Psi_0
      void compute_PvP0(double *const PvP0, double &ratio, int indx); // for all pairs

      // evaluate derivative pieces for the linear method grad_E vector
      void compute_grad_E(double * const grad_E, double * const grad_log_psi, int indx); 
        
      // compute LM and derivatives (currently ddv_grad_log_psi) values to check derivatives against finite difference test
      void LM_deriv_test(double * const grad_log_psi, double ** ddv_gradlogpsi, double * ddv_laplogpsi, double ** der1, double ** der2, const double fd_delta);

      // evaluate derivative pieces for the linear method grad_E vector
      void compute_ddv_dnlogpsi(double ** ddv_gradlogpsi, double * ddv_laplogpsi, double ** der1, double ** der2, int indx);

      // do math for the ith elec outer product to go to d/d_cij d^n/dmu^n log(psi) for linear method grad_E vector
      void outerprod_to_ddv_dnlogpsi(double * A_naxno, double * ddv_dnlogpsi);

      void matprint(const int n, const int m, const double *mat, const std::string &fmt);

      ///////////////////////////////////
      //// ORBITAL CUSPING FUNCTIONS ////
      ///////////////////////////////////

      // switching function for cusped oritals
      double b_func(double cusp_radius, double nuc_r);
      // vector of the gradient of the switching function for cusped orbitals
      void d1_b_func(double cusp_radius, double nuc_r, int i, int n, double diff[3], double d1_b[3]);
      // vector of the laplacian elements of the switching function for cusped orbitals
      void d2_b_func(double cusp_radius, double nuc_r, int i, int n, double diff[3], double d2_b[3]);

      ///// orbital basis functions //////

      // slater s orbital for gaussian cusps
      double slater_s_cusp(double a0, double zeta, double r);
      // NON-orthogonalized gaussian s orbital value
      double STOnG_s(const double r, const double pi, int ao_ind);
      // orthogonalized gaussian s orbital value
      double STOnG_s(const double r, const double pi, int ao_ind, int core_ao_ind, double proj);

      // NON-orthogonalized gaussian p orbital value
      double STOnG_p(const double r, const double dist_xi, const double pi, int ao_ind);

      // d orbital value
      double STOnG_d(const double dist, double (&dist_xyz)[3], const double pi, int ao_ind, int ao_type);

      ///// orbital derivative info /////

      // vector of gradient of the slater s orbital for gaussian cusps
      void d1_slater_s_cusp(double a0, double zeta, double r, int i, int n, double diff[3], double d1_Q_vec[3]);
      // vector of laplacian elements of the slater s orbital for gaussian cusps
      void d2_slater_s_cusp(double a0, double zeta, double r, int i, int n, double diff[3], double d2_Q_vec[3]);

      // vector of UN-cusped NON-orthogonalized gaussian gradient of s orbital for ith electron in pth orbital
      void d1_STOnG_s(const double r, const double pi, int ao_ind, const double delta[3], double d1_vec[3]);
      // vector of gradient orthogonalized gaussian of s orbital for elctron i in orb p
      void d1_STOnG_s(const double r, const double pi, int ao_ind, int core_ao_ind, double proj, const double delta[3], double d1_vec[3]);

      // vector of laplacian UN-cusped NON-orthogonalized gaussian of s orbital for ith electron in pth orbital
      void d2_STOnG_s(const double r, const double pi, int ao_ind, const double delta[3], double d2_vec[3]);
      // vector of laplacian element of the orthogonalized gaussian of s orbital for elctron i in orb p
      void d2_STOnG_s(const double r, const double pi, int ao_ind, int core_ao_ind, double proj, const double delta[3], double d2_vec[3]);

      // vector of NON-orthogonalized gaussian gradient of p orbital for ith electron in pth orbital
      void d1_STOnG_p(const double r, const double pi, int ao_ind, const double delta[3], double d1_vec[3]);
      // vector of laplacian NON-orthogonalized gaussian of p orbital for ith electron in pth orbital
      void d2_STOnG_p(const double r, const double pi, int ao_ind, const double delta[3], double d2_vec[3]);

      // vector of gradient of d orbital
      void d1_STOnG_d(const double dist, double (&delta)[3], const double pi, int ao_ind, int ao_type, double d1_vec[3]);
      // vector of laplacian of d orbitals
      void d2_STOnG_d(const double dist, double (&delta)[3], const double pi, int ao_ind, int ao_type, double d2_vec[3]);

      // vector of gradient of (1-b)X term
      void d1_one_minus_b(double &orb_total, double &b_val, double (&der1_total)[3], double (&d1_b_vec)[3], double (&d1_Pn_vec)[3]);
      // vector of laplacian of (1-b)X term
      void d2_one_minus_b(double &orb_total, double &b_val, double (&der1_total)[3], double (&d1_b_vec)[3], double (&der2_total)[3], double (&d2_b_vec)[3], double (&d2_Pn_vec)[3]);

      // basis function for new cusped basis
      double Pn_func(double b, double Q, double orb_total, double nuc_dist, double n, double rc);
      // vector of gradient of new P basis function
      void d1_Pn_func(double &orb_total, double &b_val, double &Q_fn, double (&der1_total)[3], double (&d1_b_func)[3], double (&d1_slater_s_cusp)[3], double &nuc_dist, double (&diff)[3], double &n, double &qn, double (&d1_P_vec)[3], double &rc);
      // vector of laplaciam of new P basis function
      void d2_Pn_func(double &orb_total, double &b_val, double &Q_fn, double (&der1_total)[3], double (&d1_b_func)[3], double (&d1_slater_s_cusp)[3], double (&der2_total)[3], double (&d2_b_func)[3], double (&d2_slater_s_cusp)[3], double &nuc_dist, double (&diff)[3], double &n, double &qn, double (&d2_P_vec)[3], double &rc);

      ////// general orbital //////

      // fill xmat with slater orbitals - UN-cusped NON-orthogonal
      void slater_orb(const PosVec &e_pos, const PosVec &n_pos, const double pi, std::vector<double> &xmat);

    }; // ABDeterminant class

    // reminants from Erics example on how to make a C++ unittest
    void slater_orb_func_for_ABDeterminant(const PosVec &e_pos, const PosVec &n_pos, const double pi, std::vector<double> &xmat,
                                           const std::vector<int> &bf_orbs, const std::vector<double> &bf_exp,
                                           const std::vector<int> &bf_cen,
                                           const int no);

  } // end namespace qmc

} // end namespace essqc

#endif
