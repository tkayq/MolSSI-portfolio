#include <typeinfo>
#include <iostream> // include for printing
#include <cmath>
#include <boost/format.hpp>
#include <essqc/exception.h>
#include <essqc/lapack_blas_interface.h>
#include "wff_1_det.hpp"

int essqc::qmc::ABDeterminant::get_num_of_vp() const
{
	// currently hardcoded in the constructor via the parent class arguments to be na*no
	return m_nvp;
}

// switching function for cusped orbitals - PASSED
double essqc::qmc::ABDeterminant::b_func(double cusp_radius, double nuc_dist)
{

	// 5th order polynomial math    

	// cusp switching function parameters
	const double c1 = (-6.0 / pow(cusp_radius, 5));
	const double c2 = (15.0 / pow(cusp_radius, 4));
	const double c3 = (-10.0 / pow(cusp_radius, 3));
	const double c4 = 0.0;
	const double c5 = 0.0;
	const double c6 = 1.0;

	const double abs_nuc_dist = std::abs(nuc_dist);

	const double b_val = c1 * pow(abs_nuc_dist, 5) + c2 * pow(abs_nuc_dist, 4) + c3 * pow(abs_nuc_dist, 3) + c4 * pow(abs_nuc_dist, 2) + c5 * abs_nuc_dist + c6;

	return b_val;
}

// vector of the gradient of the switching function for cusped orbitals --- PASSED
void essqc::qmc::ABDeterminant::d1_b_func(double cusp_radius, double nuc_dist, int i, int n, double diff[3], double d1_b[3])
{

	// 5th order polynomial math
	const double c1 = (-6.0 / pow(cusp_radius, 5));
	const double c2 = (15.0 / pow(cusp_radius, 4));
	const double c3 = (-10.0 / pow(cusp_radius, 3));
	const double c4 = 0.0;
	const double c5 = 0.0;
	const double c6 = 1.0;

	// loop through xyz to calc derivative terms
	for (int l = 0; l < 3; l++)
	{
		d1_b[l] = diff[l] * (5 * c1 * pow(nuc_dist, 3) + 4 * c2 * pow(nuc_dist, 2) + 3 * c3 * nuc_dist + 2 * c4 + c5 / nuc_dist); 
	}
}

// vector of the laplacian elements of the switching function for cusped orbitals --- PASSED 
void essqc::qmc::ABDeterminant::d2_b_func(double cusp_radius, double nuc_dist, int i, int n, double diff[3], double d2_b[3])
{

	// 5th order polynomial math
	const double c1 = (-6.0 / pow(cusp_radius, 5));
	const double c2 = (15.0 / pow(cusp_radius, 4));
	const double c3 = (-10.0 / pow(cusp_radius, 3));
	const double c4 = 0.0;
	const double c5 = 0.0;
	const double c6 = 1.0;

	// loop through xyz to calc derivative terms
	for (int l = 0; l < 3; l++)
	{
		d2_b[l] = c1 * (5 * pow(nuc_dist, 3) + 15 * pow(diff[l], 2) * nuc_dist) + c2 * (4 * pow(nuc_dist, 2) + 8 * pow(diff[l], 2)) + 3 * c3 * (nuc_dist + pow(diff[l], 2) / nuc_dist) + 2 * c4 + c5 * (1 / nuc_dist - pow(diff[l], 2) / pow(nuc_dist, 3));
	}
}

// vector of gradient of (1-b)X term
void essqc::qmc::ABDeterminant::d1_one_minus_b(double & orb_total, double & b_val, double (&der1_total)[3], double (&d1_b_vec)[3], double (&d1_Pn_vec)[3]) {
	for (int l = 0; l < 3; l++) {
		d1_Pn_vec[l] += (1.0 - b_val) * der1_total[l] - d1_b_vec[l] * orb_total;  
	}
} 

// vector of laplacian of (1-b)X term
void essqc::qmc::ABDeterminant::d2_one_minus_b(double & orb_total, double & b_val, double (&der1_total)[3], double (&d1_b_vec)[3], double (&der2_total)[3], double (&d2_b_vec)[3], double (&d2_Pn_vec)[3]) {
	for (int l = 0; l < 3; l++) {
		d2_Pn_vec[l] += (1.0 - b_val) * der2_total[l] - 2.0 * d1_b_vec[l] * der1_total[l] - d2_b_vec[l] * orb_total;  
	}
} 

///// orbital basis functions /////

// slater s orbital for gaussian cusps - CHECKED
double essqc::qmc::ABDeterminant::slater_s_cusp(double a0, double zeta, double r)
{
	double Q_fn = a0 * std::exp(-zeta * r);
	return Q_fn;
}

// NON-orthogonalized gaussian s orbital value
double essqc::qmc::ABDeterminant::STOnG_s(const double r, const double pi, int ao_ind)
{
	//std::cout << " ~~STOnG_s~~ ";
	double orb_total = 0.0;
	for (int k = 0; k < m_ng; k++)
	{
		double a = m_bf_exp[k * m_no + ao_ind];
		double d = m_bf_coeff[k * m_no + ao_ind];

		orb_total += d * pow(((2.0 * a) / pi), 0.75) * std::exp(-a * r * r);
	}

	return orb_total;
}

// orthogonalized gaussian s orbital value
double essqc::qmc::ABDeterminant::STOnG_s(const double r, const double pi, int ao_ind, int core_ao_ind, double proj)
{
	double orb_total = 0.0;
	orb_total = STOnG_s(r, pi, ao_ind) - proj * STOnG_s(r, pi, core_ao_ind);
	return orb_total;
}

// NON-orthogonalized gaussian p orbital value
double essqc::qmc::ABDeterminant::STOnG_p(const double r, const double dist_xi, const double pi, int ao_ind)
{
	double orb_total = 0.0;
	for (int k = 0; k < m_ng; k++)
	{
		double a = m_bf_exp[k * m_no + ao_ind];
		double d = m_bf_coeff[k * m_no + ao_ind];
		double Ns = d * pow((2.0 * a / pi), 0.75);
		orb_total += Ns * 2.0 * std::sqrt(a) * dist_xi * std::exp(-a * r * r);
	} // contracted gaussians
	return orb_total;
}

double essqc::qmc::ABDeterminant::STOnG_d(const double dist, double (&dist_xyz)[3], const double pi, int ao_ind, int ao_type) {

  double orb_total = 0.0;
  for (int k = 0; k < m_ng; k++)
  {
    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind]; 
    double N = d * pow(2048.0 * pow(a,7.0) / pow(pi, 3.0), 0.25);
    
    // 3dxy
    if (ao_type == 5) {
      orb_total += N * dist_xyz[0] * dist_xyz[1] * std::exp(-a * dist * dist);
    }
    // 3dyz
    else if (ao_type == 6) {
      orb_total += N * dist_xyz[1] * dist_xyz[2] * std::exp(-a * dist * dist);
    }
    // 3dz^2
    else if (ao_type == 7) {
      double M = pow(144.0, -0.25);
      orb_total += M * N * (3 * pow(dist_xyz[2], 2.0) - pow(dist, 2.0)) * std::exp(-a * dist * dist);
    }
    // 3dxz
    else if (ao_type == 8) {
      orb_total += N * dist_xyz[0] * dist_xyz[2] * std::exp(-a * dist * dist);
    }
    // 3dx^2-y^2
    else {
      double M = 0.5;
      orb_total += M * N * (pow(dist_xyz[0], 2.0) - pow(dist_xyz[1], 2.0)) * std::exp(-a * dist * dist);
    }
  } // contracted gaussians
  
  return orb_total;
}

///// orbital derivative info /////

// vector of gradient of the slater s orbital for gaussian cusps --- CHECKED
void essqc::qmc::ABDeterminant::d1_slater_s_cusp(double a0, double zeta, double r, int i, int n, double diff[3], double d1_Q_vec[3])
{
	for (int l = 0; l < 3; l++)
	{
		d1_Q_vec[l] = -(a0 * zeta * diff[l] * std::exp(-zeta * r)) / r;
	}
}

// vector of laplacian elements of the slater s orbital for gaussian cusps --- CHECKED
void essqc::qmc::ABDeterminant::d2_slater_s_cusp(double a0, double zeta, double r, int i, int n, double diff[3], double d2_Q_vec[3])
{
	for (int l = 0; l < 3; l++) // xyz
	{
		d2_Q_vec[l] = a0 * zeta * std::exp(-zeta * r) * ((zeta * pow(diff[l], 2)) / pow(r, 2) + pow(diff[l], 2) / pow(r, 3) - 1 / r);
	}
}

// basis function for new cusped basis
double essqc::qmc::ABDeterminant::Pn_func(double b, double Q, double orb_total, double nuc_dist, double n, double rc) {
	return b * Q * pow(nuc_dist / rc, n);
}

// vector of gradient of new P basis function
void essqc::qmc::ABDeterminant::d1_Pn_func(double & orb_total, double & b_val, double & Q_fn, double (&der1_total)[3], double (&d1_b_func)[3], double (&d1_slater_s_cusp)[3], double & nuc_dist, double (&diff)[3], double & n, double & qn, double (&d1_P_vec)[3], double & rc) {
	for (int l = 0; l < 3; l++) {
		d1_P_vec[l] += qn * ( b_val * ( d1_slater_s_cusp[l] * pow(nuc_dist / rc, n) + n * Q_fn * pow(nuc_dist / rc, (n-1.0)) * diff[l] / (nuc_dist * rc)) + d1_b_func[l] * ( Q_fn * pow(nuc_dist / rc, n) )); 
	}
}

// vector of laplacian of new P basis function
void essqc::qmc::ABDeterminant::d2_Pn_func(double & orb_total, double & b_val, double & Q_fn, double (&der1_total)[3], double (&d1_b_func)[3], double (&d1_slater_s_cusp)[3], double (&der2_total)[3], double (&d2_b_func)[3], double (&d2_slater_s_cusp)[3], double & nuc_dist, double (&diff)[3], double & n, double & qn, double (&d2_P_vec)[3], double & rc) {
	for (int l = 0; l < 3; l++) {
		d2_P_vec[l] += qn * ( b_val * ( d2_slater_s_cusp[l] * pow(nuc_dist / rc, n) + 2.0 * d1_slater_s_cusp[l] * n / rc * pow(nuc_dist / rc, (n-1.0)) * diff[l] / nuc_dist + Q_fn * n * (n-1.0) * pow(nuc_dist / rc, (n-2.0)) * pow(diff[l], 2) / pow(nuc_dist, 2) / pow(rc, 2) + n / rc * Q_fn * pow(nuc_dist / rc, (n-1.0)) * (1.0 / nuc_dist - pow(diff[l], 2) / pow(nuc_dist, 3)) ) + 2.0 * d1_b_func[l] * ( d1_slater_s_cusp[l] * pow(nuc_dist / rc, n) + Q_fn * n / rc * pow(nuc_dist / rc, (n-1.0)) * diff[l] / nuc_dist) + d2_b_func[l] * ( Q_fn * pow(nuc_dist / rc, n)));
	}
}

// vector of gradient UN-cusped NON-orthogonalized gaussian of s orbital for elctron i in orb p
void essqc::qmc::ABDeterminant::d1_STOnG_s(const double r, const double pi, int ao_ind, const double delta[3], double d1_vec[3]) 
{
	for (int k = 0; k < m_ng; k++)
	{
		double a = m_bf_exp[k * m_no + ao_ind];
		double d = m_bf_coeff[k * m_no + ao_ind];

		d1_vec[0] += -delta[0] * 2.0 * a * d * pow((2.0 * a / pi), 0.75) * std::exp(-a * r * r);
		d1_vec[1] += -delta[1] * 2.0 * a * d * pow((2.0 * a / pi), 0.75) * std::exp(-a * r * r);
		d1_vec[2] += -delta[2] * 2.0 * a * d * pow((2.0 * a / pi), 0.75) * std::exp(-a * r * r);
	} // contracted gaussians
}

// vector of gradient orthogonalized gaussian of s orbital for elctron i in orb p
void essqc::qmc::ABDeterminant::d1_STOnG_s(const double r, const double pi, int ao_ind, int core_ao_ind, double proj, const double delta[3], double d1_vec[3]) 
{
	double d1_current_orb[3] = {0.0, 0.0, 0.0}; 
	double d1_core_orb[3] = {0.0, 0.0, 0.0};
	d1_STOnG_s(r, pi, ao_ind, delta, d1_current_orb);
	d1_STOnG_s(r, pi, core_ao_ind, delta, d1_core_orb);

	for (int j = 0; j < 3; j++)
		d1_vec[j] = d1_current_orb[j] - proj * d1_core_orb[j]; 
}

// vector of laplacian UN-cusped NON-orthogonalized gaussian of s orbital for elctron i in orb p
void essqc::qmc::ABDeterminant::d2_STOnG_s(const double r, const double pi, int ao_ind, const double delta[3], double d2_vec[3])
{
	for (int k = 0; k < m_ng; k++)
	{
		double a = m_bf_exp[k * m_no + ao_ind];
		double d = m_bf_coeff[k * m_no + ao_ind];
		double Ns = pow(((2.0 * a) / pi), 0.75);
		d2_vec[0] += -2.0 * a * d * Ns * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[0] * delta[0]);
		d2_vec[1] += -2.0 * a * d * Ns * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[1] * delta[1]);
		d2_vec[2] += -2.0 * a * d * Ns * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[2] * delta[2]);
	} // contracted gaussians
}

// vector of laplacian element of the orthogonalized gaussian of s orbital for elctron i in orb p
void essqc::qmc::ABDeterminant::d2_STOnG_s(const double r, const double pi, int ao_ind, int core_ao_ind, double proj, const double delta[3], double d2_vec[3])
{
	double d2_current_orb[3] = {0.0, 0.0, 0.0};
	double d2_core_orb[3] = {0.0, 0.0, 0.0}; 
	d2_STOnG_s(r, pi, ao_ind, delta, d2_current_orb);
	d2_STOnG_s(r, pi, core_ao_ind, delta, d2_core_orb);

	for (int j = 0; j < 3; j++)
		d2_vec[j] = d2_current_orb[j] - proj * d2_core_orb[j]; 
}

// vector of NON-orthogonalized gaussian gradient of p orbital
void essqc::qmc::ABDeterminant::d1_STOnG_p(const double r, const double pi, int ao_ind, const double delta[3], double d1_vec[3])
{
	int orb_type = m_bf_orbs[ao_ind];

	if (orb_type == 2)
	{
		for (int k = 0; k < m_ng; k++)
		{
			double a = m_bf_exp[k * m_no + ao_ind];
			double d = m_bf_coeff[k * m_no + ao_ind];
			double Ns = pow(((2.0 * a) / pi), 0.75);
			double Np = Ns * 2.0 * std::sqrt(a);

			d1_vec[0] += d * Np * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[0] * delta[0]);
			d1_vec[1] += -2.0 * a * d * Np * delta[0] * delta[1] * std::exp(-a * r * r);
			d1_vec[2] += -2.0 * a * d * Np * delta[0] * delta[2] * std::exp(-a * r * r);
		} // contracted gaussians
	}     // px
	else if (orb_type == 3)
	{
		for (int k = 0; k < m_ng; k++)
		{
			double a = m_bf_exp[k * m_no + ao_ind];
			double d = m_bf_coeff[k * m_no + ao_ind];
			double Ns = pow(((2.0 * a) / pi), 0.75);
			double Np = Ns * 2.0 * std::sqrt(a);

			d1_vec[0] += -2.0 * a * d * Np * delta[1] * delta[0] * std::exp(-a * r * r);
			d1_vec[1] += d * Np * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[1] * delta[1]);
			d1_vec[2] += -2.0 * a * d * Np * delta[1] * delta[2] * std::exp(-a * r * r);
		} // contracted gaussians
	}     // py
	else if (orb_type == 4)
	{
		for (int k = 0; k < m_ng; k++)
		{
			double a = m_bf_exp[k * m_no + ao_ind];
			double d = m_bf_coeff[k * m_no + ao_ind];
			double Ns = pow(((2.0 * a) / pi), 0.75);
			double Np = Ns * 2.0 * std::sqrt(a);

			d1_vec[0] += -2.0 * a * d * Np * delta[2] * delta[0] * std::exp(-a * r * r);
			d1_vec[1] += -2.0 * a * d * Np * delta[2] * delta[1] * std::exp(-a * r * r);
			d1_vec[2] += d * Np * std::exp(-a * r * r) * (1.0 - 2.0 * a * delta[2] * delta[2]);
		} // contracted gaussians
	}     // pz
}

// vector of laplacian NON-orthogonalized gaussian of p orbital for ith electron in pth orbital
void essqc::qmc::ABDeterminant::d2_STOnG_p(const double r, const double pi, int ao_ind, const double delta[3], double d2_vec[3])
{
	int orb_type = m_bf_orbs[ao_ind];

	if (orb_type == 2)
	{
		for (int k = 0; k < m_ng; k++)
		{
			double a = m_bf_exp[k * m_no + ao_ind];
			double d = m_bf_coeff[k * m_no + ao_ind];
			double Ns = pow(((2.0 * a) / pi), 0.75);
			double Np = Ns * 2.0 * std::sqrt(a);

			d2_vec[0] += d * Np * std::exp(-a * r * r) * (-6.0 * a * delta[0] + 4.0 * a * a * delta[0] * delta[0] * delta[0]);
			d2_vec[1] += 2.0 * a * d * Np * delta[0] * std::exp(-a * r * r) * (2.0 * a * delta[1] * delta[1] - 1.0);
			d2_vec[2] += 2.0 * a * d * Np * delta[0] * std::exp(-a * r * r) * (2.0 * a * delta[2] * delta[2] - 1.0);
		} // contracted gaussians
	}     // px
	else if (orb_type == 3)
	{
		for (int k = 0; k < m_ng; k++)
		{
			double a = m_bf_exp[k * m_no + ao_ind];
			double d = m_bf_coeff[k * m_no + ao_ind];
			double Ns = pow(((2.0 * a) / pi), 0.75);
			double Np = Ns * 2.0 * std::sqrt(a);

			d2_vec[0] += 2.0 * a * d * Np * delta[1] * std::exp(-a * r * r) * (2.0 * a * delta[0] * delta[0] - 1.0);
			d2_vec[1] += d * Np * std::exp(-a * r * r) * (-6.0 * a * delta[1] + 4.0 * a * a * delta[1] * delta[1] * delta[1]);
			d2_vec[2] += 2.0 * a * d * Np * delta[1] * std::exp(-a * r * r) * (2.0 * a * delta[2] * delta[2] - 1.0);
		} // contracted gaussians
	}     // py
	else if (orb_type == 4)
	{
		for (int k = 0; k < m_ng; k++)
		{
			double a = m_bf_exp[k * m_no + ao_ind];
			double d = m_bf_coeff[k * m_no + ao_ind];
			double Ns = pow(((2.0 * a) / pi), 0.75);
			double Np = Ns * 2.0 * std::sqrt(a);

			d2_vec[0] += 2.0 * a * d * Np * delta[2] * std::exp(-a * r * r) * (2.0 * a * delta[0] * delta[0] - 1.0);
			d2_vec[1] += 2.0 * a * d * Np * delta[2] * std::exp(-a * r * r) * (2.0 * a * delta[1] * delta[1] - 1.0);
			d2_vec[2] += d * Np * std::exp(-a * r * r) * (-6.0 * a * delta[2] + 4.0 * a * a * delta[2] * delta[2] * delta[2]);
		} // contracted gaussians
	}     // pz
}

void essqc::qmc::ABDeterminant::d1_STOnG_d(const double dist, double (&delta)[3], const double pi, int ao_ind, int ao_type, double d1_vec[3]) {

  for (int k = 0; k < m_ng; k++)
  {
    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind]; 
    double N = d * pow(2048.0 * pow(a,7.0) / pow(pi, 3.0), 0.25);
    double x = delta[0];
    double y = delta[1];
    double z = delta[2];
    double r = dist;
    double exp = std::exp(-a * dist * dist);

    // 3dxy
    if (ao_type == 5) {
      d1_vec[0] += N * y * (1.0 - 2.0 * a * x * x ) * exp;   // check
      d1_vec[1] += N * x * (1.0 - 2.0 * a * y * y ) * exp;   // check
      d1_vec[2] += -2.0 * N * a * x * y * z * exp;	     // check
    }
    // 3dyz
    else if (ao_type == 6) {
      d1_vec[0] += -2.0 * N * a * x * y * z * exp; 	    // check
      d1_vec[1] += N * z * (1.0 - 2.0 * a * y * y ) * exp;  // check
      d1_vec[2] += N * y * (1.0 - 2.0 * a * z * z ) * exp;  // check
    }
    // 3dz^2
    else if (ao_type == 7) {
      double M = pow(144.0, -0.25);
      d1_vec[0] += 2.0 * M * N * x * (a * r * r - 3.0 * a * z * z - 1.0) * exp;			// check 
      d1_vec[1] += 2.0 * M * N * y * (a * r * r - 3.0 * a * z * z - 1.0) * exp;			// check
      d1_vec[2] += -2.0 * M * N * z * (2.0 * a * z * z - a * y * y - a * x * x - 2.0) * exp;    // check
    }
    // 3dxz
    else if (ao_type == 8) {
      d1_vec[0] += N * z * (1.0 - 2.0 * a * x * x ) * exp; // check
      d1_vec[1] += -2.0 * N * a * x * y * z * exp;	   // check
      d1_vec[2] += N * x * (1.0 - 2.0 * a * z * z ) * exp; // check
    }
    // 3dx^2-y^2
    else {
      double M = 0.5; 
      d1_vec[0] += -2.0 * M * N * x * (a * x * x - a * y * y - 1.0) * exp; // check
      d1_vec[1] += 2.0 * M * N * y * (a * y * y - a * x * x - 1.0) * exp;  // check
      d1_vec[2] += -2.0 * M * N * a * z * (x * x - y * y) * exp;           // check
    }
  } // contracted gaussians
}

void essqc::qmc::ABDeterminant::d2_STOnG_d(const double dist, double (&delta)[3], const double pi, int ao_ind, int ao_type, double d2_vec[3]) {

  for (int k = 0; k < m_ng; k++)
  {
    double a = m_bf_exp[k * m_no + ao_ind];
    double d = m_bf_coeff[k * m_no + ao_ind]; 
    double N = d * pow(2048.0 * pow(a,7.0) / pow(pi, 3.0), 0.25);
    double x = delta[0];
    double y = delta[1];
    double z = delta[2];
    double r = dist;
    double a2 = a * a;
    double x2 = x * x;
    double y2 = y * y;
    double z2 = z * z;
    double exp = std::exp(-a * dist * dist);

    // 3dxy
    if (ao_type == 5) {
      d2_vec[0] += 2.0 * N * a * x * y * (2.0 * a * x * x - 3.0) * exp;  // check
      d2_vec[1] += 2.0 * N * a * x * y * (2.0 * a * y * y - 3.0) * exp;  // check
      d2_vec[2] += 2.0 * N * a * x * y * (2.0 * a * z * z - 1.0) * exp;  // check
    }
    // 3dyz
    else if (ao_type == 6) {
      d2_vec[0] += 2.0 * N * a * y * z * (2.0 * a * x * x - 1.0) * exp; // check
      d2_vec[1] += 2.0 * N * a * y * z * (2.0 * a * y * y - 3.0) * exp; // check
      d2_vec[2] += 2.0 * N * a * y * z * (2.0 * a * z * z - 3.0) * exp; // check
    }
    // 3dz^2
    else if (ao_type == 7) {
      double M = pow(144.0, -0.25);
      d2_vec[0] += -2.0 * M * N * (2.0 * a * a * x * x * x * x + a * x * x * (2.0 * a * y * y - 4.0 * a * z * z - 5.0) - a * y * y + 2.0 * a * z * z + 1.0) * exp; // check
      d2_vec[1] += -2.0 * M * N * (2.0 * a * a * y * y * y * y + a * y * y * (2.0 * a * x * x - 4.0 * a * z * z - 5.0) - a * x * x + 2.0 * a * z * z + 1.0) * exp; // check
      d2_vec[2] += -2.0 * M * N * ((2.0 * a2 * z2 - a) * (x2 + y2) - 4.0 * a2 * z2 * z2 + 10.0 * a * z2 - 2.0) * exp; 						   // check 
    }
    // 3dxz
    else if (ao_type == 8) {
      d2_vec[0] += 2.0 * N * a * x * z * (2.0 * a * x * x - 3.0) * exp; // check
      d2_vec[1] += 2.0 * N * a * x * z * (2.0 * a * y * y - 1.0) * exp; // check
      d2_vec[2] += 2.0 * N * a * x * z * (2.0 * a * z * z - 3.0) * exp; // check
    }
    // 3dx^2-y^2
    else {
      double M = 0.5;
      d2_vec[0] += 2.0 * M * N * (2.0 * a * a * x * x * x * x - a * x * x * (2.0 * a * y * y + 5.0) + a * y * y + 1.0) * exp;   // check
      d2_vec[1] += 2.0 * M * N * (x2 * (2.0 * a2 * y2 - a) - 2.0 * a2 * y2 * y2 + 5.0 * a * y2 - 1.0) * exp; 				// check
      d2_vec[2] += 2.0 * M * N * a * (2.0 * a * z * z - 1.0) * (x * x - y * y) * exp;						// check
    }
  } // contracted gaussians
}

////// general orbital ///////

// factoring all the way out of the class to make unit testing simpler
void essqc::qmc::slater_orb_func_for_ABDeterminant(const PosVec &e_pos, const PosVec &n_pos, const double pi, std::vector<double> &xmat,
		const std::vector<int> & bf_orbs, const std::vector<double> & bf_exp,
		const std::vector<int> & bf_cen,
		const int no) {

	// get numbers of electrons and nuclei
	const int ne = e_pos.nparticles();
	const int nn = n_pos.nparticles();

	for (int p = 0; p < no; p++)
	{
		for (int i = 0; i < ne; i++)
		{
			const double r = PosVec::single_particle_distance(e_pos.get_pos(i), n_pos.get_pos(bf_cen[p]));
			const double rx = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(bf_cen[p]), 0);
			const double ry = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(bf_cen[p]), 1);
			const double rz = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(bf_cen[p]), 2);

			// 1s orbital
			if (bf_orbs[p] == 0)
			{
				const double z = bf_exp[p];
				double N = std::sqrt((z * z * z / pi));
				xmat[i + p * ne] = N * std::exp(-z * r);
			}
			// 2s orbital
			if (bf_orbs[p] == 1)
			{
				const double z = bf_exp[p];
				double N = std::sqrt((z * z * z * z * z / (3 * pi))); // STO
				xmat[i + p * ne] = N * r * std::exp(-z * r);
			}
			// 2px orbital
			if (bf_orbs[p] == 2)
			{
				const double z = bf_exp[p];
				double N = std::sqrt((z * z * z * z * z / pi)); // STO
				xmat[i + p * ne] = N * rx * std::exp(-z * r);
			}
			// 2py orbital
			if (bf_orbs[p] == 3)
			{
				const double z = bf_exp[p];
				double N = std::sqrt((z * z * z * z * z / pi)); // STO
				xmat[i + p * ne] = N * ry * std::exp(-z * r);
			}
			// 2pz orbital
			if (bf_orbs[p] == 4)
			{
				const double z = bf_exp[p];
				double N = std::sqrt((z * z * z * z * z / pi)); // STO
				xmat[i + p * ne] = N * rz * std::exp(-z * r);
			}
			// 3dz2 orbital
			if (bf_orbs[p] == 5)
			{
				const double z = bf_exp[p];
				double N = 1.0 / 3.0 * std::sqrt((z * z * z * z * z * z * z / (2 * pi))); // STO
				xmat[i + p * ne] = N * (3 * rz * rz - r * r) * std::exp(-z * r);
			}
			// 3dxz orbital
			if (bf_orbs[p] == 6)
			{
				const double z = bf_exp[p];
				double N = std::sqrt((2 * z * z * z * z * z * z * z / (3 * pi))); // STO
				xmat[i + p * ne] = N * rx * rz * std::exp(-z * r);
			}
			// 3dyz orbital
			if (bf_orbs[p] == 7)
			{
				const double z = bf_exp[p];
				double N = std::sqrt((2 * z * z * z * z * z * z * z / (3 * pi))); // STO
				xmat[i + p * ne] = N * ry * rz * std::exp(-z * r);
			}
			// 3dx^2-y^2 orbital
			if (bf_orbs[p] == 8)
			{
				const double z = bf_exp[p];
				double N = std::sqrt((z * z * z * z * z * z * z / (6 * pi))); // STO
				xmat[i + p * ne] = N * (rx * rx - ry * ry) * std::exp(-z * r);
			}
			// 3dxy orbital
			if (bf_orbs[p] == 9)
			{
				const double z = bf_exp[p];
				double N = std::sqrt((2 * z * z * z * z * z * z * z / (3 * pi))); // STO
				xmat[i + p * ne] = N * rx * ry * std::exp(-z * r);
			}
		}
	}

}

// fill xmat with slater orbitals - UN-cusped NON-orthogonal
void essqc::qmc::ABDeterminant::slater_orb(const PosVec &e_pos, const PosVec &n_pos, const double pi, std::vector<double> &xmat)
{
	slater_orb_func_for_ABDeterminant(e_pos, n_pos, pi, xmat, m_bf_orbs, m_bf_exp, m_bf_cen, m_no);
}

void essqc::qmc::ABDeterminant::get_full_X_mat(const PosVec &e_pos, const PosVec &n_pos, std::vector<double> &xmat)
{
	const int ne = e_pos.nparticles();
	const int nn = n_pos.nparticles();

	const double pi = 3.14159265359;

	// resize matrix if necessary
	if (xmat.size() != ne * m_no){
		xmat.assign(ne * m_no, 0.0);
		std::cout << " reassign xmat size: " << ne << "x" << m_no << std::endl;
	}

	if (m_use_STO)
	{
		slater_orb(e_pos, n_pos, pi, xmat);
	}
	else if (m_use_GTO)
	{
        ////////////////////////////////// 
        /*/////////////////////////////////
		To MolSSI: 

		This is where I stitched the evaluation of the cusping function to the central VMC sampling framework

		The X matrix elements contains the evaluation of an electron position in a gaussian-type orbital (GTO)
		  IF an electron falls within the cusp radius (defined in the python dictionary and imported here via the header file)
		  the a linear combination of the cusping function (Q - an expansion of Slaters) and the original GTO is calculated. 
		  The switching function (b(r) - a 5th order polynomial) controls the mixing of the cusp and GTO functions

		  ELSE IF and electron does not lay in a cusp region of any of the orbitals, the GTO is evaluated normally
		     note: the cusp procedure transofrms the GTO into a basis in whcih the n>1 s orbitals are orthogonalized against their 1s core
			       to ensure similar curvature of the Gaussians, so the cusp radii are transferable

		- TKQ
		*////////////////////////////////// 
		////////////////////////////////// 
		if (m_use_cusp) // cusped STOnG gaussian orbital basis
		{
			for (int p = 0; p < m_no; p++)  // eval all AO columns of Xmat
			{
				for (int i = 0; i < ne; i++)	// eval provided rows of xmat 
				{ 
					double proj = 0.0;
					int core_orb_ind = 0;

					double orb_total = 0.0;
					const double dist = PosVec::single_particle_distance(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p])); // btw elec an orbital center

					if (m_bf_orbs[p] == 0 || m_bf_orbs[p] == 1) 
					{
						if (m_orth_orb[p] == 0) // NO orth - 1s core
						{
							orb_total = STOnG_s(dist, pi, p);
						}
						else // orth - valence s orbital 
						{
							core_orb_ind = m_orth_orb[p] - 1;             // index of core orbital to orthogonalize against
							proj = m_proj_mat[core_orb_ind * m_no + p]; // get projection of orb onto core from python
							orb_total = STOnG_s(dist, pi, p, core_orb_ind, proj);
						}
					} // s orbital
					else if (m_bf_orbs[p] == 2 || m_bf_orbs[p] == 3 || m_bf_orbs[p] == 4)
					{                                                                                                                               // gaussian p orbital
						const double dist_xi = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p]), m_bf_orbs[p] - 2); // m_bf_orbs[p]-2 maps p-orbital type to x y or z (ie px=2 so we need dist_xi == 0)
						orb_total = STOnG_p(dist, dist_xi, pi, p);
					} // p orbital

	  				// d orbital 
	  				else {
          				  double dist_xyz[3];
          				  for (int j = 0; j < 3; j++) {
          				    dist_xyz[j] = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p]), j); 
	  				  }
          				  orb_total = STOnG_d(dist, dist_xyz, pi, p, m_bf_orbs[p]);
	  				}
					// eval cusp if in range of nuc
					double b_val = 0.0;
					double Q_fn = 0.0;
					double Pn_val = 0.0;
					int counter = 0;

					// loop thru nuc to correct cusp on given orb (p)
					for (int n = 0; n < nn; n++)
					{
						if (m_cusp_a0_mat[p * nn + n] != 0.0) 
						{
							const double nuc_dist = PosVec::single_particle_distance(e_pos.get_pos(i), n_pos.get_pos(n));

							double rc = m_cusp_radius_mat[p * nn + n];

							if (nuc_dist < rc) // cusp in nuc cusp_radius
							{
								// evaluate switching function
								b_val = b_func(m_cusp_radius_mat[p * nn + n], nuc_dist);
								double zeta = m_Z[n];

								// s orbital slater cusp
								Q_fn = slater_s_cusp(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist);

								// evaluate P functions
								Pn_val += (1.0 - b_val) * orb_total;

								int npbf = m_n_vec.size();
								for (int k = 0; k < npbf; k++) { // sub with length of the index array
									Pn_val += m_cusp_coeff_mat[n * npbf + p * nn * npbf + k] * Pn_func(b_val, Q_fn, orb_total, nuc_dist, m_n_vec[k], rc); // change k to index of appropriate vector and coeff
								}

								counter++;
							}   // if electron within cusp radius
						} 	// if orbital over given nuc has a cusp, a0 != 0.0
					}   // nuclei centers

					if (counter > 1)
						throw essqc::Exception("essqc::qmc::ABDeterminant::get_full_X_mat(), electron may be within only one nuclear center");

					if (m_get_slater_derivs) {   // zero out non-slater cusp contributions
						orb_total = 0.0;
						b_val = 1.0;
					}

					// if cusping the orbital
					if (counter == 1) {         
						xmat[i + p * ne] = Pn_val; 
					}
					// no cusp
					else {
						xmat[i + p * ne] = orb_total;
					}  

				}   // electrons
			}   // orbitals
		}   // with cusps
		else      // no cusps	PASSED
		{
			for (int p = 0; p < m_no; p++) {
				for (int i = 0; i < ne; i++) {
					double total = 0;
					const double dist = PosVec::single_particle_distance(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p]));

					if (m_bf_orbs[p] == 0 or m_bf_orbs[p] == 1) // gaussian s orbital
					{
						total = STOnG_s(dist, pi, p);
					} 
					else if (m_bf_orbs[p] == 2 || m_bf_orbs[p] == 3 || m_bf_orbs[p] == 4) // gaussian p orbital
					{
						const double dist_xi = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p]), m_bf_orbs[p] - 2); // m_bf_orbs[p]-2 maps p-orbital type to x y or z (ie px=2 so we need dist_xi == 0)
						total = STOnG_p(dist, dist_xi, pi, p);
					}

	  				// add d-orbitals here
	  				else {
          				  double dist_xyz[3];
          				  for (int j = 0; j < 3; j++) {
          				    dist_xyz[j] = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p]), j); 
	  				  }
          				  total = STOnG_d(dist, dist_xyz, pi, p, m_bf_orbs[p]);
	  				}

					xmat[i+p*ne] = total;

				}	// electrons
			}   // orbitals
		}	// no cusps

	}      // use_GTO
	else if (m_use_HAO)
	{
		// TODO fill in hydrogenic orbitals here
	} // use_HAO

}

void essqc::qmc::ABDeterminant::initialize_internal_data()
{
  //std::cout << __FILE__ << " in initialize_internal_data()" << std::endl;
	// Get X mat
	//std::cout << "     reinitialize internal data "; // << k << std::endl;
	this->get_full_X_mat(m_e1pos, m_nuc, m_x);

	double threshold = 1e-10;
	// replace elements with 0s if they're small enough
	for (int i = 0; i < m_na; i++)
	{
		for (int j = 0; j < m_no; j++)
		{
			if (abs(m_x[i + m_na * j]) < threshold)
			{
				m_x[i + m_na * j] = 0.0;
			}
		}
	}

	// Get XC ma
	essqc::dgemm('N', 'N', m_na, m_na, m_no, 1.0, &m_x.at(0), m_na, &m_C.at(0), m_no, 0.0, &m_xc.at(0), m_na);

	// Get XC inverse
	int lwork = m_na * m_na;
	std::vector<double> work(lwork, 0.0);
	std::vector<int> ipiv(m_na, 0);
	essqc::log_det_and_inverse(m_na, &m_xc.at(0), m_logdet, m_detsign, &m_xci.at(0), &ipiv.at(0), &work.at(0), lwork);
}

int essqc::qmc::ABDeterminant::print_spin()
{
	return m_e1s;
}

void essqc::qmc::ABDeterminant::copy_m_active(std::vector<int> &m_act_e)
{

	std::copy(m_active_e.begin(), m_active_e.end(), m_act_e.begin());
}

double essqc::qmc::ABDeterminant::get_new_to_old_ratio()
{

	if (m_e1s != m_mei.spin)
	{
		//std::cout << " ---> wrong spin for current det ---> " << std::endl;
		return 1.0;
	}
	else
	{
		essqc::qmc::ABDeterminant::get_full_X_mat(m_mei.pos, m_nuc, m_pxrow);

		// evaluate the new values for the corresponding row of the XC matrix
		essqc::dgemm('N', 'N', 1, m_na, m_no, 1.0, &m_pxrow.at(0), 1, &m_C.at(0), m_no, 0.0, &m_pxcrow.at(0), 1);

		// get the ratio of the new to old determinant value
		return essqc::ddot(m_na, &m_pxcrow.at(0), 1, &m_xci.at(m_na * m_mei.ind), 1);
	}
}

// accept and update
void essqc::qmc::ABDeterminant::accept_poposed_move()
{
	if (m_e1s == m_mei.spin)
	{

		// new X and XC rows calculated in the get_new_to_old_ratio function
		//std::cout << "IN accepted_proposed_move ";
		double ratio = this->get_new_to_old_ratio();
		// double ratio = essqc::ddot(m_na, &m_pxcrow.at(0), 1, &m_xci.at(m_na*m_mei.ind), 1);

		// update row of X matrix with accepted move
		essqc::dcopy(m_no, &m_pxrow.at(0), 1, &m_x.at(m_mei.ind), m_na);

		// update m_pxcrow so that it holds the difference between the new and old XC rows
		essqc::daxpy(m_na, -1.0, &m_xc.at(m_mei.ind), m_na, &m_pxcrow.at(0), 1);

		// update the row in the XC matrix
		essqc::daxpy(m_na, 1.0, &m_pxcrow.at(0), 1, &m_xc.at(m_mei.ind), m_na);

		// get the v^T (XC)^-1 vector   ( cost is O(m_na^2) )
		essqc::dgemm('N', 'N', 1, m_na, m_na, 1.0, &m_pxcrow.at(0), 1, &m_xci.at(0), m_na, 0.0, &m_vtxci.at(0), 1);

		// put the column of (XC)^-1 that we need in m_pxcrow
		essqc::dcopy(m_na, &m_xci.at(m_na * m_mei.ind), 1, &m_pxcrow.at(0), 1);

		// update (XC)^-1   ( cost is O(m_na^2) )
		essqc::dgemm('N', 'N', m_na, m_na, 1, -1.0 / ratio, &m_pxcrow.at(0), m_na, &m_vtxci.at(0), 1, 1.0, &m_xci.at(0), m_na);

		// update stored determinant value information
		m_logdet = m_logdet + std::log(std::fabs(ratio));
		if (std::signbit(ratio))
			m_detsign = -m_detsign;
	}
}

void essqc::qmc::ABDeterminant::get_orbs_and_derivs(const PosVec &e_pos, const PosVec &n_pos, double *orbs, double **der1, double **der2)
{
	// get numbers of electrons and nuclei
	const int ne = e_pos.nparticles();
	const int nn = n_pos.nparticles();
	const double pi = 3.14159265359;

	if (m_use_STO)
	{
		// fill the matrices with their values
		for (int p = 0; p < m_no; p++)
		{
			for (int i = 0; i < ne; i++)
			{

				// get pointers to the electron and the nucleus position
				const double *const eptr = e_pos.get_pos(i);
				const double *const nptr = n_pos.get_pos(m_bf_cen[p]);

				// get the vector between the electron and the nucleus
				double delta[3];
				delta[0] = eptr[0] - nptr[0];
				delta[1] = eptr[1] - nptr[1];
				delta[2] = eptr[2] - nptr[2];

				// get the distance distance between the electron and the nucleus
				const double r = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
				const double pi = 3.14159265359;

				// 1s orbital
				if (m_bf_orbs[p] == 0)
				{
					double z = m_bf_exp[p];
					const double N = std::sqrt(z * z * z / pi);

					// get the exponential part
					const double ex = std::exp(-z * r);

					// record the first derivatives
					const double rinv = 1.0 / r;
					const double ex_over_r = ex * rinv;
					der1[0][i + p * ne] = -N * z * delta[0] * ex_over_r;
					der1[1][i + p * ne] = -N * z * delta[1] * ex_over_r;
					der1[2][i + p * ne] = -N * z * delta[2] * ex_over_r;

					// record the second derivatives
					const double dd = rinv * rinv + rinv;
					der2[0][i + p * ne] = N * (delta[0] * delta[0] * z / r + delta[0] * delta[0] / (r * r) - 1.0) * z * ex_over_r;
					der2[1][i + p * ne] = N * (delta[1] * delta[1] * z / r + delta[1] * delta[1] / (r * r) - 1.0) * z * ex_over_r;
					der2[2][i + p * ne] = N * (delta[2] * delta[2] * z / r + delta[2] * delta[2] / (r * r) - 1.0) * z * ex_over_r;
				}
				// 2s orbital
				if (m_bf_orbs[p] == 1)
				{
					double z = m_bf_exp[p];

					// get the exponential part
					const double ex = std::exp(-z * r);

					// record the first derivatives
					const double rinv = 1.0 / r;
					const double ex_over_r = ex * rinv;
					const double N = std::sqrt(z * z * z * z * z / (3 * pi));
					der1[0][i + p * ne] = N * (ex_over_r * delta[0] - z * delta[0] * ex); // STO
					der1[1][i + p * ne] = N * (ex_over_r * delta[1] - z * delta[1] * ex);
					der1[2][i + p * ne] = N * (ex_over_r * delta[2] - z * delta[2] * ex);

					// record the second derivatives
					const double dd = rinv * rinv + rinv;
					der2[0][i + p * ne] = N * (z * z * delta[0] * delta[0] * ex / r - z * delta[0] * delta[0] * ex / (r * r) - z * ex + ex * (1 / r - delta[0] * delta[0] / (r * r * r)));
					der2[1][i + p * ne] = N * (z * z * delta[1] * delta[1] * ex / r - z * delta[1] * delta[1] * ex / (r * r) - z * ex + ex * (1 / r - delta[1] * delta[1] / (r * r * r)));
					der2[2][i + p * ne] = N * (z * z * delta[2] * delta[2] * ex / r - z * delta[2] * delta[2] * ex / (r * r) - z * ex + ex * (1 / r - delta[2] * delta[2] / (r * r * r)));
				}
				// 2px orbital
				if (m_bf_orbs[p] == 2)
				{
					double z = m_bf_exp[p];

					// get the exponential part
					const double ex = std::exp(-z * r);

					// record the first derivatives
					const double rinv = 1.0 / r;
					const double ex_over_r = ex * rinv;
					const double N = std::sqrt(z * z * z * z * z / pi);
					der1[0][i + p * ne] = N * (ex - z * delta[0] * delta[0] * ex / r);
					der1[1][i + p * ne] = N * (-z * delta[1] * delta[0] * ex / r); // STO
					der1[2][i + p * ne] = N * (-z * delta[2] * delta[0] * ex / r);

					// record the second derivatives
					const double dd = rinv * rinv + rinv;
					der2[0][i + p * ne] = N * (delta[0] * ex * (z * z * delta[0] * delta[0] / (r * r) + z * delta[0] * delta[0] / (r * r * r) - 3 * z / r));
					der2[1][i + p * ne] = N * (delta[0] * z * ex * (z * delta[1] * delta[1] / (r * r) + delta[1] * delta[1] / (r * r * r) - 1 / r));
					der2[2][i + p * ne] = N * (delta[0] * z * ex * (z * delta[2] * delta[2] / (r * r) + delta[2] * delta[2] / (r * r * r) - 1 / r));
				}
				// 2py orbital
				if (m_bf_orbs[p] == 3)
				{
					double z = m_bf_exp[p];

					// get the exponential part
					const double ex = std::exp(-z * r);

					// record the first derivatives
					const double rinv = 1.0 / r;
					const double ex_over_r = ex * rinv;
					const double N = std::sqrt(z * z * z * z * z / pi);
					der1[0][i + p * ne] = N * (-z * delta[0] * delta[1] * ex / r);
					der1[1][i + p * ne] = N * (ex - z * delta[1] * delta[1] * ex / r); // STO
					der1[2][i + p * ne] = N * (-z * delta[2] * delta[1] * ex / r);

					// record the second derivatives
					const double dd = rinv * rinv + rinv;
					der2[0][i + p * ne] = N * (delta[1] * z * ex * (z * delta[0] * delta[0] / (r * r) + delta[0] * delta[0] / (r * r * r) - 1 / r));
					der2[1][i + p * ne] = N * (delta[1] * ex * (z * z * delta[1] * delta[1] / (r * r) + z * delta[1] * delta[1] / (r * r * r) - 3 * z / r));
					der2[2][i + p * ne] = N * (delta[1] * z * ex * (z * delta[2] * delta[2] / (r * r) + delta[2] * delta[2] / (r * r * r) - 1 / r));
				}
				// 2pz orbital
				if (m_bf_orbs[p] == 4)
				{
					double z = m_bf_exp[p];

					// get the exponential part
					const double ex = std::exp(-z * r);

					// record the first derivatives
					const double rinv = 1.0 / r;
					const double ex_over_r = ex * rinv;
					const double N = std::sqrt(z * z * z * z * z / pi);
					der1[0][i + p * ne] = N * (-z * delta[0] * delta[2] * ex / r);
					der1[1][i + p * ne] = N * (-z * delta[1] * delta[2] * ex / r);
					der1[2][i + p * ne] = N * (ex - z * delta[2] * delta[2] * ex / r);

					// record the second derivatives
					const double dd = rinv * rinv + rinv;
					der2[0][i + p * ne] = N * (delta[2] * z * ex * (z * delta[0] * delta[0] / (r * r) + delta[0] * delta[0] / (r * r * r) - 1 / r));
					der2[1][i + p * ne] = N * (delta[2] * z * ex * (z * delta[1] * delta[1] / (r * r) + delta[1] * delta[1] / (r * r * r) - 1 / r));
					der2[2][i + p * ne] = N * (delta[2] * ex * (z * z * delta[2] * delta[2] / (r * r) + z * delta[2] * delta[2] / (r * r * r) - 3 * z / r));
				}
				// 3dz2 orbital
				if (m_bf_orbs[p] == 5)
				{
					double z = m_bf_exp[p];

					// get the exponential part
					const double ex = std::exp(-z * r);

					// record the first derivatives
					const double rinv = 1.0 / r;
					const double ex_over_r = ex * rinv;
					const double N = 1.0 / 3.0 * std::sqrt((z * z * z * z * z * z * z / (2 * pi))); // STO
					der1[0][i + p * ne] = N * (-2.0 * ex * delta[0] - z * ex * delta[0] * (-delta[0] * delta[0] - delta[1] * delta[1] + 2 * delta[2] * delta[2]) / r);
					der1[1][i + p * ne] = N * (-2.0 * ex * delta[1] - z * ex * delta[1] * (-delta[0] * delta[0] - delta[1] * delta[1] + 2 * delta[2] * delta[2]) / r);
					der1[2][i + p * ne] = N * (4 * ex * delta[2] - z * delta[2] * ex * (-delta[0] * delta[0] - delta[1] * delta[1] + 2 * delta[2] * delta[2]) / r);

					// record the second derivatives
					const double dd = rinv * rinv + rinv;
					der2[0][i + p * ne] = N * (-2.0 * ex + 4.0 * z * ex * delta[0] * delta[0] / r + (-delta[0] * delta[0] - delta[1] * delta[1] + 2.0 * delta[2] * delta[2]) * (z * ex * delta[0] * delta[0] / (r * r * r) + z * z * ex * delta[0] * delta[0] / (r * r) - z * ex / r));
					der2[1][i + p * ne] = N * (-2.0 * ex + 4.0 * z * ex * delta[1] * delta[1] / r + (-delta[0] * delta[0] - delta[1] * delta[1] + 2.0 * delta[2] * delta[2]) * (z * ex * delta[1] * delta[1] / (r * r * r) + z * z * ex * delta[1] * delta[1] / (r * r) - z * ex / r));
					der2[2][i + p * ne] = N * (4.0 * ex - 8.0 * z * ex * delta[2] * delta[2] / r + (-delta[0] * delta[0] - delta[1] * delta[1] + 2.0 * delta[2] * delta[2]) * (z * ex * delta[2] * delta[2] / (r * r * r) + z * z * ex * delta[2] * delta[2] / (r * r) - z * ex / r));
				}
				// 3dxz orbital
				if (m_bf_orbs[p] == 6)
				{
					double z = m_bf_exp[p];

					// get the exponential part
					const double ex = std::exp(-z * r);

					// record the first derivatives
					const double rinv = 1.0 / r;
					const double ex_over_r = ex * rinv;
					const double N = std::sqrt((2.0 * z * z * z * z * z * z * z / (3 * pi))); // STO
					der1[0][i + p * ne] = N * (ex * delta[2] - z * ex * delta[0] * delta[0] * delta[2] / r);
					der1[1][i + p * ne] = N * (-z * ex * delta[0] * delta[1] * delta[2] / r);
					der1[2][i + p * ne] = N * (ex * delta[0] - z * ex * delta[2] * delta[2] * delta[0] / r);

					// record the second derivatives
					const double dd = rinv * rinv + rinv;
					der2[0][i + p * ne] = N * delta[2] * (-2.0 * z * ex * delta[0] / r + delta[0] * (z * ex * delta[0] * delta[0] / (r * r * r) + z * z * ex * delta[0] * delta[0] / (r * r) - z * ex / r));
					der2[1][i + p * ne] = N * delta[0] * delta[2] * (z * ex * delta[1] * delta[1] / (r * r * r) + z * z * ex * delta[1] * delta[1] / (r * r) - z * ex / r);
					der2[2][i + p * ne] = N * delta[0] * (-2.0 * z * ex * delta[2] / r + delta[2] * (z * ex * delta[2] * delta[2] / (r * r * r) + z * z * ex * delta[2] * delta[2] / (r * r) - z * ex / r));
				}
				// 3dyz orbital
				if (m_bf_orbs[p] == 7)
				{
					double z = m_bf_exp[p];

					// get the exponential part
					const double ex = std::exp(-z * r);

					// record the first derivatives
					const double rinv = 1.0 / r;
					const double ex_over_r = ex * rinv;
					const double N = std::sqrt((2.0 * z * z * z * z * z * z * z / (3 * pi))); // STO
					der1[1][i + p * ne] = N * (ex * delta[2] - z * ex * delta[1] * delta[1] * delta[2] / r);
					der1[0][i + p * ne] = N * (-z * ex * delta[1] * delta[0] * delta[2] / r);
					der1[2][i + p * ne] = N * (ex * delta[1] - z * ex * delta[2] * delta[2] * delta[1] / r);

					// record the second derivatives
					const double dd = rinv * rinv + rinv;
					der2[1][i + p * ne] = N * delta[2] * (-2.0 * z * ex * delta[1] / r + delta[1] * (z * ex * delta[1] * delta[1] / (r * r * r) + z * z * ex * delta[1] * delta[1] / (r * r) - z * ex / r));
					der2[0][i + p * ne] = N * delta[1] * delta[2] * (z * ex * delta[0] * delta[0] / (r * r * r) + z * z * ex * delta[0] * delta[0] / (r * r) - z * ex / r);
					der2[2][i + p * ne] = N * delta[1] * (-2.0 * z * ex * delta[2] / r + delta[2] * (z * ex * delta[2] * delta[2] / (r * r * r) + z * z * ex * delta[2] * delta[2] / (r * r) - z * ex / r));
				}
				// 3dx2-y2 orbital
				if (m_bf_orbs[p] == 8)
				{
					double z = m_bf_exp[p];

					// get the exponential part
					const double ex = std::exp(-z * r);

					// record the first derivatives
					const double rinv = 1.0 / r;
					const double ex_over_r = ex * rinv;
					const double N = std::sqrt((z * z * z * z * z * z * z / (6 * pi))); // STO
					der1[0][i + p * ne] = N * (2.0 * ex * delta[0] - z * ex * delta[0] * (delta[0] * delta[0] - delta[1] * delta[1]) / r);
					der1[1][i + p * ne] = N * (-2.0 * ex * delta[1] - z * ex * delta[1] * (delta[0] * delta[0] - delta[1] * delta[1]) / r);
					der1[2][i + p * ne] = N * (-z * ex * delta[2] * (delta[0] * delta[0] - delta[1] * delta[1]) / r);

					// record the second derivatives
					const double dd = rinv * rinv + rinv;
					der2[0][i + p * ne] = N * (2.0 * ex - 4.0 * z * ex * delta[0] * delta[0] / r + (delta[0] * delta[0] - delta[1] * delta[1]) * (z * ex * delta[0] * delta[0] / (r * r * r) + z * z * ex * delta[0] * delta[0] / (r * r) - z * ex / r));
					der2[1][i + p * ne] = N * (-2.0 * ex + 4.0 * z * ex * delta[1] * delta[1] / r + (delta[0] * delta[0] - delta[1] * delta[1]) * (z * ex * delta[1] * delta[1] / (r * r * r) + z * z * ex * delta[1] * delta[1] / (r * r) - z * ex / r));
					der2[2][i + p * ne] = N * (delta[0] * delta[0] - delta[1] * delta[1]) * (z * ex * delta[2] * delta[2] / (r * r * r) + z * z * ex * delta[2] * delta[2] / (r * r) - z * ex / r);
				}
				// 3dxy orbital
				if (m_bf_orbs[p] == 9)
				{
					double z = m_bf_exp[p];

					// get the exponential part
					const double ex = std::exp(-z * r);

					// record the first derivatives
					const double rinv = 1.0 / r;
					const double ex_over_r = ex * rinv;
					const double N = std::sqrt((2.0 * z * z * z * z * z * z * z / (3 * pi))); // STO
					der1[0][i + p * ne] = N * (ex * delta[1] - z * ex * delta[0] * delta[0] * delta[1] / r);
					der1[2][i + p * ne] = N * (-z * ex * delta[0] * delta[1] * delta[2] / r);
					der1[1][i + p * ne] = N * (ex * delta[0] - z * ex * delta[1] * delta[1] * delta[0] / r);

					// record the second derivatives
					const double dd = rinv * rinv + rinv;
					der2[0][i + p * ne] = N * delta[1] * (-2.0 * z * ex * delta[0] / r + delta[0] * (z * ex * delta[0] * delta[0] / (r * r * r) + z * z * ex * delta[0] * delta[0] / (r * r) - z * ex / r));
					der2[2][i + p * ne] = N * delta[0] * delta[1] * (z * ex * delta[2] * delta[2] / (r * r * r) + z * z * ex * delta[2] * delta[2] / (r * r) - z * ex / r);
					der2[1][i + p * ne] = N * delta[0] * (-2.0 * z * ex * delta[1] / r + delta[1] * (z * ex * delta[1] * delta[1] / (r * r * r) + z * z * ex * delta[1] * delta[1] / (r * r) - z * ex / r));
				}
			}
		}
	} // use_STO
	// gaussian orbital basis
	else if (m_use_GTO)
	{
        ////////////////////////////////// 
        /*/////////////////////////////////
		To MolSSI: 

		This is where I stitched the evaluation of the cusping function when calculating the derivatives needed to compute the kinetic energy
		in addition to the LM matrix elements 
        
		- TKQ
		*////////////////////////////////// 
		////////////////////////////////// 
		if (m_use_cusp) // cusped STOnG gaussian orbital basis
		{
			// fill the matrices with their values
			for (int p = 0; p < m_no; p++)
			{
				for (int i = 0; i < ne; i++)
				{

					// get pointers to the electron and the nucleus position
					const double *const eptr = e_pos.get_pos(i);
					const double *const nptr = n_pos.get_pos(m_bf_cen[p]);

					// get the vector between the electron and the nucleus
					double delta[3];
					delta[0] = eptr[0] - nptr[0];
					delta[1] = eptr[1] - nptr[1];
					delta[2] = eptr[2] - nptr[2];

					const double dist = PosVec::single_particle_distance(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p]));
					double orb_total = 0.0;
					double der1_total[3] = {0.0, 0.0, 0.0};
					double der2_total[3] = {0.0, 0.0, 0.0};

					// s orbital
					if (m_bf_orbs[p] == 0 || m_bf_orbs[p] == 1)
					{
						if (m_orth_orb[p] == 0) // NO orthogonalization - 1s core
						{
							orb_total = STOnG_s(dist, pi, p);
							d1_STOnG_s(dist, pi, p, delta, der1_total);
							d2_STOnG_s(dist, pi, p, delta, der2_total);

						}
						else // orthogonalized s orbital --- valence
						{
							int core_orb_ind = m_orth_orb[p] - 1;             // index of core orbital to orthogonalize against
							double proj = m_proj_mat[core_orb_ind * m_no + p]; // get projection of orb onto core from python

							orb_total = STOnG_s(dist, pi, p, core_orb_ind, proj);
							d1_STOnG_s(dist, pi, p, core_orb_ind, proj, delta, der1_total);
							d2_STOnG_s(dist, pi, p, core_orb_ind, proj, delta, der2_total);

						}
					} // s orbital
					// p orbital
					else if (m_bf_orbs[p] == 2 || m_bf_orbs[p] == 3 || m_bf_orbs[p] == 4)
					{                                                                                                                               // gaussian p orbital 
						const double dist_xi = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p]), m_bf_orbs[p] - 2); // m_bf_orbs[p]-2 maps p-orbital type to x y or z (ie px=2 so we need dist_xi == 0)
						orb_total = STOnG_p(dist, dist_xi, pi, p);

						d1_STOnG_p(dist, pi, p, delta, der1_total);
						d2_STOnG_p(dist, pi, p, delta, der2_total);

					} // p-orbital

          				// d-orbitals
	  				else {
          				  orb_total = STOnG_d(dist, delta, pi, p, m_bf_orbs[p]);
          				  d1_STOnG_d(dist, delta, pi, p, m_bf_orbs[p], der1_total);
          				  d2_STOnG_d(dist, delta, pi, p, m_bf_orbs[p], der2_total);
	  				}
					double b_val = 0.0;
					double d1_b_vec[3] = {0.0, 0.0, 0.0};
					double d2_b_vec[3] = {0.0, 0.0, 0.0};

					double Q_fn = 0.0;
					double d1_Q_vec[3] = {0.0, 0.0, 0.0};
					double d2_Q_vec[3] = {0.0, 0.0, 0.0};

					double d1_Pn_vec[3] = {0.0, 0.0, 0.0};
					double d2_Pn_vec[3] = {0.0, 0.0, 0.0};

					int counter = 0;

					// loop through nuclei to correct the cusps over each nuceli on the given orb (p)
					for (int n = 0; n < nn; n++)
					{
						if (m_cusp_a0_mat[p * nn + n] != 0.0) 
						{
							//std::cout << "non-zero cusp_a0 matrix element, check e- position for cusp in get_orbs_and_derivs()";
							double nuc_dist = PosVec::single_particle_distance(e_pos.get_pos(i), n_pos.get_pos(n));
							double diff[3] = {0.0, 0.0, 0.0};

							double rc = m_cusp_radius_mat[p * nn + n];

							// if a cusp is needed because e- near nuc and a0 > 0.0
							if (nuc_dist < m_cusp_radius_mat[p * nn + n])	// this can only be true for 1 electron-nuclear pair
							{
								counter++;
								for (int l = 0; l < 3; l++) {
									diff[l] = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(n), l); // (x_i - X_nuc)
								}

								// evaluate switching function
								b_val = b_func(m_cusp_radius_mat[p * nn + n], nuc_dist);
								d1_b_func(m_cusp_radius_mat[p * nn + n], nuc_dist, i, n, diff, d1_b_vec);
								d2_b_func(m_cusp_radius_mat[p * nn + n], nuc_dist, i, n, diff, d2_b_vec);

								double zeta = m_Z[n];
								// FOR FUTURE WORK: change cusp condition - if atom centered p-orbital - if we start cusping p-orbs
								//if (m_bf_orbs[p] > 1 && n == m_bf_cen[p])
								//    zeta = m_Z[n] / 2.0;

								// s orbital slater cusp
								Q_fn = slater_s_cusp(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist);
								d1_slater_s_cusp(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist, i, n, diff, d1_Q_vec);
								d2_slater_s_cusp(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist, i, n, diff, d2_Q_vec);

								int npbf = m_n_vec.size();

								d1_one_minus_b(orb_total, b_val, der1_total, d1_b_vec, d1_Pn_vec); 
								d2_one_minus_b(orb_total, b_val, der1_total, d1_b_vec, der2_total, d2_b_vec, d2_Pn_vec); 

								for (int k = 0; k < m_n_vec.size(); k++) { // change the k !!!  
									d1_Pn_func(orb_total, b_val, Q_fn, der1_total, d1_b_vec, d1_Q_vec, nuc_dist, diff, m_n_vec[k], m_cusp_coeff_mat[n * npbf + p * nn * npbf + k], d1_Pn_vec, rc);
									d2_Pn_func(orb_total, b_val, Q_fn, der1_total, d1_b_vec, d1_Q_vec, der2_total, d2_b_vec, d2_Q_vec, nuc_dist, diff, m_n_vec[k], m_cusp_coeff_mat[n * npbf + p * nn * npbf + k], d2_Pn_vec, rc);
								}

							}   // if electron in cusping region
						}       // if orbital over current nucleus is cusped, a0 != 0.0
					}			// nuclear centers

					// for cusp test, save just the slater KE vals if get_slater_derivs == true
					if (m_get_slater_derivs) {   // zero out non-slater cusp contributions
						orb_total = 0.0;
						der1_total[0] = 0.0;
						der1_total[1] = 0.0;
						der1_total[2] = 0.0;

						der2_total[0] = 0.0;
						der2_total[1] = 0.0;
						der2_total[2] = 0.0;

						b_val = 1.0;
						d1_b_vec[0] = 0.0;
						d1_b_vec[1] = 0.0;
						d1_b_vec[2] = 0.0;

						d2_b_vec[0] = 0.0;
						d2_b_vec[1] = 0.0;
						d2_b_vec[2] = 0.0;
					}            

					// if using cusped orbs 
					if (counter == 1) {
						der1[0][i + p * ne] = d1_Pn_vec[0];
						der1[1][i + p * ne] = d1_Pn_vec[1];
						der1[2][i + p * ne] = d1_Pn_vec[2];

						der2[0][i + p * ne] = d2_Pn_vec[0];
						der2[1][i + p * ne] = d2_Pn_vec[1];
						der2[2][i + p * ne] = d2_Pn_vec[2];
					}
					// if using no cusps for this AO
					else { 
						der1[0][i + p * ne] = der1_total[0];
						der1[1][i + p * ne] = der1_total[1];
						der1[2][i + p * ne] = der1_total[2];

						der2[0][i + p * ne] = der2_total[0];
						der2[1][i + p * ne] = der2_total[1];
						der2[2][i + p * ne] = der2_total[2];

					}

					if (counter > 1)
						std::cout << "counter at: " << counter << std::endl;

				} // electrons
			}     // orbitals
		}
		else // no cusps
		{
			// fill the matrices with their values
			for (int p = 0; p < m_no; p++) {
				for (int i = 0; i < ne; i++) {

					// get pointers to the electron and the nucleus position
					const double * const eptr = e_pos.get_pos(i);
					const double * const nptr = n_pos.get_pos(m_bf_cen[p]);

					// get the vector between the electron and the nucleus
					double delta[3];
					delta[0] = eptr[0] - nptr[0];
					delta[1] = eptr[1] - nptr[1];
					delta[2] = eptr[2] - nptr[2];

					// get the distance distance between the electron and the nucleus
					const double r = PosVec::single_particle_distance(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p]));

					double der1_total[3] = {0.0, 0.0, 0.0};
					double der2_total[3] = {0.0, 0.0, 0.0};

					// s-orbital
					if (m_bf_orbs[p] == 0 || m_bf_orbs[p] == 1)
					{
						d1_STOnG_s(r, pi, p, delta, der1_total);
						d2_STOnG_s(r, pi, p, delta, der2_total);
					} // p-orbital
					else if (m_bf_orbs[p] == 2 || m_bf_orbs[p] == 3 || m_bf_orbs[p] == 4)
					{
						d1_STOnG_p(r, pi, p, delta, der1_total);
						d2_STOnG_p(r, pi, p, delta, der2_total);
					}
					// d-orbital
	  				else {
          				  //orb_total = STOnG_d(dist, delta, pi, p, m_bf_orbs[p]);
          				  d1_STOnG_d(r, delta, pi, p, m_bf_orbs[p], der1_total);
          				  d2_STOnG_d(r, delta, pi, p, m_bf_orbs[p], der2_total);
	  				}

					der1[0][i+p*ne] = der1_total[0];
					der1[1][i+p*ne] = der1_total[1];
					der1[2][i+p*ne] = der1_total[2];
					der2[0][i+p*ne] = der2_total[0];
					der2[1][i+p*ne] = der2_total[1];
					der2[2][i+p*ne] = der2_total[2];

					//std::cout << "\t der1_total ";
					//for (auto element : der1_total)
					//	std::cout << element << " ";
					//std::cout << "\t der2_total ";
					//for (auto element : der2_total)
					//	std::cout << element << " ";
				}	// electrons
			}		// orbitals
		}			// no cusps
		/*
		   std::cout << "m_xc in orbs_and_derivs should be equiv for w and w/out cusps"<< std::endl;
		   for (int m = 0; m < m_na; m++){
		   for (int n = 0; n < m_na; n++){
		   std::cout << m_xc[m+n*m_na] << " ";
		   }
		   std::cout << std::endl;
		   }
		   std::cout << std::endl;
		   */
	}				// use_GTO
	else if (m_use_HAO)
	{
		// TODO fill in hydrogenic orbitals here
	} // use_HAO
}

void essqc::qmc::ABDeterminant::compute_psi(double &psi)
{   
	// calculate wfn here (not using for ratio in the local_e_psi_squared accumulator)
	std::vector<int> ipiv(m_na, 0);

	// copy of current XC (and inverse) matrix to calculate the determinant 
	std::vector<double> xc_copy;
	std::vector<double> xci_copy;

	xc_copy.assign(m_na*m_na, 0.0);
	xci_copy.assign(m_na*m_na, 0.0);

	essqc::dcopy(m_na*m_na, &m_xc.at(0), 1, &xc_copy.at(0), 1);
	essqc::dcopy(m_na*m_na, &m_xci.at(0), 1, &xci_copy.at(0), 1);

	// is it necessary to use the copied matrices?
	double det = essqc::get_det(m_na, &xc_copy.at(0), &xci_copy.at(0), &ipiv.at(0));

	psi *= det; 

}

void essqc::qmc::ABDeterminant::compute_psi_piece_for_ratio(double &psi)
{   
	// calculate sq of wfn here (not using for ratio in the local_e_psi_squared accumulator)

	// do I need ipiv at all?
	std::vector<int> ipiv(m_na, 0);

	// copy of current XC (and inverse) matrix to calculate the determinant 
	std::vector<double> xc_copy;
	std::vector<double> xci_copy;

	xc_copy.assign(m_na*m_na, 0.0);
	xci_copy.assign(m_na*m_na, 0.0);

	essqc::dcopy(m_na*m_na, &m_xc.at(0), 1, &xc_copy.at(0), 1);
	essqc::dcopy(m_na*m_na, &m_xci.at(0), 1, &xci_copy.at(0), 1);

	// is it necessary to use the copied matrices?
	double det = essqc::get_det(m_na, &xc_copy.at(0), &xci_copy.at(0), &ipiv.at(0));

	psi *= (det * det); 

}

void essqc::qmc::ABDeterminant::compute_ke_pieces(double *const d1logx, double *const d2logx)
{

	// get room to store the first and second derivatives
	const int space_needed = 7 * m_na * m_no + m_na * m_na;
	if (m_workspace.size() < space_needed)
		m_workspace.assign(space_needed, 0.0);

	// get pointers to where we will put orbital values and derivatives
	double * der1[3];
	double * der2[3];
	double * orbs = &m_workspace.at(0 * m_na * m_no);
	der1[0] = &m_workspace.at(1 * m_na * m_no); // x 1st derivatives of the atomic orbitals
	der1[1] = &m_workspace.at(2 * m_na * m_no); // y 1st derivatives of the atomic orbitals
	der1[2] = &m_workspace.at(3 * m_na * m_no); // z 1st derivatives of the atomic orbitals
	der2[0] = &m_workspace.at(4 * m_na * m_no); // x 2nd derivatives of the atomic orbitals
	der2[1] = &m_workspace.at(5 * m_na * m_no); // y 2nd derivatives of the atomic orbitals
	der2[2] = &m_workspace.at(6 * m_na * m_no); // z 2nd derivatives of the atomic orbitals

	// get pointers to where we will put products of matrices
	double *const proda = &m_workspace.at(7 * m_na * m_no + 0 * m_na * m_na);

	// get orbital values and derivatives
	essqc::qmc::ABDeterminant::get_orbs_and_derivs(m_e1pos, m_nuc, orbs, der1, der2);

	// loop over x, y, and z
	for (int mu = 0; mu < 3; mu++)
	{
		// get contributions involving the term   Tr[ (XC)^-1 dX/dri C ]
		// proda = dX/dri C
		essqc::dgemm('N', 'N', m_na, m_na, m_no, 1.0, der1[mu], m_na, &m_C.at(0), m_no, 0.0, proda, m_na);
		for (int i = 0; i < m_na; i++)
		{
			// (dX/dri C)_ithrow . XCi_ithcol
			const double val = essqc::ddot(m_na, proda + i, m_na, &m_xci.at(i * m_na), 1);

			if (m_e1s == 0)
			{
				//std::cout << "alpha" << std::endl;
				d1logx[3 * i + mu] += val;
				d2logx[i] -= val * val;
			}
			if (m_e1s == 1)
			{
				//std::cout << "beta" << std::endl;
				d1logx[3 * m_na + 3 * i + mu] += val;
				d2logx[m_na + i] -= val * val;
			}
		}

		// get contributions involving the term   Tr[ (XC)^-1 d2X/dri2 C ]
		essqc::dgemm('N', 'N', m_na, m_na, m_no, 1.0, der2[mu], m_na, &m_C.at(0), m_no, 0.0, proda, m_na);
		for (int i = 0; i < m_na; i++)
		{
			if (m_e1s == 0)
			{
				//std::cout << "alpha" << std::endl;
				d2logx[i] += essqc::ddot(m_na, proda + i, m_na, &m_xci.at(i * m_na), 1);
			}
			if (m_e1s == 1)
			{
				//std::cout << "beta" << std::endl;
				d2logx[m_na + i] += essqc::ddot(m_na, proda + i, m_na, &m_xci.at(i * m_na), 1);
			}
		}

	}

}


// simple matrix printing function (column major)
void essqc::qmc::ABDeterminant::matprint(const int n, const int m, const double * mat, const std::string & fmt) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      std::cout << boost::format(" " + fmt) % mat[i+n*j];
    }
    std::cout << std::endl;
  }
}

// d/dc_ij Psi / Psi = ([(XC)^-1 X])^T_ij	(alpha + beta) - CHECKED AND GOOD (matches finite diff test)
void essqc::qmc::ABDeterminant::compute_PvP0(double *const PvP0, double &ratio, int indx)
{
	// X^T * [(XC)^-1]^T ---> PvP0, dgemm dim inputs checks
	essqc::dgemm('T', 'T', m_no, m_na, m_na, 1.0, &m_x.at(0), m_na, &m_xci.at(0), m_na, 1.0, PvP0, m_no);
	
}

// MolSSI
// Computing the terms to accumulate for the linear method matrices at each sample 
//   This invloved analytically calulating the derivatives of the new cusping function, as well as the GTOs to calculate and 
//   store the matrix elements necessary to build the orthogonalized LM matrices
//   
// This is far from a fully optimized code as one of my research projects is more concerned with providing the proof of principle
//   of an orbital optimization scheme that removes AOs that should not contribute much to an MO, on the fly. This slicing and dicing of the LM matrices 
//   is all done on the python side. FOr simplicity I did not go into describing the inner workings of the sLCAO algorithm
//	 in the linear method code example, but those functions are built to be flexible enough to change what is considered optimizable in the full orbital basis
//   at each LM iteration

// d/dc_ij (H Psi / Psi) = -1/2 d/dc_ij lap(psi) - (grad(psi)) . (d/dc_ij grad(psi))
//  (lap term)           = -1/2 lap_X^T [(XC)^-1]^T + 1/2 X^T [C (XC)^-1]^T lap_X^T [(XC)^-1]^T + grad_X^T K^T + X^T [C (XC)^-1]^T grad_X^T [K^-1]^T 
//  (grad.grad term)       - grad_X^T(scaled by gradlogpsi) [(XC)^-1]^T + X^T [(XC)^-1]^T C^T grad_X^T(scaled by gradlogpsi) [(XC)^-1]^T }_ij	
void essqc::qmc::ABDeterminant::compute_grad_E(double *const grad_E, double *const grad_log_psi, int indx)
{
	std::vector<double> grad_E_temp(get_num_of_vp(), 0.0);

	// TODO make der1 an input calculated in compute_ke_pieces
	const int lm_space_needed = 11 * m_na * m_no + 6 * m_na * m_na * m_no;
	std::vector<double> m_workspace_2(lm_space_needed, 0.0);	
	double * der1[3];
	double * der2[3];
	double * orbs = &m_workspace_2.at(0 * m_na * m_no);
	der1[0] = &m_workspace_2.at(1 * m_na * m_no); // x 1st derivatives of the atomic orbitals
	der1[1] = &m_workspace_2.at(2 * m_na * m_no); // y 1st derivatives of the atomic orbitals
	der1[2] = &m_workspace_2.at(3 * m_na * m_no); // z 1st derivatives of the atomic orbitals
	der2[0] = &m_workspace_2.at(4 * m_na * m_no); // x 2nd derivatives of the atomic orbitals
	der2[1] = &m_workspace_2.at(5 * m_na * m_no); // y 2nd derivatives of the atomic orbitals
	der2[2] = &m_workspace_2.at(6 * m_na * m_no); // z 2nd derivatives of the atomic orbitals
	double *const proda = &m_workspace.at(7 * m_na * m_no + 0 * m_na * m_na);

	double * gradlog_psi_ddv_grad = &m_workspace_2.at(8 * m_na * m_no);	// for grad.grad KE term
	double * gradlog_det_ddv_grad = &m_workspace_2.at(9 * m_na * m_no);	// for lap KE term

	// populated by compute_ddv_dnlogpsi() --- only alpha or beta
	double * ddv_laplogpsi = &m_workspace_2.at(10 * m_na * m_no);		// for lap KE term
	double * ddv_gradlogpsi[3];
	ddv_gradlogpsi[0] = &m_workspace_2.at(11 * m_na * m_no + 0 * m_na * m_na * m_no); // [ [CM m_no x m_na mat of ddv grad_m log det]_m=0 ... []_m=m_na+nb]_x, alpha then beta
	ddv_gradlogpsi[1] = &m_workspace_2.at(11 * m_na * m_no + 1 * m_na * m_na * m_no); 
	ddv_gradlogpsi[2] = &m_workspace_2.at(11 * m_na * m_no + 2 * m_na * m_na * m_no); 

	double * ddv_gradlogpsi_lap[3];
	ddv_gradlogpsi_lap[0] = &m_workspace_2.at(11 * m_na * m_no + 3 * m_na * m_na * m_no); // [ [CM m_no x m_na mat of ddv grad_m log det]_m=0 ... []_m=m_na+nb]_x, alpha then beta
	ddv_gradlogpsi_lap[1] = &m_workspace_2.at(11 * m_na * m_no + 4 * m_na * m_na * m_no); 
	ddv_gradlogpsi_lap[2] = &m_workspace_2.at(11 * m_na * m_no + 5 * m_na * m_na * m_no); 

	// get der1 and der2 
	// For future speed up of the code get these from compute_ke_pieces instead of recalculating
	essqc::qmc::ABDeterminant::get_orbs_and_derivs(m_e1pos, m_nuc, orbs, der1, der2);

	// need ddv_gradlogpsi piece for both grad grad and lap ke term contribution
	essqc::qmc::ABDeterminant::compute_ddv_dnlogpsi(ddv_gradlogpsi, ddv_laplogpsi, der1, der2, 0);
	essqc::dcopy(m_na * m_no*m_na, ddv_gradlogpsi[0], 1, ddv_gradlogpsi_lap[0], 1);
	essqc::dcopy(m_na * m_no*m_na, ddv_gradlogpsi[1], 1, ddv_gradlogpsi_lap[1], 1);
	essqc::dcopy(m_na * m_no*m_na, ddv_gradlogpsi[2], 1, ddv_gradlogpsi_lap[2], 1);

	double ddmu_i_scalar = 0.0;
	for (int mu	= 0; mu < 3; mu++){ 

		// get contributions involving the term   Tr[ (XC)^-1 dX/dri C ]
		// proda = dX/dmu C
		essqc::dgemm('N', 'N', m_na, m_na, m_no, 1.0, der1[mu], m_na, &m_C.at(0), m_no, 0.0, proda, m_na);
		
		for (int i = 0; i < m_na; i++)
		{

			// i need just the determinants contribution to gradlogpsi which is also already calculated in compute_ke_pieces
			// (dX/dri C)_ithrow . XCi_ithcol
			const double d2dmu2_i_scalar = essqc::ddot(m_na, proda + i, m_na, &m_xci.at(i * m_na), 1);

  	      	// multiply the d/dc gradlogpsi no x na mat by the grad log psi element for that elec/mu's matrix
			if (m_e1s == 0)
				ddmu_i_scalar = grad_log_psi[3*i + mu];
			if (m_e1s == 1)
				ddmu_i_scalar = grad_log_psi[3*m_na + 3*i + mu];

			// add scaled d/dc gradlogpsi matrix to gradlog_psi_ddv_grad running sum 
			essqc::dscal(m_no*m_na, ddmu_i_scalar, ddv_gradlogpsi[mu] + i * m_no*m_na, 1);
			essqc::daxpy(m_no*m_na, 1.0, ddv_gradlogpsi[mu] + i * m_no*m_na, 1, gradlog_psi_ddv_grad, 1);

			essqc::dscal(m_no*m_na, d2dmu2_i_scalar, ddv_gradlogpsi_lap[mu] + i * m_no*m_na, 1);
			essqc::daxpy(m_no*m_na, 1.0, ddv_gradlogpsi_lap[mu] + i * m_no*m_na, 1, gradlog_det_ddv_grad, 1);
		}
	}

	// grad.grad contributions:

	// + 2*gradlog_psi_ddv_grad contribution to grad_E
	essqc::daxpy(m_no*m_na, 2.0, gradlog_psi_ddv_grad, 1, &grad_E_temp.at(0), 1);

	// laplacian contributions:

	// + ddv_laplogpsi contribution to grad_E
	essqc::daxpy(m_no*m_na, 1.0, ddv_laplogpsi, 1, &grad_E_temp.at(0), 1);

	// - 2*gradlog_det_ddv_grad contribution to grad_E
	essqc::daxpy(m_no*m_na, -2.0, gradlog_det_ddv_grad, 1, &grad_E_temp.at(0), 1);

	// multiply by this spins grad_E contribution by -1/2 and add to full grad_E
	//essqc::daxpy(m_no*m_na, 1.0, &grad_E_temp.at(0), 1, grad_E, 1);
	essqc::daxpy(m_no*m_na, -0.5, &grad_E_temp.at(0), 1, grad_E, 1);

}

// d/dc_ij (d^n/dmu^n log psi) = d^n/dmu^n(X)^T * XC^-1^T - X^T * XC^-1^T * C^T * d^n/dmu^n(X)^T * XC^-1^T	
// ddv_gradlogpsi [3,  na * no*na] = [xyz, [noxna col major ddc_ij ddmu_elec gradlogpsi ]_elec(alpha or beta) ... ]
// ddv_laplogpsi [no*na] = same as grad but summed over mu and elec
void essqc::qmc::ABDeterminant::compute_ddv_dnlogpsi(double ** ddv_gradlogpsi, double * ddv_laplogpsi, double ** der1, double ** der2, int indx)
{
	// same math structure is needed for ddv_gradlogpsi and ddv_laplogpsi

	const int lm_space_needed = 3 * m_na * m_no;
  	std::vector<double> lm_workspace(lm_space_needed, 0.0);	

	double * d1A_naxno = &lm_workspace.at(0 * m_na * m_no);
	double * d2A_naxno = &lm_workspace.at(1 * m_na * m_no);
	double * ddv_laplogpsi_temp = &lm_workspace.at(2 * m_na * m_no);

	// loop over x, y, and z
	for (int mu = 0; mu < 3; mu++)
	{

		// loop over m_na to get d/dc_ij (d/dmu_i log det(XC)), no x na matrix corresonding to ith elec
		for (int i = 0; i < m_na; i++)
		{
			//std::cout << "elec ind = " << i << std::endl;

			// reinitialize A_naxno to 0 - bc dger adds to existing A_naxno
			essqc::dscal(m_no*m_na, 0.0, d1A_naxno, 1);
			essqc::dscal(m_no*m_na, 0.0, d2A_naxno, 1);
			essqc::dscal(m_no*m_na, 0.0, ddv_laplogpsi_temp, 1);

      		// GOOD
			// d1A_naxno = outer product of dX/dmu ith row (length no) by XC^-1 ith col (length na)
			essqc::dger(m_no, m_na, 1.0, der1[mu] + i, m_na, &m_xci.at(i * m_na), 1, d1A_naxno, m_no);	// outer product CORRECT

			// output ddv_gradlogpsi[mu] to ith position
			essqc::qmc::ABDeterminant::outerprod_to_ddv_dnlogpsi(d1A_naxno, ddv_gradlogpsi[mu] + m_na*m_no * i);
			// d2A_naxno = outer product of dX/dmu ith row (length no) by C ith col (length na)
			essqc::dger(m_no, m_na, 1.0, der2[mu] + i, m_na, &m_xci.at(i * m_na), 1, d2A_naxno, m_no);	// outer product CORRECT

			// output to running total of ddv_laplogpsi 
			essqc::qmc::ABDeterminant::outerprod_to_ddv_dnlogpsi(d2A_naxno, ddv_laplogpsi_temp);
			essqc::daxpy(m_no*m_na, 1.0, ddv_laplogpsi_temp, 1, ddv_laplogpsi, 1);
		}

	} // mu for loop

}

// d/dc_ij (dn log psi) = dnX^T * XC^-1^T - X^T * XC^-1^T * C^T * dnXi^T * XC^-1^T	[3, m_no, m_na]
// n can represent gradient or laplacian term
// A_naxno is the no x na outerproduct of dnX^T_ithcol * C_ithrowA
// populates ddv_dnlogpsi, an no x na CM vector corresponding to an electron and mu
void essqc::qmc::ABDeterminant::outerprod_to_ddv_dnlogpsi(double * A_naxno, double * ddv_dnlogpsi){

 	std::vector<double> B_naxna(m_na * m_na, 0.0);	
	double * B_naxna_ptr = B_naxna.data(); 

 	std::vector<double> B_naxna_temp(m_na * m_na, 0.0);	
	double * B_naxna_temp_ptr = B_naxna_temp.data(); 

	// d/dc_ij (dn log psi) = dnX^T * XC^-1^T < adding first tern to contribut to grad log psi
	essqc::daxpy(m_na*m_no, 1.0, A_naxno, 1, ddv_dnlogpsi, 1);
  
    // GOOD
	// C^T * A_naxno: multiply the transpose of m_C by A_naxno and save the result to B_naxna
	essqc::dgemm('T', 'N', m_na, m_na, m_no, 1.0, &m_C.at(0), m_no, A_naxno, m_no, 0.0, B_naxna_ptr, m_na);	// B_naxna is [m_na, m_na]

    // GOOD - 02102025 - fix H bug by assigning output to B_temp instead of overwritting B
	// XC^-1^T * A_naxno: multiply transpose of m_xci by B_naxna_ptr and save the result to B_naxna
	essqc::dgemm('T', 'N', m_na, m_na, m_na, 1.0, &m_xci.at(0), m_na, B_naxna_ptr, m_na, 0.0, B_naxna_temp_ptr, m_na);	// B_naxna is [m_na, m_na]

	// X^T * A_naxno: multiply m_x^T by B_naxna_ptr and save -= to ddv_dnlogpsi = A_naxno 
	// d/dc_ij (dn log psi) = dnX^T * XC^-1^T - X^T * XC^-1^T * C^T * dnXi^T * XC^-1^T	--- [m_no, m_na] CM vector
	essqc::dgemm('T', 'N', m_no, m_na, m_na, -1.0, &m_x.at(0), m_na, B_naxna_temp_ptr, m_na, 1.0, ddv_dnlogpsi, m_no);	// ddv_gradlogpsi = A_naxno is [m_no, m_na]

}

// compute LM and derivatives values to check derivatives against finite difference test
// ddv_gradlogpsi is the derivative of the gradient of the log psi wrt the variational parameters (3 x m_no x m_na)	for alpha or beta <-- pop in compute_ddv_dnlogpsi
void essqc::qmc::ABDeterminant::LM_deriv_test(double * const grad_log_psi, double ** ddv_gradlogpsi, double * ddv_laplogpsi, double ** der1, double ** der2, const double fd_delta){

	essqc::qmc::ABDeterminant::compute_ddv_dnlogpsi(ddv_gradlogpsi, ddv_laplogpsi, der1, der2, 0);	// 0 is arbirary index
}

// compute orbital and derivative values to check derivatives against finite difference test
void essqc::qmc::ABDeterminant::orbs_and_derivs_test(double * orbs, double ** der1, double ** der2, const double fd_delta){

	// copy temp orbs to orbs (alpha or beta)
	for (int i = 0; i < m_na * m_no; i++) {
		*(orbs + i) = m_x[i];
	}

    	double val = 0.0;
	double* dummy_orb = &val; 
	essqc::qmc::ABDeterminant::get_orbs_and_derivs(m_e1pos, m_nuc, dummy_orb, der1, der2);
	
	// Assume der1 and der2 are 2D arrays with dimensions rows x cols
	int rows = 3; // The number of rows in the 2D arrays
	int cols = m_na * m_no; // The number of columns in the 2D arrays

}
