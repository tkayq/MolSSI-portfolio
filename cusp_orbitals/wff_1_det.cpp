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

	// 4th order polynomial math 

	// cusp switching function parameters
	//const double c1 = (-3.0 / pow(cusp_radius, 4));
	//const double c2 = (8.0 / pow(cusp_radius, 3));
	//const double c3 = (-6.0 / pow(cusp_radius, 2));
	//const double c4 = 0.0;
	//const double c5 = 1.0;

	//const double abs_nuc_dist = std::abs(nuc_dist);

	//const double b_val = c1 * pow(abs_nuc_dist, 4) + c2 * pow(abs_nuc_dist, 3) + c3 * pow(abs_nuc_dist, 2) + c4 * abs_nuc_dist + c5;
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
	// 4th order polynomial math

	// cusp switching function parameters
	//const double c1 = (-3.0 / pow(cusp_radius, 4));
	//const double c2 = (8.0 / pow(cusp_radius, 3));
	//const double c3 = (-6.0 / pow(cusp_radius, 2));
	//const double c4 = 0.0;
	//const double c5 = 1.0;

	//// loop through xyz to calc derivative terms
	//for (int l = 0; l < 3; l++)
	//{
	//  //const double diff = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[n]), l); // (x_i - X_nuc)
	//  d1_b[l] = diff[l] * (4.0 * c1 * pow(nuc_dist, 2) + 3.0 * c2 * nuc_dist + 2.0 * c3 + c4 / nuc_dist);
	//}
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

	// 4th order polynomial math
	// cusp switching function parameters
	//const double c1 = (-3.0 / pow(cusp_radius, 4));
	//const double c2 = (8.0 / pow(cusp_radius, 3));
	//const double c3 = (-6.0 / pow(cusp_radius, 2));
	//const double c4 = 0.0;
	//const double c5 = 1.0;

	//// loop through xyz to calc derivative terms
	//for (int l = 0; l < 3; l++)
	//{
	//  //const double diff = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[n]), l); // (x_i - X_nuc)
	//  d2_b[l] = c1 * (4.0 * pow(nuc_dist, 2) + 8.0 * pow(diff[l], 2)) + 3.0 * c2 * (pow(diff[l], 2) / nuc_dist + nuc_dist) + c4 * (1.0 / nuc_dist - pow(diff[l], 2) / pow(nuc_dist, 3)) + 2.0 * c3;
	//}
}

void essqc::qmc::ABDeterminant::d1_one_minus_b(double & orb_total, double & b_val, double (&der1_total)[3], double (&d1_b_vec)[3], double (&d1_Pn_vec)[3]) {
	for (int l = 0; l < 3; l++) {

		d1_Pn_vec[l] += (1.0 - b_val) * der1_total[l] - d1_b_vec[l] * orb_total;  
	}

} 

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
	//double N = 1 / std::sqrt( 4 * 3.1415926 ) * pow(zeta, 1.5) / ( 2.0 * std::sqrt(2.0) );
	//double Q_fn = N * (2.0 - zeta * r) * std::exp(-zeta * r / 2.0);
	return Q_fn;
}

// PASSED
// NON-orthogonalized gaussian s orbital value
double essqc::qmc::ABDeterminant::STOnG_s(const double r, const double pi, int ao_ind)
{
	//std::cout << " ~~STOnG_s~~ ";
	double orb_total = 0.0;
	for (int k = 0; k < m_ng; k++)
	{
		double a = m_bf_exp[k * m_no + ao_ind];
		double d = m_bf_coeff[k * m_no + ao_ind];

		//std::cout << " " << d * pow(((2.0 * a)/pi), 0.75) * std::exp(-a * r * r );
		orb_total += d * pow(((2.0 * a) / pi), 0.75) * std::exp(-a * r * r);
	}
	//std::cout << std::endl;

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
		//double N = 1 / std::sqrt( 4 * 3.1415926 ) * pow(zeta, 1.5) / ( 2.0 * std::sqrt(2.0) );
		//d1_Q_vec[l] = -N * zeta * diff[l] / r * std::exp(-zeta * r / 2.0) * (2.0 - zeta * r / 2.0);
		//const double diff = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[n]), l); // (x_i - X_nuc)
		d1_Q_vec[l] = -(a0 * zeta * diff[l] * std::exp(-zeta * r)) / r;
	}
}

// vector of laplacian elements of the slater s orbital for gaussian cusps --- CHECKED
void essqc::qmc::ABDeterminant::d2_slater_s_cusp(double a0, double zeta, double r, int i, int n, double diff[3], double d2_Q_vec[3])
{
	for (int l = 0; l < 3; l++) // xyz
	{
		//const double diff = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[n]), l); // (x_i - X_nuc)
		//double N = 1 / std::sqrt( 4 * 3.1415926 ) * pow(zeta, 1.5) / ( 2.0 * std::sqrt(2.0) );
		//d2_Q_vec[l] = -N * zeta * std::exp(-zeta * r / 2.0) * ( (1 / r - pow(diff[l], 2.0) / pow(r, 3.0)) * (2.0 - zeta * r / 2.0) - (zeta * pow(diff[l], 2) / (2.0 * pow(r, 2))) * (3 - zeta * r / 2.0)); 
		d2_Q_vec[l] = a0 * zeta * std::exp(-zeta * r) * ((zeta * pow(diff[l], 2)) / pow(r, 2) + pow(diff[l], 2) / pow(r, 3) - 1 / r);
	}
	//return d2_Q_vec;
}

// basis function for new cusped basis
double essqc::qmc::ABDeterminant::Pn_func(double b, double Q, double orb_total, double nuc_dist, double n, double rc) {
	//return (1.0 - b) * orb_total + b * Q * pow(nuc_dist / rc, n);
	return b * Q * pow(nuc_dist / rc, n);
}

// vector of gradient of new P basis function
void essqc::qmc::ABDeterminant::d1_Pn_func(double & orb_total, double & b_val, double & Q_fn, double (&der1_total)[3], double (&d1_b_func)[3], double (&d1_slater_s_cusp)[3], double & nuc_dist, double (&diff)[3], double & n, double & qn, double (&d1_P_vec)[3], double & rc) {

	for (int l = 0; l < 3; l++) {

		//d1_P_vec[l] += qn * (der1_total[l] + b_val * ( d1_slater_s_cusp[l] * pow(nuc_dist / rc, n) + n * Q_fn * pow(nuc_dist / rc, (n-1.0)) * diff[l] / (nuc_dist * rc) - der1_total[l] ) + d1_b_func[l] * ( Q_fn * pow(nuc_dist / rc, n) - orb_total )); 
		d1_P_vec[l] += qn * ( b_val * ( d1_slater_s_cusp[l] * pow(nuc_dist / rc, n) + n * Q_fn * pow(nuc_dist / rc, (n-1.0)) * diff[l] / (nuc_dist * rc)) + d1_b_func[l] * ( Q_fn * pow(nuc_dist / rc, n) )); 

	}

}

// vector of laplacian of new P basis function
void essqc::qmc::ABDeterminant::d2_Pn_func(double & orb_total, double & b_val, double & Q_fn, double (&der1_total)[3], double (&d1_b_func)[3], double (&d1_slater_s_cusp)[3], double (&der2_total)[3], double (&d2_b_func)[3], double (&d2_slater_s_cusp)[3], double & nuc_dist, double (&diff)[3], double & n, double & qn, double (&d2_P_vec)[3], double & rc) {

	for (int l = 0; l < 3; l++) {

		//d2_P_vec[l] += qn * (der2_total[l] + b_val * ( d2_slater_s_cusp[l] * pow(nuc_dist / rc, n) + 2.0 * d1_slater_s_cusp[l] * n / rc * pow(nuc_dist / rc, (n-1.0)) * diff[l] / nuc_dist + Q_fn * n * (n-1.0) * pow(nuc_dist / rc, (n-2.0)) * pow(diff[l], 2) / pow(nuc_dist, 2) / pow(rc, 2) + n / rc * Q_fn * pow(nuc_dist / rc, (n-1.0)) * (1.0 / nuc_dist - pow(diff[l], 2) / pow(nuc_dist, 3)) - der2_total[l] ) + 2.0 * d1_b_func[l] * ( d1_slater_s_cusp[l] * pow(nuc_dist / rc, n) + Q_fn * n / rc * pow(nuc_dist / rc, (n-1.0)) * diff[l] / nuc_dist - der1_total[l] ) + d2_b_func[l] * ( Q_fn * pow(nuc_dist / rc, n) - orb_total));
		d2_P_vec[l] += qn * ( b_val * ( d2_slater_s_cusp[l] * pow(nuc_dist / rc, n) + 2.0 * d1_slater_s_cusp[l] * n / rc * pow(nuc_dist / rc, (n-1.0)) * diff[l] / nuc_dist + Q_fn * n * (n-1.0) * pow(nuc_dist / rc, (n-2.0)) * pow(diff[l], 2) / pow(nuc_dist, 2) / pow(rc, 2) + n / rc * Q_fn * pow(nuc_dist / rc, (n-1.0)) * (1.0 / nuc_dist - pow(diff[l], 2) / pow(nuc_dist, 3)) ) + 2.0 * d1_b_func[l] * ( d1_slater_s_cusp[l] * pow(nuc_dist / rc, n) + Q_fn * n / rc * pow(nuc_dist / rc, (n-1.0)) * diff[l] / nuc_dist) + d2_b_func[l] * ( Q_fn * pow(nuc_dist / rc, n)));

	}

}

// PASSED 
// vector of gradient UN-cusped NON-orthogonalized gaussian of s orbital for elctron i in orb p
void essqc::qmc::ABDeterminant::d1_STOnG_s(const double r, const double pi, int ao_ind, const double delta[3], double d1_vec[3]) 
{
	for (int k = 0; k < m_ng; k++)
	{
		double a = m_bf_exp[k * m_no + ao_ind];
		double d = m_bf_coeff[k * m_no + ao_ind];
		//const double Ns = 2.0 * a * d * pow((2.0 * a / pi), 0.75);
		d1_vec[0] += -delta[0] * 2.0 * a * d * pow((2.0 * a / pi), 0.75) * std::exp(-a * r * r);
		d1_vec[1] += -delta[1] * 2.0 * a * d * pow((2.0 * a / pi), 0.75) * std::exp(-a * r * r);
		d1_vec[2] += -delta[2] * 2.0 * a * d * pow((2.0 * a / pi), 0.75) * std::exp(-a * r * r);
		//std::cout << "new " << k << ": \t"  << "\t\t" << std::exp(-a * r * r) << std::endl;
		//std::cout << "new " << k << ": \t"  << "\t\t" << a << "\t\t" << d << "\t\t" << Ns << "\t\t" << std::exp(-a * r * r) << std::endl;
		//std::cout << "new " << k << ": \t" << d1_vec[0] << "\t" << d1_vec[1] << "\t" << d1_vec[2] << std::endl;
		//std::cout << "new a and d " << k << ": \t" << a << "\t" << d << std::endl;
		//std::cout << "new delta " << k << ": \t" << delta[0] << "\t" << delta[1] << "\t" << delta[2] << std::endl;
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

	//return d1_vec;
}
// vector of laplacian NON-orthogonalized gaussian of p orbital for ith electron in pth orbital
void essqc::qmc::ABDeterminant::d2_STOnG_p(const double r, const double pi, int ao_ind, const double delta[3], double d2_vec[3])
{
	//double d2_vec[3] = {0.0, 0.0, 0.0};

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
//  if (ao_type==5) {
//      std::cout << "3dxy:" << std::endl;
//      std::cout << "d1_vec[0]: " << d1_vec[0] << std::endl;
//      std::cout << "d1_vec[1]: " << d1_vec[1] << std::endl;
//      std::cout << "d1_vec[2]: " << d1_vec[2] << std::endl;
//  }
//  else if (ao_type==6) {
//      std::cout << "3dyz:" << std::endl;
//      std::cout << "d1_vec[0]: " << d1_vec[0] << std::endl;
//      std::cout << "d1_vec[1]: " << d1_vec[1] << std::endl;
//      std::cout << "d1_vec[2]: " << d1_vec[2] << std::endl;
//  }
//  else if (ao_type==7) {
//      std::cout << "3dz2:" << std::endl;
//      std::cout << "d1_vec[0]: " << d1_vec[0] << std::endl;
//      std::cout << "d1_vec[1]: " << d1_vec[1] << std::endl;
//      std::cout << "d1_vec[2]: " << d1_vec[2] << std::endl;
//  }
//  else if (ao_type==8) {
//      std::cout << "3dxz:" << std::endl;
//      std::cout << "d1_vec[0]: " << d1_vec[0] << std::endl;
//      std::cout << "d1_vec[1]: " << d1_vec[1] << std::endl;
//      std::cout << "d1_vec[2]: " << d1_vec[2] << std::endl;
//  }
//  else {
//      std::cout << "3dx2-y2:" << std::endl;
//      std::cout << "d1_vec[0]: " << d1_vec[0] << std::endl;
//      std::cout << "d1_vec[1]: " << d1_vec[1] << std::endl;
//      std::cout << "d1_vec[2]: " << d1_vec[2] << std::endl;
//  }


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
//  if (ao_type==5) {
//      std::cout << "3dxy:" << std::endl;
//      std::cout << "d2_vec[0]: " << d2_vec[0] << std::endl;
//      std::cout << "d2_vec[1]: " << d2_vec[1] << std::endl;
//      std::cout << "d2_vec[2]: " << d2_vec[2] << std::endl;
//  }
//  else if (ao_type==6) {
//      std::cout << "3dyz:" << std::endl;
//      std::cout << "d2_vec[0]: " << d2_vec[0] << std::endl;
//      std::cout << "d2_vec[1]: " << d2_vec[1] << std::endl;
//      std::cout << "d2_vec[2]: " << d2_vec[2] << std::endl;
//  }
//  else if (ao_type==7) {
//      std::cout << "3dz2:" << std::endl;
//      std::cout << "d2_vec[0]: " << d2_vec[0] << std::endl;
//      std::cout << "d2_vec[1]: " << d2_vec[1] << std::endl;
//      std::cout << "d2_vec[2]: " << d2_vec[2] << std::endl;
//  }
//  else if (ao_type==8) {
//      std::cout << "3dxz:" << std::endl;
//      std::cout << "d2_vec[0]: " << d2_vec[0] << std::endl;
//      std::cout << "d2_vec[1]: " << d2_vec[1] << std::endl;
//      std::cout << "d2_vec[2]: " << d2_vec[2] << std::endl;
//  }
//  else {
//      std::cout << "3dx2-y2:" << std::endl;
//      std::cout << "d2_vec[0]: " << d2_vec[0] << std::endl;
//      std::cout << "d2_vec[1]: " << d2_vec[1] << std::endl;
//      std::cout << "d2_vec[2]: " << d2_vec[2] << std::endl;
//  }

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

// value of the gaussian UN-cusped NO-orthogonalization orbital of some electron i in some orbital p
/*double STOnG_orb_val(const PosVec &e_pos, const PosVec &n_pos, const double pi, const int p, const int i, double &orb_val)
  {

// get numbers of electrons and nuclei
const int ne = e_pos.nparticles();
const int nn = n_pos.nparticles();
double orb_total = 0;

const double dist = PosVec::single_particle_distance(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p]));
// gaussian s orbital
if (m_bf_orbs[p] == 0 or m_bf_orbs[p] == 1)
{
orb_total = STOnG_s(dist, pi, k * m_no + p);
// orb_total += d * pow(((2.0 * a)/pi), 0.75) * std::exp(-a * dist * dist);
}

// gaussian p orbital
else if (m_bf_orbs[p] == 2 || m_bf_orbs[p] == 3 || m_bf_orbs[p] == 4)
{
const double dist_xi = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p]), m_bf_orbs[p] - 2); // m_bf_orbs[p]-2 maps p-orbital type to x y or z (ie px=2 so we need dist_xi == 0)
orb_total = STOnG_p(dist, dist_xi, pi, k * m_no + p);
}

return orb_total;
} */

void essqc::qmc::ABDeterminant::get_full_X_mat(const PosVec &e_pos, const PosVec &n_pos, std::vector<double> &xmat)
{
	//std::cout << "IN get_full_X_mat " << std::endl;
	const int ne = e_pos.nparticles();
	const int nn = n_pos.nparticles();
	//std::cout << " num electons and num nuc in get_full_X_mat: " << ne << " " << nn << std::endl;

	const double pi = 3.14159265359;

	//std::cout << "reassign xmat size" << std::endl;
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
		//std::cout << __FILE__ << " " << __LINE__ << std::endl;
		if (m_use_cusp) // cusped STOnG gaussian orbital basis
		{
			//std::cout << "orb_ind, elec_ind, e-nucdist, bf_orb, orthorb, orbtype, orb total (coreorbind, projterm) " << std::endl;
			for (int p = 0; p < m_no; p++)  // eval all AO columns of Xmat
			{
        //std::cout << "orb " << p << std::endl;
				for (int i = 0; i < ne; i++)	// eval provided rows of xmat 
				{ //std::cout << std::endl;
					//std::cout << p << " " << i;  
					// initialize here to print out if in cusp 
					double proj = 0.0;
					int core_orb_ind = 0;

					double orb_total = 0.0;
					const double dist = PosVec::single_particle_distance(e_pos.get_pos(i), n_pos.get_pos(m_bf_cen[p])); // btw elec an orbital center
          //std::cout << " " << dist << " " << m_bf_orbs[p] << " " << m_orth_orb[p]; // << " (" << std::exp(-dist) << ")";

					if (m_bf_orbs[p] == 0 || m_bf_orbs[p] == 1) 
					{
						if (m_orth_orb[p] == 0) // NO orth - 1s core
						{
							orb_total = STOnG_s(dist, pi, p);
							//std::cout << " 1score " << orb_total << " ";
						}
						else // orth - valence s orbital 
						{
							core_orb_ind = m_orth_orb[p] - 1;             // index of core orbital to orthogonalize against
							proj = m_proj_mat[core_orb_ind * m_no + p]; // get projection of orb onto core from python
							orb_total = STOnG_s(dist, pi, p, core_orb_ind, proj);
							//std::cout << " sorth_orb " << orb_total << " " << core_orb_ind << " " << proj;
						}
						//std::cout << std::endl;
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
					//std::cout << "xmat element BEFORE: " << xmat[i+p*ne] << std::endl;

					// loop thru nuc to correct cusp on given orb (p)
					for (int n = 0; n < nn; n++)
					{
						if (m_cusp_a0_mat[p * nn + n] != 0.0) 
						{
							const double nuc_dist = PosVec::single_particle_distance(e_pos.get_pos(i), n_pos.get_pos(n));
							//std::cout << "non-zero cusp_a0 matrix element, check e- position for cusp in get_full_X_mat() " << nuc_dist << " " << std::endl;

							double rc = m_cusp_radius_mat[p * nn + n];

							if (nuc_dist < rc) // cusp in nuc cusp_radius
							{
								// evaluate switching function
								b_val = b_func(m_cusp_radius_mat[p * nn + n], nuc_dist);
								double zeta = m_Z[n];

								// s orbital slater cusp
								Q_fn = slater_s_cusp(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist);

								// evaluate P functions
								//
								Pn_val += (1.0 - b_val) * orb_total;

								int npbf = m_n_vec.size();
								for (int k = 0; k < npbf; k++) { // sub with length of the index array
									Pn_val += m_cusp_coeff_mat[n * npbf + p * nn * npbf + k] * Pn_func(b_val, Q_fn, orb_total, nuc_dist, m_n_vec[k], rc); // change k to index of appropriate vector and coeff
								}

								//orb_ind, elec_ind, nuc_ind, cusp_val, gauss_val, slater_val, b_val, elec-nuc_dist, nuc x, y, z, elec x, y, z, elec-orb_dist, proj, core_orb_ind, zeta
								/*std::cout << p << ", " << i << ", " << n <<  ", " << boost::format("%.12f, ") % ((1 - b_val) * orb_total + b_val * Q_fn) << boost::format("%.12f, ") % orb_total << boost::format("%.12f, ") % Q_fn;
								  std::cout << boost::format("%.12f, ") % b_val << boost::format("%.12f, ") % nuc_dist; // nuc, elec dist
								  std::cout << boost::format("%.12f, %.12f, %.12f, ") % (n_pos.get_pos(n))[0] % (n_pos.get_pos(n))[1] % (n_pos.get_pos(n))[2];
								  std::cout << boost::format("%.12f, %.12f, %.12f, ") % (e_pos.get_pos(i))[0] % (e_pos.get_pos(i))[1] % (e_pos.get_pos(i))[2];
								  std::cout << boost::format("%.12f, ") % dist << proj << ", " <<  core_orb_ind << ", " <<  zeta << std::endl; // orbital, elec dist
								  */ 
								counter++;
							}   // if electron within cusp radius
						} 	  // if orbital over given nuc has a cusp, a0 != 0.0
					}    	  // nuclei centers

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

					//xmat[i + p * ne] = (1 - b_val) * orb_total + b_val * Q_fn;

				}   // electrons
			}     // orbitals
			//std::cout << std::endl;
		}         // with cusps
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

				}		// electrons
			}        	// orbitals
		}				// no cusps

	}                 	// use_GTO
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

	/*std::cout << std::endl;
	std::cout << "In Xmat in init: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_na, m_no, &m_x.at(0), "%14.6f");
	std::cout << std::endl;
	  
	std::cout << std::endl;
	std::cout << "In Cmat in init: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_no, m_na, &m_C.at(0), "%14.6f");
	std::cout << std::endl;
  */

	// Get XC ma
	essqc::dgemm('N', 'N', m_na, m_na, m_no, 1.0, &m_x.at(0), m_na, &m_C.at(0), m_no, 0.0, &m_xc.at(0), m_na);

	// Get XC inverse
	int lwork = m_na * m_na;
	std::vector<double> work(lwork, 0.0);
	std::vector<int> ipiv(m_na, 0);
	essqc::log_det_and_inverse(m_na, &m_xc.at(0), m_logdet, m_detsign, &m_xci.at(0), &ipiv.at(0), &work.at(0), lwork);
	//std::cout << __FILE__ << " DONE in initialize_internal_data()" << std::endl;

	/*////// PRINT ///////////
	std::cout << std::endl;
	std::cout << "In Xmat in init: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_na, m_no, &m_x.at(0), "%14.6f");
	std::cout << std::endl;
	  
	std::cout << std::endl;
	std::cout << "In Cmat in init: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_no, m_na, &m_C.at(0), "%14.6f");
	std::cout << std::endl;
	  
	std::cout << std::endl;
	std::cout << "In XCmat in init: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_na, m_na, &m_xc.at(0), "%14.6f");
	std::cout << std::endl;
	  
	std::cout << std::endl;
	std::cout << "In XC inverse mat in init: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_na, m_na, &m_xci.at(0), "%14.6f");
	std::cout << std::endl;
  */	
	  
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
		//std::cout << " SPIN MATCHED " << std::endl;
		//// evaluate the new values for the corresponding row of the X matrix of orbital values
		//std::cout << " in get_new_to_old(), ind: "<< m_mei.ind << std::endl;
		//std::cout << " m_pxrow BEFORE";

		//for ( auto element : m_pxrow)
		//    std::cout << element << " ";
		//std::cout << " ===> evaluate x mat with move ===> "; // << std::endl;
		essqc::qmc::ABDeterminant::get_full_X_mat(m_mei.pos, m_nuc, m_pxrow);

		/*std::cout << " m_pxrow AFTER";
		  for ( auto element : m_pxrow)
		  std::cout << element << " ";
		  std::cout << std::endl; */

		// evaluate the new values for the corresponding row of the XC matrix
		essqc::dgemm('N', 'N', 1, m_na, m_no, 1.0, &m_pxrow.at(0), 1, &m_C.at(0), m_no, 0.0, &m_pxcrow.at(0), 1);

		// get the ratio of the new to old determinant value
		return essqc::ddot(m_na, &m_pxcrow.at(0), 1, &m_xci.at(m_na * m_mei.ind), 1);
		// return 1.0;

		// return 0.5;
	}
}

// accept and update
void essqc::qmc::ABDeterminant::accept_poposed_move()
{
	//std::cout << "accepted ";

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
		//std::cout << __FILE__ << " " << __LINE__ << std::endl;
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
					//std::cout << " R " << dist << std::endl;
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
							//std::cout << std::endl;
							//std::cout << " orthogonalize in get_orbs_and_derivs ";
							int core_orb_ind = m_orth_orb[p] - 1;             // index of core orbital to orthogonalize against
							//std::cout << " core_orb_ind: " << core_orb_ind;
							double proj = m_proj_mat[core_orb_ind * m_no + p]; // get projection of orb onto core from python
							//std::cout << " proj of valence orb against core: " << proj << std::endl;

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
								//std::cout << "elec within range add cusp! orb/nuc ind: " << p << "/" << n << " nuc_dist " << nuc_dist << " elec: " << i << " orb_val " << orb_total; // << std::endl;
								for (int l = 0; l < 3; l++) {
									diff[l] = PosVec::single_particle_distance_xi(e_pos.get_pos(i), n_pos.get_pos(n), l); // (x_i - X_nuc)
								}

								// evaluate switching function
								b_val = b_func(m_cusp_radius_mat[p * nn + n], nuc_dist);
								d1_b_func(m_cusp_radius_mat[p * nn + n], nuc_dist, i, n, diff, d1_b_vec);
								d2_b_func(m_cusp_radius_mat[p * nn + n], nuc_dist, i, n, diff, d2_b_vec);

								double zeta = m_Z[n];
								// change cusp condition - if atom centered p-orbital - if we start cusping p-orbs
								//if (m_bf_orbs[p] > 1 && n == m_bf_cen[p])
								//    zeta = m_Z[n] / 2.0;

								// s orbital slater cusp
								Q_fn = slater_s_cusp(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist);
								d1_slater_s_cusp(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist, i, n, diff, d1_Q_vec);
								d2_slater_s_cusp(m_cusp_a0_mat[p * nn + n], zeta, nuc_dist, i, n, diff, d2_Q_vec);

								// evaluate P functions 
								//for (int p = 0; p < num_nuc_cusp; p++) {
								//  for (int i = 0; i < num_orb_cusp; i++) {
								//    for (int j = 0; j < npbf; j++) {
								//      std::cout << m_cusp_coeff_mat[p * npbf + i * num_nuc_cusp * npbf + j] << std::endl;
								//    }
								//  }
								//}
								//std::cout << "powers of n" << std::endl;
								int npbf = m_n_vec.size();

								d1_one_minus_b(orb_total, b_val, der1_total, d1_b_vec, d1_Pn_vec); 
								d2_one_minus_b(orb_total, b_val, der1_total, d1_b_vec, der2_total, d2_b_vec, d2_Pn_vec); 

								for (int k = 0; k < m_n_vec.size(); k++) { // change the k !!!  
									//double d1P[3] = {0.0, 0.0, 0.0};
									//double d2P[3] = {0.0, 0.0, 0.0};
									d1_Pn_func(orb_total, b_val, Q_fn, der1_total, d1_b_vec, d1_Q_vec, nuc_dist, diff, m_n_vec[k], m_cusp_coeff_mat[n * npbf + p * nn * npbf + k], d1_Pn_vec, rc);
									d2_Pn_func(orb_total, b_val, Q_fn, der1_total, d1_b_vec, d1_Q_vec, der2_total, d2_b_vec, d2_Q_vec, nuc_dist, diff, m_n_vec[k], m_cusp_coeff_mat[n * npbf + p * nn * npbf + k], d2_Pn_vec, rc);
									//d1_Pn_vec[0] += d1P[0];
									//d1_Pn_vec[1] += d1P[1];
									//d1_Pn_vec[2] += d1P[2];
									//d2_Pn_vec[0] += d2P[0];
									//d2_Pn_vec[1] += d2P[1];
									//d2_Pn_vec[2] += d2P[2];
									//std::cout << m_n_vec[k] << std::endl;
									//std::cout << m_cusp_coeff_mat[n * npbf + p * nn * npbf + k] << std::endl;
								}
								//std::cout << "d2_Pn_vec[1] " << d2_Pn_vec[1] << std::endl;
								//
								// ///////////////////////////////////
								//std::cout << std::endl;

								//std::cout << "   b_val " << b_val;
								//std::cout << "\t d1_b_vec ";
								//for (auto element : d1_b_vec)
								//	std::cout << element << " ";
								//std::cout << "\t d2_b_vec ";
								//for (auto element : d2_b_vec)
								//	std::cout << element << " ";

								//std::cout << std::endl;

								//std::cout << " Q_fn " << Q_fn;
								//std::cout << "\t d1_Q_vec ";
								//for (auto element : d1_Q_vec)
								//	std::cout << element << " ";
								//std::cout << "\t d2_Q_vec ";
								//for (auto element : d2_Q_vec)
								//	std::cout << element << " ";

								//std::cout << std::endl;
								//std::cout << "    elec within nuc range add cusp! orb/nuc ind: " << p << "/" << n << ", bval: " << b_val << ", Q_fn: " << Q_fn << " og d1: " << der1_total[0] << " " << der1_total[1] << " " << der1_total[2] << " new d1: ";
								//std::cout << " " << der1_total[0] + b_val * (d1_Q_vec[0] - der1_total[0]) + d1_b_vec[0] * (Q_fn - orb_total);
								//std::cout << " " << der1_total[1] + b_val * (d1_Q_vec[1] - der1_total[1]) + d1_b_vec[1] * (Q_fn - orb_total);
								//std::cout << " " << der1_total[2] + b_val * (d1_Q_vec[2] - der1_total[2]) + d1_b_vec[2] * (Q_fn - orb_total);
								//std::cout << ", orb_total: " << orb_total;
								//std::cout << "\t der1_total ";
								//for (auto element : der1_total)
								//	std::cout << element << " ";
								//std::cout << "\t der2_total ";
								//for (auto element : der2_total)
								//	std::cout << element << " ";
								//std::cout << std::endl; 


							}   // if electron in cusping region
						}       // if orbital over current nucleus is cusped, a0 != 0.0
					}			// nuclear centers

					//der1[0][i + p * ne] = d1_Q_vec[0];
					//der1[1][i + p * ne] = d1_Q_vec[1];
					//der1[2][i + p * ne] = d1_Q_vec[2];

					//der2[0][i + p * ne] = d2_Q_vec[0];// + b_val * (d2_Q_vec[0] - der2_total[0]) + 2 * d1_b_vec[0] * (d1_Q_vec[0] - der1_total[0]) + d2_b_vec[0] * (Q_fn - orb_total);
					//der2[1][i + p * ne] = d2_Q_vec[1];// + b_val * (d2_Q_vec[1] - der2_total[1]) + 2 * d1_b_vec[1] * (d1_Q_vec[1] - der1_total[1]) + d2_b_vec[1] * (Q_fn - orb_total);
					//der2[2][i + p * ne] = d2_Q_vec[2];// + b_val * (d2_Q_vec[2] - der2_total[2]) + 2 * d1_b_vec[2] * (d1_Q_vec[2] - der1_total[2]) + d2_b_vec[2] * (Q_fn - orb_total);

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
					//der1[0][i + p * ne] = der1_total[0] + b_val * (d1_Q_vec[0] - der1_total[0]) + d1_b_vec[0] * (Q_fn - orb_total);
					//der1[1][i + p * ne] = der1_total[1] + b_val * (d1_Q_vec[1] - der1_total[1]) + d1_b_vec[1] * (Q_fn - orb_total);
					//der1[2][i + p * ne] = der1_total[2] + b_val * (d1_Q_vec[2] - der1_total[2]) + d1_b_vec[2] * (Q_fn - orb_total);

					//der2[0][i + p * ne] = der2_total[0] + b_val * (d2_Q_vec[0] - der2_total[0]) + 2 * d1_b_vec[0] * (d1_Q_vec[0] - der1_total[0]) + d2_b_vec[0] * (Q_fn - orb_total);
					//der2[1][i + p * ne] = der2_total[1] + b_val * (d2_Q_vec[1] - der2_total[1]) + 2 * d1_b_vec[1] * (d1_Q_vec[1] - der1_total[1]) + d2_b_vec[1] * (Q_fn - orb_total);
					//der2[2][i + p * ne] = der2_total[2] + b_val * (d2_Q_vec[2] - der2_total[2]) + 2 * d1_b_vec[2] * (d1_Q_vec[2] - der1_total[2]) + d2_b_vec[2] * (Q_fn - orb_total);

					if (counter > 1)
						std::cout << "counter at: " << counter << std::endl;

					//std::cout << std::endl;
					//std::cout << "\t der1 ";
					//for (auto element : der1)
					//	std::cout << element << " ";
					//std::cout << "\t der2 ";
					//for (auto element : der2)
					//	std::cout << element << " ";
					//std::cout << std::endl;
					// no cusp deriv result
					//der1[0][i + p * ne] = der1_total[0];
					//der1[1][i + p * ne] = der1_total[1];
					//der1[2][i + p * ne] = der1_total[2];

					//der2[0][i + p * ne] = der2_total[0];
					//der2[1][i + p * ne] = der2_total[1];
					//der2[2][i + p * ne] = der2_total[2];

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

    
	/*std::cout << "In compute_psi, det of XC: " << det << " psi " << psi << std::endl;

	std::cout << "In compute_psi, print copied XC mat: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	  for (int i=0; i < m_na; i++) {		// loop through electrons to print each row

	  for (int j=0; j < m_na; j++) {		// loop through AOs
	  std::cout <<  boost::format("%14.6f") %  xc_copy[j*m_na + i] << "\t";
	  }
	  std::cout << std::endl;
	  }
	  std::cout << std::endl;

	  std::cout << "In compute_psi, print copied XC inverse mat: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	  for (int i=0; i < m_na; i++) {		// loop through electrons to print each row

	  for (int j=0; j < m_na; j++) {		// loop through AOs
	  std::cout <<  boost::format("%14.6f") %  xci_copy[j*m_na + i] << "\t";
	  }
	  std::cout << std::endl;
	  }
	  std::cout << std::endl;
	 */ 
	  
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

	//std::cout << "In compute_psi_piece_for_ratio, det of XC: " << det << " psi " << psi << std::endl;

	/*std::cout << "In compute_psi_piece_for_ratio, print copied XC mat: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	  for (int i=0; i < m_na; i++) {		// loop through electrons to print each row

	  for (int j=0; j < m_na; j++) {		// loop through AOs
	  std::cout <<  boost::format("%14.6f") %  xc_copy[j*m_na + i] << "\t";
	  }
	  std::cout << std::endl;
	  }
	  std::cout << std::endl;

	  std::cout << "In compute_psi_piece_for_ratio, print copied XC inverse mat: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	  for (int i=0; i < m_na; i++) {		// loop through electrons to print each row

	  for (int j=0; j < m_na; j++) {		// loop through AOs
	  std::cout <<  boost::format("%14.6f") %  xci_copy[j*m_na + i] << "\t";
	  }
	  std::cout << std::endl;
	  }
	  std::cout << std::endl;
	 */ 
}

void essqc::qmc::ABDeterminant::compute_ke_pieces(double *const d1logx, double *const d2logx)
{
	/*std::cout << "d2logx Before filling: "; 
	for (int p = 0; p < m_na * 2; p++)	// loop ove num total e-
	  std::cout << d2logx[p] << " ";
	std::cout << std::endl;
	*/

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

	//std::cout << " --- in compute_ke_pieces calling get_orbs_and_derivs" << std::endl; 
	// get orbital values and derivatives
	essqc::qmc::ABDeterminant::get_orbs_and_derivs(m_e1pos, m_nuc, orbs, der1, der2);

  /*std::cout << " --- in compute_ke_pieces after calling get_orbs_and_derivs" << std::endl;
	std::cout << "der1:" << std::endl;
	for (int mu = 0; mu < 3; mu++)
	{
		for (int i = 0; i < m_na * m_no; i++)
		{
			std::cout << der1[mu][i] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << "der2:" << std::endl;
	for (int mu = 0; mu < 3; mu++)
	{
		for (int i = 0; i < m_na * m_no; i++)
		{
			std::cout << der2[mu][i] << " ";
		}
		std::cout << std::endl;
	}
  */
  	

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

	/*std::cout << std::endl;
	std::cout << "d1logx: "; //std::endl;
	for (int p = 0; p < m_na * 2 * 3; p++)
	  std::cout << d1logx[p] << " ";
	std::cout << std::endl;
	std::cout << "d2logx: "; //std::endl;
	for (int p = 0; p < m_na * 2; p++)
	  std::cout << d2logx[p] << " ";
	std::cout << std::endl;
	
	std::cout << "In compute_ke_pieces, print Xmat: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	for (int i=0; i < m_na; i++) {		// loop through electrons to print each row
	
	    for (int j=0; j < m_no; j++) {		// loop through AOs
	        std::cout <<  boost::format("%14.6f") %  m_x[j*m_na + i] << "\t";
	    }
	    std::cout << std::endl;
	}
	std::cout << std::endl;
  */
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
	//std::cout << " compute_PvP0 in wff_1_det " << __LINE__ << " do_opt, False if yes, True if no opt" << std::endl;

	// X^T * [(XC)^-1]^T ---> PvP0, dgemm dim inputs checks
	essqc::dgemm('T', 'T', m_no, m_na, m_na, 1.0, &m_x.at(0), m_na, &m_xci.at(0), m_na, 1.0, PvP0, m_no);
	//essqc::dgemm('T', 'T', m_no, m_na, m_na, 1.0, &m_x.at(0), m_na, &m_xci.at(0), m_na, 1.0, &PvP0[indx], m_no);

	
	/*std::cout << std::endl;
	std::cout << "In compute_PvP0, print Xmat: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_na, m_no, &m_x.at(0), "%14.6f");
	std::cout << std::endl;
	std::cout << "In compute_PvP0, print xc inverse: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_na, m_na, &m_xci.at(0), "%14.6f");
	std::cout << std::endl;
	std::cout << "In compute_PvP0, print PvP0: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_no, m_na, PvP0, "%14.6f");
	std::cout << std::endl;
	*/
}

// d/dc_ij (H Psi / Psi) = -1/2 d/dc_ij lap(psi) - (grad(psi)) . (d/dc_ij grad(psi))
//  (lap term)           = -1/2 lap_X^T [(XC)^-1]^T + 1/2 X^T [C (XC)^-1]^T lap_X^T [(XC)^-1]^T + grad_X^T K^T + X^T [C (XC)^-1]^T grad_X^T [K^-1]^T 
//  (grad.grad term)       - grad_X^T Q^T + X^T [(XC)^-1]^T C^T grad_X^T [Q^-1]^T }_ij	where Q is XC^-1 with every row (ie mth elec) is scaled by d/d mu_m gradlogpsi
//
//  OR (rearranged to be reimplemented in the code): BELOW IS WHAT IS IMPLEMENTED
//
//  (grad.grad term)       - grad_X^T(scaled by gradlogpsi) [(XC)^-1]^T + X^T [(XC)^-1]^T C^T grad_X^T(scaled by gradlogpsi) [(XC)^-1]^T }_ij	
//  ^^^ grad grad currently only being calculated - TKQ 07152024
//void essqc::qmc::ABDeterminant::compute_grad_E(double *const grad_E, double *const grad_log_psi, int indx)
//{
//
//	std::cout << " compute_gradE in wff_1_det - currently hacked to only return grad( grad.grad term of KE) " << __LINE__ << " nvp " << get_num_of_vp() << std::endl;
//
//	// get room to store the first and second derivatives
//	const int space_needed_2 = 7 * m_na * m_no + m_na * m_na;
//	if (m_workspace_2.size() < space_needed_2)
//		m_workspace_2.assign(space_needed_2, 0.0);
//
//	// get pointers to where we will put orbital values and derivatives
//	double * der1[3];
//	double * der2[3];
//	double * orbs = &m_workspace_2.at(0 * m_na * m_no);
//	der1[0] = &m_workspace_2.at(1 * m_na * m_no); // x 1st derivatives of the atomic orbitals
//	der1[1] = &m_workspace_2.at(2 * m_na * m_no); // y 1st derivatives of the atomic orbitals
//	der1[2] = &m_workspace_2.at(3 * m_na * m_no); // z 1st derivatives of the atomic orbitals
//	der2[0] = &m_workspace_2.at(4 * m_na * m_no); // x 2nd derivatives of the atomic orbitals
//	der2[1] = &m_workspace_2.at(5 * m_na * m_no); // y 2nd derivatives of the atomic orbitals
//	der2[2] = &m_workspace_2.at(6 * m_na * m_no); // z 2nd derivatives of the atomic orbitals
//
//	// get pointers to where we will put products of matrices
//	double *const proda = &m_workspace_2.at(7 * m_na * m_no + 0 * m_na * m_na);
//
//	// get orbital values and derivatives
//	essqc::qmc::ABDeterminant::get_orbs_and_derivs(m_e1pos, m_nuc, orbs, der1, der2);
//
//    // TODO access der1 and der2 from compute_ke_pieces instead of recalculating
//
//    /*std::cout << " --- in compute_grad_E after calling get_orbs_and_derivs - make sure they match compute_ke" << std::endl;
//	std::cout << "der1:" << std::endl;
//	for (int mu = 0; mu < 3; mu++)
//	{
//		for (int i = 0; i < m_na * m_no; i++)
//		{
//			std::cout << der1[mu][i] << " ";
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//	std::cout << "der2:" << std::endl;
//	for (int mu = 0; mu < 3; mu++)
//	{
//		for (int i = 0; i < m_na * m_no; i++)
//		{
//			std::cout << der2[mu][i] << " ";
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//    */
//
//	const int ddvspace_needed = 2 * m_na * m_no + m_na * m_na;
//	if (m_ddvworkspace.size() < ddvspace_needed)
//		m_ddvworkspace.assign(ddvspace_needed, 0.0);
//
//	// space for the LM matrix math pieces - should these be const pointers?
//	double * gradgradterm = &m_ddvworkspace.at(0 * m_na * m_no);
//	double * A_naxno = &m_ddvworkspace.at(1 * m_na * m_no);
//	double * B_naxna = &m_ddvworkspace.at(2 * m_na * m_no + 0 * m_na * m_na);
//
//	// I think I can condense gradgradterm into A_naxno workspace
//
//	// loop over x, y, and z
//	for (int mu = 0; mu < 3; mu++)
//	{
//		essqc::dscal(m_no*m_na, 0.0, A_naxno, 1);
//
//        // TODO remove this from loop - taking excess of memory
//		essqc::dcopy(m_na*m_no, der1[mu], 1, A_naxno, 1);
//
//		for (int i = 0; i < m_na; i++)
//		{
//
//			// scale every ith row of d/dmu X by grad_i,mu logpsi => A_naxno 
//			if (m_e1s == 0)
//			{
//				// dscal: length of vector, scalar, vector, space btw elements of vector
//				essqc::dscal(m_na, grad_log_psi[3 * i + mu], A_naxno + i, m_na);	// ith row of grad_mu X (aka A_naxno) * mu-th term of ith elec in grad log psi
//			}
//			if (m_e1s == 1)
//			{
//				essqc::dscal(m_na, grad_log_psi[3 * m_na + 3 * i + mu], A_naxno + i, m_na);	// ith row of grad_mu X (aka A_naxno) * mu-th term of ith elec in grad log psi
//			}
//		}
//
//		//  gradgradterm = grad_X^T(scaled by gradlogpsi) [(XC)^-1]^T 
//		// multiply A_naxno^T with m_xci^T and save the result to gradgradterm
//		essqc::dgemm('T', 'T', m_no, m_na, m_na, 1.0, A_naxno, m_na, &m_xci.at(0), m_na, 0.0, gradgradterm, m_no);	// grad grad term is [m_no, m_na]
//
//		// C^T grad_X^T(scaled by gradlogpsi) [(XC)^-1]^T =>B_naxna
//		// multiply the transpose of m_C by gradgradterm and save the result to B_naxna
//		essqc::dgemm('T', 'N', m_na, m_na, m_no, 1.0, &m_C.at(0), m_no, gradgradterm, m_no, 0.0, B_naxna, m_na);	// B_naxna is [m_na, m_na]
//
//		// [(XC)^-1]^T C^T grad_X^T(scaled by gradlogpsi) [(XC)^-1]^T =>B_naxna
//		// multiply transpose of m_xci by B_naxna and save the result to B_naxna
//		essqc::dgemm('T', 'N', m_na, m_na, m_na, 1.0, &m_xci.at(0), m_na, B_naxna, m_na, 0.0, B_naxna, m_na);	// B_naxna is [m_na, m_na]
//
//		//  gradgradterm = grad_X^T(scaled by gradlogpsi) [(XC)^-1]^T + X^T [(XC)^-1]^T C^T grad_X^T(scaled by gradlogpsi) [(XC)^-1]^T }_ij	
//		// multiply m_x^T by B_naxna and save the result to grad_E_temp
//		essqc::dgemm('T', 'N', m_no, m_na, m_na, -1.0, &m_x.at(0), m_na, B_naxna, m_na, 1.0, gradgradterm, m_no);	// grad grad term is [m_no, m_na]
//
//	} // mu for loop
//
//	// Returning the analytically calculated form of = 2 * sum_mu sum_m { ( d/dmu_m log(psi) ) . d/dc_ij ( d/dmu_m log(psi) ) }
//	essqc::daxpy(m_no * m_na, 2.0, gradgradterm, 1, grad_E, 1);	// [no,na] mat for alpha or beta contribution added to grad_E
//
//
//	// must return lap term as well and multiply by -1/2 to get full grad_E - currently hacked 
//
//}
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

	// get der1 and der2 -- TODO get from compute_ke_pieces instead of recalculating
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
			//std::cout << "mu = " << mu << ", i = " << i << std::endl;

			// i need just the determinants contribution to gradlogpsi which is also already calculated in compute_ke_pieces
			// (dX/dri C)_ithrow . XCi_ithcol
			const double d2dmu2_i_scalar = essqc::ddot(m_na, proda + i, m_na, &m_xci.at(i * m_na), 1);

  	      	// multiply the d/dc gradlogpsi no x na mat by the grad log psi element for that elec/mu's matrix
			if (m_e1s == 0)
				ddmu_i_scalar = grad_log_psi[3*i + mu];
			if (m_e1s == 1)
				ddmu_i_scalar = grad_log_psi[3*m_na + 3*i + mu];
			//std::cout << "ddmu_i_scalar = " << ddmu_i_scalar << std::endl;
			//std::cout << "ddv_gradlogpsi for elec: " << i << std::endl;	
			//std::cout << std::endl;
			//matprint(m_no, m_na, ddv_gradlogpsi[mu] + m_na*m_no * i, "%14.6e");
			//std::cout << std::endl;
			//std::cout << std::endl;

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

	//std::cout << "grad_E running total" << std::endl;	
	//std::cout << std::endl;
	//matprint(m_no, m_na, grad_E, "%14.6e");
	//std::cout << std::endl;
	//std::cout << std::endl;

}

// d/dc_ij (d^n/dmu^n log psi) = d^n/dmu^n(X)^T * XC^-1^T - X^T * XC^-1^T * C^T * d^n/dmu^n(X)^T * XC^-1^T	
// ddv_gradlogpsi [3,  na * no*na] = [xyz, [noxna col major ddc_ij ddmu_elec gradlogpsi ]_elec(alpha or beta) ... ]
// ddv_laplogpsi [no*na] = same as grad but summed over mu and elec
void essqc::qmc::ABDeterminant::compute_ddv_dnlogpsi(double ** ddv_gradlogpsi, double * ddv_laplogpsi, double ** der1, double ** der2, int indx)
{
	// smae math struct is is needed for ddv_gradlogpsi and ddv_laplogpsi
	//std::cout << " compute_ddv_dnlogpsi in wff_1_det " << __LINE__ << " nvp " << get_num_of_vp() << std::endl;

	const int lm_space_needed = 3 * m_na * m_no;
  	std::vector<double> lm_workspace(lm_space_needed, 0.0);	

	double * d1A_naxno = &lm_workspace.at(0 * m_na * m_no);
	double * d2A_naxno = &lm_workspace.at(1 * m_na * m_no);
	double * ddv_laplogpsi_temp = &lm_workspace.at(2 * m_na * m_no);

	/*std::cout << "XC^-1 in compute_ddv_dnlogpsi " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_na, m_na, &m_xci.at(0), "%14.6f");
	std::cout << std::endl;
  */

	// loop over x, y, and z
	for (int mu = 0; mu < 3; mu++)
	{
		/*std::cout << "*********** mu = " << mu << std::endl;

	  std::cout << "der1[mu]:" << std::endl;
	  for (int i = 0; i < m_na * m_no; i++)
	  {
	  	std::cout << der1[mu][i] << " ";
	  }
	  std::cout << std::endl;
    */

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
      /*std::cout << "d1A_naxno = outer product of dX/dmu ith row (length no) by XC^-1 ith col (length na) " << std::endl;
	    matprint(m_no, m_na, d1A_naxno, "%14.6e");
	    std::cout << std::endl;
      */

			// output ddv_gradlogpsi[mu] to ith position
			essqc::qmc::ABDeterminant::outerprod_to_ddv_dnlogpsi(d1A_naxno, ddv_gradlogpsi[mu] + m_na*m_no * i);
			// d2A_naxno = outer product of dX/dmu ith row (length no) by C ith col (length na)
			essqc::dger(m_no, m_na, 1.0, der2[mu] + i, m_na, &m_xci.at(i * m_na), 1, d2A_naxno, m_no);	// outer product CORRECT
      /*std::cout << "d2A_naxno = outer product of dX/dmu ith row (length no) by C ith col (length na)" << std::endl;
	    matprint(m_na, m_no, d2A_naxno, "%14.6e");
	    std::cout << std::endl;
      */

			// output to running total of ddv_laplogpsi 
			essqc::qmc::ABDeterminant::outerprod_to_ddv_dnlogpsi(d2A_naxno, ddv_laplogpsi_temp);
			essqc::daxpy(m_no*m_na, 1.0, ddv_laplogpsi_temp, 1, ddv_laplogpsi, 1);
      /*std::cout << "contribution to output to running total of ddv_laplogpsi " << std::endl;
	    matprint(m_na, m_no, ddv_laplogpsi_temp, "%14.6e");
	    std::cout << std::endl;
      */
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

 	//std::vector<double> C_temp(m_no * m_na, 0.0);	
	//double * C_temp_ptr = C_temp.data(); 

	/*std::cout << "ddv_gradlogpsi[mu] before = d1A - xT.xciT.cT.d1A:" << std::endl;
	matprint(m_no, m_na, ddv_dnlogpsi, "%14.6e");
	std::cout << std::endl;
  */

	// d/dc_ij (dn log psi) = dnX^T * XC^-1^T < adding first tern to contribut to grad log psi
	essqc::daxpy(m_na*m_no, 1.0, A_naxno, 1, ddv_dnlogpsi, 1);
  
	/*std::cout << "ddv_gradlogpsi[mu] = d1A " << std::endl;
	matprint(m_no, m_na, ddv_dnlogpsi, "%14.6e");
	std::cout << std::endl;
  */

  // GOOD
	// C^T * A_naxno: multiply the transpose of m_C by A_naxno and save the result to B_naxna
	essqc::dgemm('T', 'N', m_na, m_na, m_no, 1.0, &m_C.at(0), m_no, A_naxno, m_no, 0.0, B_naxna_ptr, m_na);	// B_naxna is [m_na, m_na]
	/*std::cout << "C^T @ A = B_naxna_ptr\n" << std::endl;	
	matprint(m_na, m_na, B_naxna_ptr, "%14.6e");
	std::cout << std::endl;
	std::cout << std::endl;
  */

  // GOOD - 02102025 - fix H bug by assigning output to B_temp instead of overwritting B
	// XC^-1^T * A_naxno: multiply transpose of m_xci by B_naxna_ptr and save the result to B_naxna
	essqc::dgemm('T', 'N', m_na, m_na, m_na, 1.0, &m_xci.at(0), m_na, B_naxna_ptr, m_na, 0.0, B_naxna_temp_ptr, m_na);	// B_naxna is [m_na, m_na]
	/*std::cout << "xci^T @ B = B_naxna_temp_ptr -- TKQ edited here to output to B_naxna_temp_ptr = \n" << std::endl;	
	matprint(m_na, m_na, B_naxna_temp_ptr, "%14.6e");
	std::cout << std::endl;
	std::cout << std::endl;
  */

	// X^T * A_naxno: multiply m_x^T by B_naxna_ptr and save -= to ddv_dnlogpsi = A_naxno 
	// d/dc_ij (dn log psi) = dnX^T * XC^-1^T - X^T * XC^-1^T * C^T * dnXi^T * XC^-1^T	--- [m_no, m_na] CM vector
	essqc::dgemm('T', 'N', m_no, m_na, m_na, -1.0, &m_x.at(0), m_na, B_naxna_temp_ptr, m_na, 1.0, ddv_dnlogpsi, m_no);	// ddv_gradlogpsi = A_naxno is [m_no, m_na]
	/*std::cout << "ddv_dnlogpsi -= x^T @ B_temp - complete for grad term\n" << std::endl;	
	matprint(m_no, m_na, ddv_dnlogpsi, "%14.6e");
	std::cout << std::endl;
	std::cout << std::endl;
  */

}



// compute LM and derivatives values to check derivatives against finite difference test
// ddv_gradlogpsi is the derivative of the gradient of the log psi wrt the variational parameters (3 x m_no x m_na)	for alpha or beta <-- pop in compute_ddv_dnlogpsi
void essqc::qmc::ABDeterminant::LM_deriv_test(double * const grad_log_psi, double ** ddv_gradlogpsi, double * ddv_laplogpsi, double ** der1, double ** der2, const double fd_delta){
	// ddv_gradlogpsi is [3, na * no*na]
	// der1 is [3, naxno]
	/*std::cout << std::endl;
	std::cout << "print Xmat in LM_deriv_test: " << std::endl;	
	matprint(m_na, m_no, &m_x.at(0), "%14.6f");
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "print Cmat in LM_deriv_test: " << std::endl;	
	matprint(m_no, m_na, &m_C.at(0), "%14.6f");
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "print der1[0] in LM_deriv_test: " << std::endl;
	matprint(m_na, m_no, der1[0], "%14.6f");
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "print der1[1] in LM_deriv_test: " << std::endl;
	matprint(m_na, m_no, der1[1], "%14.6f");
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "print der1[2] in LM_deriv_test: " << std::endl;
	matprint(m_na, m_no, der1[2], "%14.6f");
	std::cout << std::endl;
	*/
	
	essqc::qmc::ABDeterminant::compute_ddv_dnlogpsi(ddv_gradlogpsi, ddv_laplogpsi, der1, der2, 0);	// 0 is arbirary index
}

// compute orbital and derivative values to check derivatives against finite difference test
void essqc::qmc::ABDeterminant::orbs_and_derivs_test(double * orbs, double ** der1, double ** der2, const double fd_delta){
	// orbs points to alpha or beta position already
	/*std::cout << std::endl;
	std::cout << "print Xmat in orbs_derivs_test: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_na, m_no, &m_x.at(0), "%14.6f");
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "print Cmat in orbs_derivs_test: " << std::endl;	// CHECK IF INDEXATION HERE IS CORRECT
	matprint(m_no, m_na, &m_C.at(0), "%14.6f");
	std::cout << std::endl;
	*/

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

	/*std::cout << "Contents of orbs:" << std::endl;
	for (int i = 0; i < cols; ++i) {
	    std::cout << orbs[i] << " ";
	}
	std::cout << std::endl;

	std::cout << "Contents of der1:" << std::endl;
	for (int i = 0; i < rows; ++i) {
	    for (int j = 0; j < cols; ++j) {
	        std::cout << der1[i][j] << " ";
	    }
	    std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Contents of der2:" << std::endl;
	for (int i = 0; i < rows; ++i) {
	    for (int j = 0; j < cols; ++j) {
	        std::cout << der2[i][j] << " ";
	    }
	    std::cout << std::endl;
	}
	*/
}

// SKT's test for calculating grad E
void essqc::qmc::ABDeterminant::compute_grad_E_skt(double *const grad_E, double *const grad_log_psi, int indx)
{

	std::cout << " compute_grad_E_skt in wff_1_det " << __LINE__ << " nvp " << get_num_of_vp() << std::endl;
	std::vector<double> grad_E_temp(get_num_of_vp(), 0.0);
	// get room to store the first and second derivatives
	const int space_needed_2 = 9 * m_na * m_no + m_na * m_na;
	if (m_workspace_2.size() < space_needed_2)
		m_workspace_2.assign(space_needed_2, 0.0);

	// get pointers to where we will put orbital values and derivatives
	double * der1[3];
	double * der2[3];
	//double * xci_x[0];
	double * orbs = &m_workspace_2.at(0 * m_na * m_no);
	der1[0] = &m_workspace_2.at(1 * m_na * m_no); // x 1st derivatives of the atomic orbitals
	der1[1] = &m_workspace_2.at(2 * m_na * m_no); // y 1st derivatives of the atomic orbitals
	der1[2] = &m_workspace_2.at(3 * m_na * m_no); // z 1st derivatives of the atomic orbitals
	der2[0] = &m_workspace_2.at(4 * m_na * m_no); // x 2nd derivatives of the atomic orbitals
	der2[1] = &m_workspace_2.at(5 * m_na * m_no); // y 2nd derivatives of the atomic orbitals
	der2[2] = &m_workspace_2.at(6 * m_na * m_no); // z 2nd derivatives of the atomic orbitals
	double * xci_x = &m_workspace_2.at(7 * m_na * m_no); // z 2nd derivatives of the atomic orbitals

	// get pointers to where we will put products of matrices
	double *const xci_der_x = &m_workspace_2.at(8 * m_na * m_no);
	double *const der_X_C = &m_workspace_2.at(9 * m_na * m_no + 0 * m_na * m_na);

	// get orbital values and derivatives
	essqc::qmc::ABDeterminant::get_orbs_and_derivs(m_e1pos, m_nuc, orbs, der1, der2);

	std::cout << std::endl;
	std::cout << "der[0]: " << std::endl;	
	matprint(m_na, m_no, der1[0], "%14.6f");
	std::cout << std::endl;

	//////////////////////////////////////////////////////////////////////////////////////

        // First compute (XC)^-1 X (same for all electrons and positions)
	essqc::dgemm('N', 'N', m_na, m_no, m_na, 1.0, &m_xci.at(0), m_na, &m_x.at(0), m_na, 0.0, xci_x, m_na);
	std::cout << std::endl;
	std::cout << "xci_x: " << std::endl;	
	matprint(m_na, m_no, xci_x, "%14.6f");
	std::cout << std::endl;

	
	// loop over x, y, and z
	for (int mu = 0; mu < 3; mu++) {
 
		std::cout << "*********** mu = " << mu << std::endl;
	    // get matrices for grad grad term
            // (dX)C
	    essqc::dgemm('N', 'N', m_na, m_na, m_no, 1.0, der1[mu], m_na, &m_C.at(0), m_no, 0.0, der_X_C, m_na);
	    // (XC)^-1(dX)
	    //essqc::dgemm('N', 'N', m_na, m_no, m_na, 1.0, m_xci.at(0), m_na, der1[mu], m_na, 0.0, xci_der_x, m_na);
		std::cout << std::endl;
		std::cout << "der_X_C: " << std::endl;	
		matprint(m_na, m_na, der_X_C, "%14.6f");
		std::cout << std::endl;
            

	    // loop over electrons
	    for (int i = 0; i < m_na; i++) {
	                
			std::cout << "i = " << i << std::endl;
	        // Loop over rows of C matrix whose coeff we're optimizing
                for (int k = 0; k < m_no; k++) {

	            // Loop over columns of C matrix whose coeff we're optimizing
	            for (int l = 0; l < m_na; l++) {
		
			// calculate d/dcij gradlogpsi
			double xci_dot_d1_x = essqc::ddot(m_na, &m_xci.at(l), m_na, der1[mu] + k * m_na, 1);
			std::cout << std::endl;
			std::cout << "xci_dot_d1_x: " << xci_dot_d1_x  <<std::endl;	

			double d1xc_dot_xci_x = essqc::ddot(m_na, xci_x + m_na * k, 1, der_X_C + i, m_na); 
			std::cout << std::endl;
			std::cout << "d1xc_dot_xci_x: " << d1xc_dot_xci_x  <<std::endl;	

			double xci_li = m_xci[l + m_na * i]; 
			std::cout << std::endl;
			std::cout << "xci_li: " << xci_li  <<std::endl;	

	        double d_dc_gradlogpsi = xci_dot_d1_x - xci_li * d1xc_dot_xci_x;
			std::cout << std::endl;
			std::cout << "d_dc_gradlogpsi: " << d_dc_gradlogpsi  <<std::endl;	
			
			// if alpha spin determinant
	        	if (m_e1s == 0) {

			    // add contribution from this electron and this coordinate
		            grad_E_temp[k + m_no * l] -= d_dc_gradlogpsi * grad_log_psi[3 * i + mu];
	        	}
	        	
			// if beta spin determinant
	        	if (m_e1s == 1) {

			    // add contribution from this electron and this coordinate
		            grad_E_temp[k + m_no * l] -= d_dc_gradlogpsi * grad_log_psi[3 * i + 3 * m_na + mu];
	        	}
                        
		    
	            } // end loop over coulumns of C
		  

	         } // end loop over rows of C

	    } // end loop over electrons
	    
//	    // get matrices for grad nabla term
//            // (d2X)C
//	    essqc::dgemm('N', 'N', m_na, m_na, m_no, 1.0, der2[mu], m_na, &m_C.at(0), m_no, 0.0, der_X_C, m_na);
//	    // (XC)^-1(d2X)
//	    //essqc::dgemm('N', 'N', m_na, m_no, m_na, 1.0, m_xci.at(0), m_na, der2[mu], m_na, 0.0, xci_der_x, m_na);
//
//            // loop over electrons
//	    for (int i = 0; i < m_na; i++) {
//	                
//
//	        // Loop over rows of C matrix whose coeff we're optimizing
//                for (int k = 0; k < m_no; k++) {
//
//	            // Loop over columns of C matrix whose coeff we're optimizing
//	            for (int l = 0; l < m_na; l++) {
//		    
//			// calculate d/dcij nablalogpsi
//			double xci_dot_d2_x = essqc::ddot(m_na, &m_xci.at(l), m_na, der2[mu] + k * m_na, 1);
//			double d2xc_dot_xci_x = essqc::ddot(m_na, xci_x + m_na * k, 1, der_X_C + i, m_na); 
//			double xci_li = m_xci[l + m_na * i]; 
//	                double d_dc_nablalogpsi = xci_dot_d2_x - xci_li * d2xc_dot_xci_x;
//
//			// add contribution from this electron and this coordinate
//		        grad_E_temp[k + m_no * l] -= 0.5 * d_dc_nablalogpsi;
//	        	
//	            } // end loop over coulumns of C
//		  
//
//	         } // end loop over rows of C
//
//	    } // end loop over electrons
	} // end loop over coordinates

	for (int i = 0; i < m_na*m_no; i++) {
	    grad_E[i] = grad_E_temp[i];
	}

}
