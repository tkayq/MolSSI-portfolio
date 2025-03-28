#include "../threading.h"
#include "accumulator_Hmat.hpp"

// ONLY USE WITH SIMPLE_SAMPLE
void essqc::qmc::AccumulatorHmat::accumulate_detail(const essqc::qmc::AccDataPack & dp) {

}

// for batching
void essqc::qmc::AccumulatorHmat::accumulate_detail(const essqc::qmc::AccDataPack & dp, std::vector<int> & active_e, std::vector<int> & active_n) {
   const int nvp = dp.PvP0.size();

   double m_le_cur_value = m_acc_elocal->get_samp_le();
   std::vector<double> H_PvP0 = dp.grad_E;

   for ( int l = 0; l < nvp; l++ ) {
   	H_PvP0[l] += m_le_cur_value * dp.PvP0[l]; 
   }

   // nvp x nvp outer product stored in column major order
   for (int i = 0; i < nvp+1; i++) {
       for (int j = 0; j < nvp+1; j++) {
	  if (i == 0 && j == 0) {
              m_Hmat_sum[0] += m_le_cur_value;  // Padding for the top-left corner
          } else if (i == 0) {
              m_Hmat_sum[j * (nvp+1)] += H_PvP0[j - 1];  // Pad the first row with vector values
          } else if (j == 0) {
              m_Hmat_sum[i] += m_le_cur_value * dp.PvP0[i - 1];  // Pad the first column with vector values
          } else {
              m_Hmat_sum[j * (nvp+1) + i] += dp.PvP0[i - 1] * H_PvP0[j - 1];  // Calculate the outer product
	  }
       }
   }

}


void essqc::qmc::AccumulatorHmat::finalize_block() {
    const int tid = omp_get_thread_num();
    double * const store_start = m_accumulation_data_ptr + ( tid * m_nbpt + m_bc ) * m_mat_dim * m_mat_dim;
    for (int i = 0; i < m_mat_dim * m_mat_dim; i++) {	
	store_start[i] += m_Hmat_sum[i]/double(m_spb);
        //m_accumulation_data_ptr[(tid * m_nbpt + m_bc) * m_mat_dim * m_mat_dim + i] += m_PvP0_sum[i]/double(m_spb); 
    }
    //std::cout << "setting m_Hmat_sum to 0 in finalize_block" << std::endl;
    std::fill(m_Hmat_sum.begin(), m_Hmat_sum.end(), 0.0);

}

void essqc::qmc::AccumulatorHmat::get_numpy_array_dim(const int nblock, const int nnuc, const int max_dim, double * const dim_ptr) const {
//    std::cout << "\t\t IN AccumulatorHmat get_numpy_array_dim()" << std::endl;
    if ( max_dim < 1 )
        throw essqc::Exception("Problem in AccumulatorHmat::get_numpy_array_dim:  the AccumulatorHmat storage array needs at least one dimension but max_dim was %i") % max_dim;
    dim_ptr[0] = nblock;
    dim_ptr[1] = m_mat_dim * m_mat_dim;
    for (int i = 2; i < max_dim; i++)
        dim_ptr[i] = 0;
}

// function to create a new accumulator of this type
std::shared_ptr<essqc::qmc::AccumulatorBase> essqc::qmc::AccumulatorHmat::factory_func(const int nb, const int spb, const int nelec, const int nnuc, py::dict & acc_dict,
                                                                                         const std::vector<std::shared_ptr<AccumulatorBase> > & existing_accumulators) const {

    // get pointers to the local energy accumulator
    const AccumulatorLocalE * const ptr_le = essqc::qmc::get_existing_accumulator<AccumulatorLocalE>(existing_accumulators);

    return std::shared_ptr<AccumulatorBase>( new AccumulatorHmat(nb, spb, ptr_le, acc_dict) );

}

