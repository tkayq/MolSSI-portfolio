#ifndef ESSQC_VMC_ACCUMULATOR_HMAT
#define ESSQC_VMC_ACCUMULATOR_HMAT

#include "accumulator_base.hpp"
#include "accumulator_local_e.hpp"

namespace essqc {

namespace qmc {

    class AccumulatorHmat : public essqc::qmc::AccumulatorBase {

        // memeber data
        protected:

	// running sum of overlap matrix elements
	std::vector<double> m_Hmat_sum;

	// total number of variational parameters + 1 for LM
	int m_mat_dim;

	// reference to the local energy of given sample
	const AccumulatorLocalE * const m_acc_elocal;

        // public member functions
        public:

            // retuns the name for this type of accumulator
            static std::string name() { return "AccumulatorHmat"; }

            // constructor
            AccumulatorHmat(const int nb,
                            const int spb,
			    const AccumulatorLocalE * const acc_elocal,
                            py::dict & acc_dict)
            : AccumulatorBase(nb, spb, acc_dict, name()),
	      m_acc_elocal(acc_elocal)
            {
    		if ( essqc::qmc::AccumulatorBase::has_dict_access() ) {
		    np::ndarray a = py::extract<np::ndarray>(acc_dict[name()]);
		    m_mat_dim = a.shape(1);
		    m_Hmat_sum.assign(m_mat_dim*m_mat_dim, 0.0);
		    //std::cout << " [nvp in accumulator_Hmat " << m_mat_dim << "] "; 
		}
	    }

            // destructor
            ~AccumulatorHmat() {}

            // retuns the name for this type of accumulator
            std::string get_name() const { return this->name(); }

            void accumulate_detail(const essqc::qmc::AccDataPack & dp);
            void accumulate_detail(const essqc::qmc::AccDataPack & dp, std::vector<int> & active_e, std::vector<int> & active_n);

            // derived accumulator does whatever it needs to do at the end of a sampling block
            void finalize_block();

            // derived class tells the dimensions that numpy should use when setting up the array to hold the accumulator's accumulated data
            void get_numpy_array_dim(const int nblock, const int nnuc, const int max_dim, double * const dim_ptr) const;

            std::shared_ptr<AccumulatorBase> factory_func(const int nb, const int spb, const int nelec, const int nnuc, py::dict & acc_dict,
                                                          const std::vector<std::shared_ptr<AccumulatorBase> > & existing_accumulators) const; // {
            //    return std::shared_ptr<AccumulatorBase>( new AccumulatorHmat(nb, spb, m_acc_elocal, acc_dict) );
            //}

    };	// class AccumulatorHmat

}	// end qmc namespace

}	// end essqc namespace


#endif
