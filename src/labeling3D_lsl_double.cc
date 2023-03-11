#include "labeling3D_lsl_generic.hpp"
#include <lsl3dlib/conf.hpp>

#include <lsl3dlib/lsl3d/unification_double.hpp>
#include <simdhelpers/defs.hpp> // Defines __HELPER_AVX256__



template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE =
    LSL3D<LabelsSolver, 
	  rle::STDZ, unify::Unify_SM_Double,
	  algo::Relabeling_Z_Border, FeatureComputation_None>;


template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_PL =
    LSL3D<LabelsSolver, 
	  rle::STDZ, unify::Unify_SM_Double_PL,
	  algo::Relabeling_Z_Border, FeatureComputation_None>;



template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_PL_CCA =
    LSL3D<LabelsSolver, 
	  rle::STDZ, unify::Unify_SM_Double_PL,
	  algo::Relabeling_Z_Border, FeatureComputation, ConfFeatures3DAll>;

#ifdef USE_LSL3D_CCA_OTF
template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_PL_CCA_OTF =
    LSL3D<LabelsSolver, 
	  rle::STDZ, unify::Unify_SM_Double_PL,
	  algo::Relabeling_Z_Border, FeatureComputation_OTF, ConfFeatures3DAll>;
#endif // USE_LSL3D_CCA_OTF

template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_PL_CCA_Line =
    LSL3D<LabelsSolver, 
	  rle::STDZ, unify::Unify_SM_Double_PL,
	  algo::Relabeling_Z_Border, FeatureComputation_Line, ConfFeatures3DAll>;

template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_PL_CCA_only =
    LSL3D<LabelsSolver, 
	  rle::STDZ, unify::Unify_SM_Double_PL,
	  algo::Relabeling_Nothing, FeatureComputation, ConfFeatures3DAll>;



REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE_PL)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE_PL_CCA)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE_PL_CCA_only)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE_PL_CCA_Line)

#ifdef USE_LSL3D_CCA_OTF
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE_PL_CCA_OTF)
#endif // USE_LSL3D_CCA_OTF
