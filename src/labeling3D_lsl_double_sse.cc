#ifdef USE_LSL_SIMD_SSE

#include "labeling3D_lsl_generic.hpp"
#include <lsl3dlib/conf.hpp>

#include <lsl3dlib/lsl3d/unification_double.hpp>
#include <simdhelpers/defs.hpp> // Defines __HELPER_AVX256__


#if defined(__SSE4_2__)
#include <lsl3dlib/rle/rle-sse.hpp>
#include <lsl3dlib/lsl3d/relabeling-sse.hpp>


template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_SSE4 =
    LSL3D<LabelsSolver, 
	  rle::sse::STDZ_V2, unify::Unify_SM_Double,
	  algo::sse::Relabeling_Z, FeatureComputation_None>;


template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_PL_SSE4 =
    LSL3D<LabelsSolver, 
	  rle::sse::STDZ_V2, unify::Unify_SM_Double_PL,
	  algo::sse::Relabeling_Z, FeatureComputation_None>;

template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_PL_SSE4_CCA =
    LSL3D<LabelsSolver, 
	  rle::sse::STDZ_V2, unify::Unify_SM_Double_PL,
	  algo::sse::Relabeling_Z, FeatureComputation, ConfFeatures3DAll>;


template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_PL_SSE4_CCA_only =
    LSL3D<LabelsSolver, 
	  rle::sse::STDZ_V2, unify::Unify_SM_Double_PL,
	  algo::Relabeling_Nothing, FeatureComputation, ConfFeatures3DAll>;

#ifdef USE_LSL3D_CCA_OTF
template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_PL_SSE4_CCA_OTF =
    LSL3D<LabelsSolver, 
	  rle::sse::STDZ_V2, unify::Unify_SM_Double_PL,
	  algo::sse::Relabeling_Z, FeatureComputation_OTF, ConfFeatures3DAll>;

template<typename LabelsSolver>
using LSL3D_FSM_DOUBLE_PL_SSE4_CCA_OTF_only =
    LSL3D<LabelsSolver, 
	  rle::sse::STDZ_V2, unify::Unify_SM_Double_PL,
	  algo::Relabeling_Nothing, FeatureComputation_OTF, ConfFeatures3DAll>;

#endif // USE_LSL3D_CCA_OTF


REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE_SSE4)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE_PL_SSE4)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE_PL_SSE4_CCA)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE_PL_SSE4_CCA_only)

#ifdef USE_LSL3D_CCA_OTF
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_DOUBLE_PL_SSE4_CCA_OTF)
#endif // USE_LSL3D_CCA_OTF

#endif // __SSE4_2__



#endif // USE_LSL_SIMD_SSE
