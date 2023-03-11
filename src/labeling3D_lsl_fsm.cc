// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "labeling3D_lsl_generic.hpp"
#include <lsl3dlib/conf.hpp>

#include <lsl3dlib/lsl3d/unification_merge.hpp>
#include <simdhelpers/defs.hpp>



template <typename LabelsSolver>
using LSL3D_ONLY_RLE =
    LSL3D<LabelsSolver, 
	  rle::STDZ, unify::Unify_Nothing,
	  algo::Relabeling_Nothing, FeatureComputation_None>;

template <typename LabelsSolver>
using LSL3D_FSM =
    LSL3D<LabelsSolver, 
    rle::STDZ, unify::Unify_SM_Separate, algo::Relabeling_Z_Border, FeatureComputation_None>;


template <typename LabelsSolver>
using LSL3D_FSM_CCA =
    LSL3D<LabelsSolver, 
    rle::STDZ, unify::Unify_SM_Separate, algo::Relabeling_Z_Border, FeatureComputation,
	  ConfFeatures3DAll>;

template <typename LabelsSolver>
using LSL3D_FSM_CCA_only =
    LSL3D<LabelsSolver, 
	  rle::STDZ, unify::Unify_SM_Separate, algo::Relabeling_Nothing, FeatureComputation,
	  ConfFeatures3DAll>;


#ifdef USE_LSL3D_CCA_OTF
template <typename LabelsSolver>
using LSL3D_FSM_CCA_OTF =
    LSL3D<LabelsSolver, 
    rle::STDZ, unify::Unify_SM_Separate, algo::Relabeling_Z_Border, FeatureComputation_OTF,
	  ConfFeatures3DAll>;
#endif // USE_LSL3D_CCA_OTF


template <typename LabelsSolver>
using LSL3D_FSM_V2 =
    LSL3D<LabelsSolver, 
	  rle::STDZ, unify::Unify_SM_Separate,
	  algo::Relabeling_Z_Border, FeatureComputation_None>;

template <typename LabelsSolver>
using LSL3D_FSM_V3 =
    LSL3D<LabelsSolver, 
	  rle::STDZ, unify::Unify_SM_Separate,
	  algo::Relabeling_Z_Border_V2, FeatureComputation_None>;




REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_CCA)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_CCA_only)

#ifdef USE_LSL3D_CCA_OTF
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_FSM_CCA_OTF)
#endif // USE_LSL3D_CCA_OTF
