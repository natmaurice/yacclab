// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <simdhelpers/defs.hpp>

#include "labeling3D_lsl_generic.hpp"


#if  defined(__ARM_NEON) && defined(USE_LSL_SIMD_NEON)
#include <lsl3dlib/rle/rle-neon.hpp>
#include <lsl3dlib/lsl3d/relabeling-neon.hpp>


template <typename LabelsSolver>
using LSL3D_ER_NEON =
    LSL3D<LabelsSolver, rle::neon::STDZ_ER_V2, unify::Unify_ER,
	  algo::neon::Relabeling_Z, FeatureComputation_None>;


template <typename LabelsSolver>
using LSL3D_ER_NEON_CCA =
    LSL3D<LabelsSolver, rle::neon::STDZ_ER_V2, unify::Unify_ER,
	  algo::neon::Relabeling_Z, FeatureComputation, ConfFeatures3DAll>;


template <typename LabelsSolver>
using LSL3D_ER_NEON_CCA_only =
    LSL3D<LabelsSolver, rle::neon::STDZ_ER_V2, unify::Unify_ER,
	  algo::Relabeling_Nothing, FeatureComputation, ConfFeatures3DAll>;


REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_ER_NEON)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_ER_NEON_CCA)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_ER_NEON_CCA_only)

#endif // __ARM_NEON && USE_LSL_SIMD_NEON
