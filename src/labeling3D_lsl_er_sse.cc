// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <simdhelpers/defs.hpp>


#include "labeling3D_lsl_generic.hpp"

#include <lsl3dlib/lsl3d/unification_er.hpp>

#if defined(__SSE4_2__) && defined(USE_LSL_SIMD_SSE)
#include <lsl3dlib/rle/rle-sse.hpp>
#include <lsl3dlib/lsl3d/relabeling-sse.hpp>


template <typename LabelsSolver>
using LSL3D_ER_SSE4 =
    LSL3D<LabelsSolver, rle::sse::STDZ_ER, unify::Unify_ER,
	  algo::sse::Relabeling_Z, FeatureComputation_None>;

template <typename LabelsSolver>
using LSL3D_ER_SSE4_CCA =
    LSL3D<LabelsSolver, rle::sse::STDZ_ER, unify::Unify_ER,
	  algo::sse::Relabeling_Z, FeatureComputation, ConfFeatures3DAll>;

template <typename LabelsSolver>
using LSL3D_ER_SSE4_CCA_only =
    LSL3D<LabelsSolver, rle::sse::STDZ_ER, unify::Unify_ER,
	  algo::Relabeling_Nothing, FeatureComputation, ConfFeatures3DAll>;


#ifdef USE_LSL3D_CCA_OTF
template <typename LabelsSolver>
using LSL3D_ER_SSE4_CCA_OTF =
    LSL3D<LabelsSolver, rle::sse::STDZ_ER, unify::Unify_ER,
	  algo::sse::Relabeling_Z, FeatureComputation_OTF, ConfFeatures3DAll>;

template <typename LabelsSolver>
using LSL3D_ER_SSE4_CCA_OTF_only =
    LSL3D<LabelsSolver, rle::sse::STDZ_ER, unify::Unify_ER,
	  algo::Relabeling_Nothing, FeatureComputation_OTF, ConfFeatures3DAll>;
#endif // USE_LSL3D_CCA_OTF


REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_ER_SSE4)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_ER_SSE4_CCA)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_ER_SSE4_CCA_only)

#ifdef USE_LSL3D_CCA_OTF
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_ER_SSE4_CCA_OTF)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LSL3D_ER_SSE4_CCA_OTF_only)
#endif // USE_LSL3D_CCA_OTF

#endif // __SSE4_2__ && USE_LSL3D_SIMD_SSE

