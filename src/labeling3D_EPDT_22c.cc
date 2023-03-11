// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "labeling3D_EPDT_22c.h"


template <typename LabelsSolver>
using EPDT_3D_22c_CCA = EPDT_3D_22c<LabelsSolver, true, ConfFeatures3DAll>;


template <typename LabelsSolver>
using EPDT_3D_22c_CCA_only = EPDT_3D_22c<LabelsSolver, false, ConfFeatures3DAll>;


REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(EPDT_3D_22c);
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(EPDT_3D_22c_CCA);
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(EPDT_3D_22c_CCA_only);
