// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "labeling3D_he_2011.h"


template <typename LabelsSolver>
using LEB_3D_CCA = LEB_3D<LabelsSolver, true, ConfFeatures3DAll>;

template <typename LabelsSolver>
using LEB_3D_CCA_only = LEB_3D<LabelsSolver, false, ConfFeatures3DAll>;

REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LEB_3D);
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LEB_3D_CCA);
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(LEB_3D_CCA_only);
