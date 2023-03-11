// Copyright (c) 2022, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "labeling3D_PRED_2021.h"

template <typename LabelsSolver>
using PRED_3D_CCA = PRED_3D<LabelsSolver, ConfFeatures3DAll>;

REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(PRED_3D);
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(PRED_3D_CCA);
