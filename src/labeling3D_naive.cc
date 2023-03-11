// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "labeling3D_naive.h"

template <typename LabelsSolver>
using naive_3D_CCA = naive_3D<LabelsSolver, ConfFeatures3DAll>;

REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(naive_3D)
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(naive_3D_CCA)
