// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "labeling3D_EPDT_26c.h"

template <typename LabelsSolver>
using EPDT_3D_26c_CCA = EPDT_3D_26c<LabelsSolver, ConfFeatures3DAll>;


REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(EPDT_3D_26c);
REGISTER_LABELING_WITH_EQUIVALENCES_SOLVERS(EPDT_3D_26c_CCA);
