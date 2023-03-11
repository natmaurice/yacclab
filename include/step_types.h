#ifndef YACCLAB_STEP_TYPES_H_
#define YACCLAB_STEP_TYPES_H_

#include <string>

enum StepType {
    ALLOC_DEALLOC = 0,
    FIRST_SCAN = 1,
    SECOND_SCAN = 2,
    RLE_SCAN = 3,
    UNIFICATION = 4,
    TRANSITIVE_CLOSURE = 5,
    RELABELING = 6,
    SETUP = 7,
    FEATURES = 8,
    REDUCTION = 9,
    ALL_SCANS = 10,
    ST_SIZE = 11,
};

std::string Step(StepType n_step);

#endif // YACCLAB_STEP_TYPES_H_
