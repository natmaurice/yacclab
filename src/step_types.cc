#include "step_types.h"


std::string Step(StepType n_step)
{
    switch (n_step) {
    case ALLOC_DEALLOC:
        return "Alloc Dealloc";
	break;
    case FIRST_SCAN:
        return "First Scan";
	break;
    case SECOND_SCAN:
        return "Second Scan";
	break;
    case RLE_SCAN:
	return "RLE";
	break;
    case UNIFICATION:
	return "Unification";
	break;
    case TRANSITIVE_CLOSURE:
	return "Transitive Closure";
	break;
    case RELABELING:
	return "Relabeling";
	break;
    case SETUP:
	return "Setup";
	break;
    case FEATURES:
	return "Features";
	break;
    case REDUCTION:
	return "Reduction";
	break;
    case ALL_SCANS:
        return "All Scans";
	break;
    case ST_SIZE: // To avoid warning on AppleClang
        break;
    }

    return "";
}

