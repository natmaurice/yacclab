#ifndef YACCLAB_SAMPLERS_HPP
#define YACCLAB_SAMPLERS_HPP

#include <lsl3dlib/papi_helper.hpp>
#include <lsl3dlib/sampler.hpp>
#include "step_types.h"
#include "performance_evaluator.h"

#include <opencv2/core.hpp>

#include <simdhelpers/defs.hpp>

#include <lsl3dlib/perf-helper.hpp>

#ifndef YACCLAB_USE_PERF
#define YACCLAB_USE_PERF false
#endif // YACCLAB_USE_PERF


struct SamplingEvents {

#if YACCLAB_USE_PERF && YACCLAB_USE_PAPI
#error "Can not use perf and papi at the same time"
#endif // USE_PERF && USE_PAPI
    
#if YACCLAB_USE_PERF
    static constexpr int CYCLES = PERF_COUNT_HW_INSTRUCTIONS;
    static constexpr int BRANCH_MISSES = PERF_COUNT_HW_BRANCH_MISSES;
    static constexpr int BRANCH_INSTR = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
    static constexpr int CACHE_L1D = PERF_COUNT_HW_CACHE_L1D; // Does not work alone (requires type == PERF_TYPE_HW_CACHE)
    static constexpr int CACHE_L1I = PERF_COUNT_HW_CACHE_L1I; // Does not work alone
    static constexpr int CACHE_L2D = -1; // Does not work alone
    static constexpr int CACHE_L3 = PERF_COUNT_HW_CACHE_LL; // Does not work alone
    static constexpr int CYCLE_NIIS = -1;
    static constexpr int TOTAL_INSTR = -1;
    static constexpr int STALLED_CYC = -1;
    static constexpr int WSTALLED_CYC = -1;
#endif // YACCLAB_USE_PERF

#if YACCLAB_USE_PAPI
    static constexpr int CYCLES = PAPIH_TCYC;
    static constexpr int BRANCH_MISSES = PAPIH_MISP;
    static constexpr int BRANCH_INSTR = PAPIH_TBR;
    static constexpr int CACHE_L1D = PAPIH_L1DM;
    static constexpr int CACHE_L1I = PAPIH_L1IM;
    static constexpr int CACHE_L2D = PAPIH_L2DM;
    static constexpr int CACHE_L3 = PAPIH_L3DM;
    static constexpr int CYCLE_NIIS = PAPIH_NIIS;
    static constexpr int TOTAL_INSTR = PAPIH_TINS;
    static constexpr int STALLED_CYC = PAPIH_STAL;
    static constexpr int WSTALLED_CYC = PAPIH_WSTAL;
#endif // YACCLAB_USE_PAPI

#if !YACCLAB_USE_PERF && !YACCLAB_USE_PAPI
    static constexpr int CYCLES = 0;
    static constexpr int BRANCH_MISSES = 0;
    static constexpr int BRANCH_INSTR = 0;
    static constexpr int CACHE_L1D = 0;
    static constexpr int CACHE_L1I = 0;
    static constexpr int CACHE_L2D = 0;
    static constexpr int CACHE_L3 = 0;
    static constexpr int CYCLE_NIIS = 0;
    static constexpr int TOTAL_INSTR = 0;
    static constexpr int STALLED_CYC = 0;
    static constexpr int WSTALLED_CYC = 0;
#endif // neither
};

struct Samplers {


#if YACCLAB_USE_PERF
    char _start_events_raw[PERFFORMAT_MAX_LENGTH];
    char _stop_events_raw[PERFFORMAT_MAX_LENGTH];
    long long diff_events[PERF_HELPER_MAX_EVENTS];    
    
    struct read_format* fmt_start = (struct read_format*)_start_events_raw;
    struct read_format* fmt_end = (struct read_format*)_stop_events_raw;
#endif // YACCLAB_USE_PERF
    
#if YACCLAB_USE_PAPI
    SamplerCycles cycles;
    
    long long start_measurements[MAX_PAPI_EVENTS];
    long long stop_measurements[MAX_PAPI_EVENTS];
    long long diff_measurements[MAX_PAPI_EVENTS];    
#endif // __HELPER_X86_ANY__

    cv::Mat1d res;    
    int res_size = 0;
    
    Samplers() {

#if YACCLAB_USE_PERF

	res_size = PERF_HELPER_MAX_EVENTS;
	//perf_helper_create();
	
#endif // YACCLAB_USE_PERF

		
#if YACCLAB_USE_PAPI
	res_size = MAX_PAPI_EVENTS;
#endif // __HELPER_X86_ANY__
	
        res.create(StepType::ST_SIZE, res_size);
	res.setTo(0);
    }

    void Reset() {
	res.setTo(0);
    }
    
    inline void Start() {

#if YACCLAB_USE_PERF
	for (size_t i = 0; i < PERF_HELPER_MAX_EVENTS; i++) {
	    diff_events[i] = 0;
	}
	read_counters(_start_events_raw, PERFFORMAT_MAX_LENGTH);
#endif // YACCLAB_USE_PERF
	
#if YACCLAB_USE_PAPI
	for (int i = 0; i < MAX_PAPI_EVENTS; i++) {
	    start_measurements[i] = 0;
	    stop_measurements[i] = 0;
	    diff_measurements[i] = 0;
	}
	read_counters(start_measurements);
#endif // YACCLAB_USE_PAPI

	
    }
    
    
    inline void Stop() {

#if YACCLAB_USE_PERF
	read_counters(_stop_events_raw, PERFFORMAT_MAX_LENGTH);
#endif // YACCLAB_USE_PERF
	
#if YACCLAB_USE_PAPI
	cycles.Stop();
	read_counters(stop_measurements);
#endif // YACCLAB_USE_PAPI

       
	
    }

    
    inline void Store(StepType type, size_t size) {

#if YACCLAB_USE_PERF
	for (size_t i = 0; i < fmt_end->nr; i++) {
	    int eventid = fmt_end->values[i].id;
	    int id = perf_helper_find_config(eventid);	    
	    if (id != -1) {
		long long value = (fmt_end->values[i].value - fmt_start->values[i].value);
		res(type, id) = value / static_cast<double>(size);
	    }
	}
#endif // YACCLAB_USE_PERF
	
#if YACCLAB_USE_PAPI
	//assert(res(StepType::ALL_SCANS, PAPIH_MISP) == 0);

	for (size_t i = 0; i < MAX_PAPI_EVENTS; i++) {
	    diff_measurements[i] = stop_measurements[i] - start_measurements[i];
	    res(type, i) = static_cast<double>(diff_measurements[i]) / static_cast<double>(size);
	}

#endif // YACCLAB_USE_PAPI
	
    }

    double Get(StepType step, int metric) const {

#if YACCLAB_USE_PERF
	metric = perf_find(metric);
#endif // YACCLAB_USE_PERF
	
	if (metric < 0) {
	    return 0;
	}	
	return res(step, metric);
    }
    
    inline void CalcDerived() {

	for (int i = 0; i < res_size; i++) {

	    double first_scan = res(StepType::FIRST_SCAN, i);
	    if (first_scan == 0.0) {
		first_scan = res(StepType::RLE_SCAN, i)
		    + res(StepType::UNIFICATION, i)
		    + res(StepType::SETUP, i);
	    }

	    double second_scan = res(StepType::SECOND_SCAN, i);
	    if (second_scan == 0.0) {		
		second_scan = res(StepType::RELABELING, i)
		    + res(StepType::TRANSITIVE_CLOSURE, i);
	    }
	    
	    if (res(StepType::ALL_SCANS, i) == 0.0) {
		res(StepType::ALL_SCANS, i) = first_scan + second_scan + res(StepType::FEATURES, i);
	    }
	}
   }
    
};

#endif // YACCLAB_SAMPLERS_HPP
