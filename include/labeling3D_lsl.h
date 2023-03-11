// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_LSL3D_H_
#define YACCLAB_LABELING_LSL3D_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

#include "unionfind.hpp"
#include "table.hpp"
#include "bindings.hpp"
#include "algos/3d/lsl.hpp"
#include "algos/utils.hpp"
#include "algos/3d/relabeling.hpp"

template <typename LabelsSolver>
class LSL3D : public Labeling3D<Connectivity3D::CONN_26> {
public:
    LSL3D() {}

    void PerformLabeling() {
	Alloc();
	FirstScan();
	SecondScan();
	Dealloc();
    }
    
    void PerformLabelingWithSteps() {
	double alloc_timing = Alloc();

	int32_t depth = img_.size.p[0];
	int32_t height = img_.size.p[1];
	int32_t width = img_.size.p[2];
	int32_t size = depth * height * width;
	
	perf_.start();
#if defined(__i386__) || defined(__x86_64__)
	dcache_sampler.Start();
	cycles_sampler.Start();
	icache_sampler.Start();
	branch_sampler.Start();
#endif // defined(__i386__) || defined(__x86_64__)

	FirstScan();
	
#if defined(__i386__) || defined(__x86_64__)
	dcache_sampler.Stop();
	cycles_sampler.Stop();
	icache_sampler.Stop();
	branch_sampler.Stop();
#endif // defined(__i386__) || defined(__x86_64__)

	
	perf_.stop();
	perf_.store(Step(StepType::FIRST_SCAN), perf_.last());
#if defined(__i386__) || defined(__x86_64__)
	dcache_sampler.Store(StepType::FIRST_SCAN, dcache_sampler.Last());
	cycles_sampler.Store(StepType::FIRST_SCAN, cycles_sampler.Last() / size);
	icache_sampler.Store(StepType::FIRST_SCAN, icache_sampler.Last());
	branch_sampler.Store(StepType::FIRST_SCAN, branch_sampler.Last());
#endif // defined(__i386__) || defined(__x86_64__)

	
	perf_.start();
#if defined(__i386__) || defined(__x86_64__)
	dcache_sampler.Start();
	cycles_sampler.Start();
	icache_sampler.Start();
	branch_sampler.Start();
#endif // defined(__i386__) || defined(__x86_64__)
	
	SecondScan();

#if defined(__i386__) || defined(__x86_64__)
	dcache_sampler.Stop();
	cycles_sampler.Stop();
	icache_sampler.Stop();
	branch_sampler.Stop();
#endif // defined(__i386__) || defined(__x86_64__)
     
	perf_.stop();
	perf_.store(Step(StepType::SECOND_SCAN), perf_.last());
#if defined(__i386__) || defined(__x86_64__)
	dcache_sampler.Store(StepType::SECOND_SCAN, dcache_sampler.Last());
	cycles_sampler.Store(StepType::SECOND_SCAN, cycles_sampler.Last() / size);
	icache_sampler.Store(StepType::SECOND_SCAN, icache_sampler.Last());
	branch_sampler.Store(StepType::SECOND_SCAN, branch_sampler.Last());
#endif // defined(__i386__) || defined(__x86_64__)
	
	perf_.start();
	Dealloc();
	perf_.stop();
	perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing);
    }

private:    
    UnionFind<int32_t> uf;
    Table<int32_t> labels;
    Table<uint8_t> bitmap;
    Stats stats;
    ERATable ERA;
    RLCTable rlc;
    int32_t nea = 0;    
    int16_t *ER, *ER0, *ER1;
    
    
    double Alloc() {

	long width = img_.size.p[2];
	long height = img_.size.p[1];
	long depth = img_.size.p[0];
	long size = width * height * depth;
	
	//double ls_t = LabelsSolver::Alloc(UPPER_BOUND_26_CONNECTIVITY, perf_);
	double ls_t = 0;
	// Memory allocation for the output image
	perf_.start();

	uf.Resize(size);
	rlc.Reserve(width, height, depth);
	ERA.Reserve(width / 2 + 1, height, depth);	

	int32_t er_slice_pitch = width * (height + 1);
	int16_t er_row_pitch = width;
	
        ER = new int16_t[2 * er_slice_pitch + er_row_pitch];
	ER0 = ER;
	ER1 = ER0 + er_slice_pitch;
	
	std::fill(ER, ER0 + 2 * er_slice_pitch + er_row_pitch, 0);
	uf.Touch();
	ERA.Touch();
	rlc.Touch();
	
	stats.label_count = 0;	
	nea = 0;
	
	// Memory allocation of the labels solver
	img_labels_.create(3, img_.size.p, CV_32SC1);
	memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);

	perf_.stop();

	
	double t = perf_.last();
	perf_.start();
	memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);

	// ER1 + ER0
	std::fill(ER, ER0 + 2 * er_slice_pitch + er_row_pitch, 0);
	uf.Touch();
	ERA.Touch();
	rlc.Touch();
	
	labels.Wrap(reinterpret_cast<int32_t*>(img_labels_.data), width, height, depth);
	bitmap.Wrap(img_.data, width, height, depth);

	perf_.stop();	
	double ma_t = t - perf_.last();
	// Return total time
	return ls_t + ma_t;
    }
    
    
    void Dealloc() {
	rlc.Clear();
	ERA.Clear();
	labels.Reset();
	delete[] ER;
	ER = nullptr;
	ER0 = nullptr;
	ER1 = nullptr;	
    }
    
    void FirstScan() {
	algo::lsl_3d_first_pass(bitmap, ER0, ER1, rlc, ERA, uf, nea);
    }

    void SecondScan() {
	equivalence_resolution_pack(uf);
	algo::relabeling(labels, ERA, rlc, uf);
    }
};

#endif // !YACCLAB_LABELING_LSL_3D_H_
