// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_ROSENFELD_2X2_H_
#define YACCLAB_LABELING_ROSENFELD_2X2_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

#include "unionfind.hpp"
#include "table.hpp"
#include "bindings.hpp"
#include "algos/rosenfeld.hpp"
#include "algos/utils.hpp"
#include "ccl_stats.hpp"


template <typename LabelsSolver>
class ROSENFELD_2X2 : public Labeling2D<Connectivity2D::CONN_8> {
public:
    ROSENFELD_2X2() {}

    void PerformLabeling() {
	Alloc();
	FirstScan();
	SecondScan();
	Apply();
	Dealloc();
    }
    
    void PerformLabelingWithSteps() {
	double alloc_timing = Alloc();

	int32_t height = img_.size.p[1];
	int32_t width = img_.size.p[2];
	int32_t size = width * height;
	
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
    Table<int32_t> parent;
    Table<int32_t> labels;
    Table<uint8_t> bitmap;
    Stats stats;
    cv::Mat1i parent_labels;

    
    double Alloc() {

        int width = img_.size.p[2];
	int height = img_.size.p[1];
	long size = width * height;
	uf.Clear();
	uf.Resize(size);
       
	stats.label_count = 0;	

	
	// Memory allocation of the labels solver
	//double ls_t = LabelsSolver::Alloc(UPPER_BOUND_26_CONNECTIVITY, perf_);
	double ls_t = 0;
	// Memory allocation for the output image
	perf_.start();

	// Memory allocation of the labels solver
	// Allocate extra space to allow for an extra SSE4 write at the end. This may happen during
	// the absolute labelling part. We could simply allocate more memory by simply adding extra
	// space at the end of each line but the method used here uses less memory.
	std::vector<cv::Range> rect;
	rect.push_back(cv::Range(0, height));
	rect.push_back(cv::Range(0, width));

	
	int parent_size[2] = {
	    height + 2, width + 2
	};
	
	parent_labels.create(3, parent_size);
	img_labels_ = parent_labels(rect);

	
	memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
	perf_.stop();
	double t = perf_.last();
	perf_.start();
	memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
	perf_.stop();

	bitmap.Resize(width, height);
	ImageToTable3D(img_, bitmap);

	//bitmap.Wrap(img_.data, width, height, depth);
	//parent.Wrap(reinterpret_cast<int32_t*>(parent_labels.data), width + 2, height + 2, depth + 2);

	parent.Resize(width + 2, height + 2);
	labels = parent.SubTable(1, 1, 0, width, height, 0);
	
	
	double ma_t = t - perf_.last();
	// Return total time
	return ls_t + ma_t;
    }
    
    void Apply() {
	TableToImage3D(labels, img_labels_);
    }
    
    void Dealloc() {
	// No free for img_labels_ because it is required at the end of the algorithm 
    }
    void FirstScan() {
	algo::rosenfeld_2x2_first_scan(bitmap, uf, labels, stats);
    }

    void SecondScan() {
	equivalence_resolution_pack(uf);
	algo::rosenfeld_naive_second_scan(uf, labels, stats);

    }
};

#endif // !YACCLAB_LABELING_ROSENFELD_2X2_H_
