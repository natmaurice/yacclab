// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_WU_2009_H_
#define YACCLAB_LABELING_WU_2009_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

#include "debug.h"

template <typename LabelsSolver>
class SAUF : public Labeling2D<Connectivity2D::CONN_8> {
protected:
    LabelsSolver ET;
public:
    SAUF() {}

    void PerformLabeling()
    {
        const int h = img_.rows;
        const int w = img_.cols;

        img_labels_ = cv::Mat1i(img_.size(), 0); // Allocation + initialization of the output image

        ET.Alloc(UPPER_BOUND_8_CONNECTIVITY); // Memory allocation of the labels solver
        ET.Setup(); // Labels solver initialization
	
        // Rosenfeld Mask
        // +-+-+-+
        // |p|q|r|
        // +-+-+-+
        // |s|x|
        // +-+-+

        // First scan
        for (int r = 0; r < h; ++r) {
            // Get row pointers
            unsigned char const * const img_row = img_.ptr<unsigned char>(r);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            for (int c = 0; c < w; ++c) {
#define CONDITION_P c > 0 && r > 0 && img_row_prev[c - 1] > 0
#define CONDITION_Q r > 0 && img_row_prev[c] > 0
#define CONDITION_R c < w - 1 && r > 0 && img_row_prev[c + 1] > 0
#define CONDITION_S c > 0 && img_row[c - 1] > 0
#define CONDITION_X img_row[c] > 0

#define ACTION_1 // nothing to do 
#define ACTION_2 img_labels_row[c] = ET.NewLabel(); // new label
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c - 1]; // x <- p
#define ACTION_4 img_labels_row[c] = img_labels_row_prev[c]; // x <- q
#define ACTION_5 img_labels_row[c] = img_labels_row_prev[c + 1]; // x <- r
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 1]; // x <- s
#define ACTION_7 img_labels_row[c] = ET.Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]); // x <- p + r
#define ACTION_8 img_labels_row[c] = ET.Merge(img_labels_row[c - 1], img_labels_row_prev[c + 1]); // x <- s + r

#include "labeling_wu_2009_tree.inc.h"
            }
        }

        // Second scan
        n_labels_ = ET.Flatten();

        for (int r = 0; r < img_labels_.rows; ++r) {
            unsigned * img_row_start = img_labels_.ptr<unsigned>(r);
            unsigned * const img_row_end = img_row_start + img_labels_.cols;
            for (; img_row_start != img_row_end; ++img_row_start) {
                *img_row_start = ET.GetLabel(*img_row_start);
            }
        }

        ET.Dealloc(); // Memory deallocation of the labels solver

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_X
	/*static long run = 0;
	run++;
	long img_cols = img_.cols;
	long img_rows = img_.rows;
	long labels_cols = img_labels_.cols;
	long labels_rows = img_labels_.rows;
	
	std::cout << std::endl << "Wu [" << run << "] " << std::endl
		  << "Image (" <<  img_cols << ", " << img_rows << ")" << std::endl
		  << img_ << std::endl << std::endl
		  << "Labels (" << labels_cols << ", " << labels_rows<< ")" << std::endl
		  << img_labels_ << std::endl;
	if (run == 3) {
	    debug();
	    }*/
    }

    void PerformLabelingWithSteps()
    {
	uint32_t height = this->img_.size.p[0];
	uint32_t width = this->img_.size.p[1];
	uint32_t size = height * width;

	Labeling::StepsDuration elapsed;
	elapsed.Init();

	double alloc_timing = Alloc();

	MEASURE_STEP_TIME(FirstScan(), StepType::FIRST_SCAN, this->perf_, elapsed,
			  this->samplers, size);

        SecondScan(elapsed);

        perf_.start();
        Dealloc();
        perf_.stop();
        perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing);
	//std::cout << "Wu: " << std::endl << img_labels_ << std::endl;

	elapsed.CalcDerivedTime();
	elapsed.StoreAll(this->perf_);
	this->samplers.CalcDerived();
    }

    void PerformLabelingMem(std::vector<uint64_t>& accesses)
    {
        const int h = img_.rows;
        const int w = img_.cols;

        ET.MemAlloc(UPPER_BOUND_8_CONNECTIVITY); // Equivalence solver

        // Data structure for memory test
        MemMat<unsigned char> img(img_);
        MemMat<int> img_labels(img_.size(), 0);

        ET.MemSetup();

        // First scan

        // Rosenfeld Mask
        // +-+-+-+
        // |p|q|r|
        // +-+-+-+
        // |s|x|
        // +-+-+

        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
#define CONDITION_P c > 0 && r > 0 && img(r - 1 , c - 1) > 0
#define CONDITION_Q r > 0 && img(r - 1, c)>0
#define CONDITION_R c < w - 1 && r > 0 && img(r - 1,c + 1) > 0
#define CONDITION_S c > 0 && img(r,c - 1)>0
#define CONDITION_X img(r,c)>0

#define ACTION_1 // nothing to do 
#define ACTION_2 img_labels(r, c) = ET.MemNewLabel(); // new label
#define ACTION_3 img_labels(r, c) = img_labels(r - 1, c - 1); // x <- p
#define ACTION_4 img_labels(r, c) = img_labels(r - 1, c); // x <- q
#define ACTION_5 img_labels(r, c) = img_labels(r - 1, c + 1); // x <- r
#define ACTION_6 img_labels(r, c) = img_labels(r, c - 1); // x <- s
#define ACTION_7 img_labels(r, c) = ET.MemMerge((unsigned)img_labels(r - 1, c - 1), (unsigned)img_labels(r - 1, c + 1)); // x <- p + r
#define ACTION_8 img_labels(r, c) = ET.MemMerge((unsigned)img_labels(r, c - 1), (unsigned)img_labels(r - 1, c + 1)); // x <- s + r

#include "labeling_wu_2009_tree.inc.h"              
            }
        }

        // Second scan
        n_labels_ = ET.MemFlatten();

        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                img_labels(r, c) = ET.MemGetLabel(img_labels(r, c));
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (uint64_t)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (uint64_t)img_labels.GetTotalAccesses();
        accesses[MD_EQUIVALENCE_VEC] = (uint64_t)ET.MemTotalAccesses();

        img_labels_ = img_labels.GetImage();

        ET.MemDealloc();

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_X
    }

private:
    double Alloc()
    {
	this->samplers.Reset();

	// Memory allocation of the labels solver
        double ls_t = ET.Alloc(UPPER_BOUND_8_CONNECTIVITY, perf_);
        // Memory allocation for the output image
        perf_.start();
        img_labels_ = cv::Mat1i(img_.size());
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        perf_.stop();
        double t = perf_.last();
        perf_.start();
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart);
        perf_.stop();
        double ma_t = t - perf_.last();
        // Return total time
        return ls_t + ma_t;
    }
    void Dealloc() {
        ET.Dealloc();
        // No free for img_labels_ because it is required at the end of the algorithm

    }
    void FirstScan() {

        const int h = img_.rows;
        const int w = img_.cols;

        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart); // Initialization

        ET.Setup();

        // Rosenfeld Mask
        // +-+-+-+
        // |p|q|r|
        // +-+-+-+
        // |s|x|
        // +-+-+

        // First scan
        for (int r = 0; r < h; ++r) {
            // Get row pointers
            unsigned char const * const img_row = img_.ptr<unsigned char>(r);
            unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

            for (int c = 0; c < w; ++c) {
#define CONDITION_P c > 0 && r > 0 && img_row_prev[c - 1] > 0
#define CONDITION_Q r > 0 && img_row_prev[c] > 0
#define CONDITION_R c < w - 1 && r > 0 && img_row_prev[c + 1] > 0
#define CONDITION_S c > 0 && img_row[c - 1] > 0
#define CONDITION_X img_row[c] > 0

#define ACTION_1 // nothing to do 
#define ACTION_2 img_labels_row[c] = ET.NewLabel(); // new label
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c - 1]; // x <- p
#define ACTION_4 img_labels_row[c] = img_labels_row_prev[c]; // x <- q
#define ACTION_5 img_labels_row[c] = img_labels_row_prev[c + 1]; // x <- r
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 1]; // x <- s
#define ACTION_7 img_labels_row[c] = ET.Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]); // x <- p + r
#define ACTION_8 img_labels_row[c] = ET.Merge(img_labels_row[c - 1], img_labels_row_prev[c + 1]); // x <- s + r

#include "labeling_wu_2009_tree.inc.h"
            }
        }

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_X
    }
    void SecondScan(Labeling::StepsDuration& elapsed)
    {
	uint32_t height = this->img_.size.p[0];
	uint32_t width = this->img_.size.p[1];
	uint32_t size = height * width;
	
	MEASURE_STEP_TIME(
	    this->n_labels_ = ET.Flatten(),
	    StepType::TRANSITIVE_CLOSURE, this->perf_, elapsed, this->samplers, size);

	MEASURE_STEP_TIME(
        for (int r = 0; r < img_labels_.rows; ++r) {
            unsigned * img_row_start = img_labels_.ptr<unsigned>(r);
            unsigned * const img_row_end = img_row_start + img_labels_.cols;
            for (; img_row_start != img_row_end; ++img_row_start) {
                *img_row_start = ET.GetLabel(*img_row_start);
            }
        },  StepType::RELABELING, this->perf_, elapsed, this->samplers, size);
    }
};

#endif // !YACCLAB_LABELING_WU_2009_H_
