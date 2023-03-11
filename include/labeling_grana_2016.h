// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_GRANA_2016_H_
#define YACCLAB_LABELING_GRANA_2016_H_

#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class PRED : public Labeling2D<Connectivity2D::CONN_8> {
protected:
    LabelsSolver ET;
public:
    PRED() {}

    void PerformLabeling()
    {
        img_labels_ = cv::Mat1i(img_.size(), 0); // Call to memset

        ET.Alloc(UPPER_BOUND_8_CONNECTIVITY);
        ET.Setup();

        // Rosenfeld Mask
        // +-+-+-+
        // |p|q|r|
        // +-+-+-+
        // |s|x|
        // +-+-+

        // First Scan
        int w(img_.cols);
        int h(img_.rows);

#define CONDITION_X img_row[c]>0
#define CONDITION_P img_row_prev[c-1]>0
#define CONDITION_Q img_row_prev[c]>0
#define CONDITION_R img_row_prev[c+1]>0

#define ACTION_1 // nothing to do 
#define ACTION_2 img_labels_row[c] = ET.NewLabel(); // new label
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c - 1]; // x <- p
#define ACTION_4 img_labels_row[c] = img_labels_row_prev[c]; // x <- q
#define ACTION_5 img_labels_row[c] = img_labels_row_prev[c + 1]; // x <- r
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 1]; // x <- s
#define ACTION_7 img_labels_row[c] = ET.Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]); // x <- p + r
#define ACTION_8 img_labels_row[c] = ET.Merge(img_labels_row[c - 1], img_labels_row_prev[c + 1]); // x <- s + r

#define COLS w

        {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(0);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(0);

#include "labeling_grana_2016_forest_0.inc.h"

        }

        for (int r = 1; r < h; ++r) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0]);

#include "labeling_grana_2016_forest.inc.h"  

        }//End rows's for

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_X
#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R

    // Second scan
        n_labels_ = ET.Flatten();

        for (int r_i = 0; r_i < img_labels_.rows; ++r_i) {
            unsigned int *b = img_labels_.ptr<unsigned int>(r_i);
            unsigned int *e = b + img_labels_.cols;
            for (; b != e; ++b) {
                *b = ET.GetLabel(*b);
            }
        }

        ET.Dealloc();
    }


    void PerformLabelingWithSteps()
    {
	Labeling::StepsDuration elapsed;
	elapsed.Init();

	uint32_t height = this->img_.size.p[0];
	uint32_t width = this->img_.size.p[1];
	uint32_t size = height * width;
	
	double alloc_timing = Alloc();

	MEASURE_STEP_TIME(FirstScan(), StepType::FIRST_SCAN, this->perf_, elapsed,
			  this->samplers, size);

        SecondScan(elapsed);

        perf_.start();
        Dealloc();
        perf_.stop();
        perf_.store(Step(StepType::ALLOC_DEALLOC), perf_.last() + alloc_timing);

	elapsed.CalcDerivedTime();
	elapsed.StoreAll(this->perf_);
	this->samplers.CalcDerived();
    }
    
    void PerformLabelingMem(std::vector<uint64_t>& accesses)
    {

        MemMat<unsigned char> img(img_);
        MemMat<int> img_labels(img_.size(), 0); 

        ET.MemAlloc(UPPER_BOUND_8_CONNECTIVITY);
        ET.MemSetup();

        // First scan
        int w(img_.cols);
        int h(img_.rows);

#define CONDITION_X img(r,c)>0
#define CONDITION_P img(r-1,c-1)>0
#define CONDITION_Q img(r-1,c)>0
#define CONDITION_R img(r-1,c+1)>0

#define ACTION_1 // nothing to do 
#define ACTION_2 img_labels(r,c) = ET.MemNewLabel(); // new label
#define ACTION_3 img_labels(r,c) = img_labels(r - 1, c - 1); // x <- p
#define ACTION_4 img_labels(r,c) = img_labels(r - 1, c); // x <- q
#define ACTION_5 img_labels(r,c) = img_labels(r - 1, c + 1); // x <- r
#define ACTION_6 img_labels(r,c) = img_labels(r, c - 1); // x <- s
#define ACTION_7 img_labels(r,c) = ET.MemMerge(img_labels(r - 1, c - 1), img_labels(r - 1, c + 1)); // x <- p + r
#define ACTION_8 img_labels(r,c) = ET.MemMerge(img_labels(r, c - 1), img_labels(r - 1, c + 1)); // x <- s + r

#define COLS w

        {
            int r = 0; 
#include "labeling_grana_2016_forest_0.inc.h"

        }

        for (int r = 1; r < h; ++r) {

#include "labeling_grana_2016_forest.inc.h"  

        }//End rows' for

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_X
#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R

        n_labels_ = ET.MemFlatten();

        // Second scan
        for (int r_i = 0; r_i < img_labels.rows; ++r_i) {
            for (int c_i = 0; c_i < img_labels.cols; ++c_i) {
                img_labels(r_i, c_i) = ET.MemGetLabel(img_labels(r_i, c_i));
            }
        }

        // Store total accesses in the output vector 'accesses'
        accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

        accesses[MD_BINARY_MAT] = (uint64_t)img.GetTotalAccesses();
        accesses[MD_LABELED_MAT] = (uint64_t)img_labels.GetTotalAccesses();
        accesses[MD_EQUIVALENCE_VEC] = (uint64_t)ET.MemTotalAccesses();

        img_labels_ = img_labels.GetImage(); 

        ET.MemDealloc();
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

    void Dealloc()
    {
        ET.Dealloc();
        // No free for img_labels_ because it is required at the end of the algorithm 
    }
    void FirstScan()
    {
        memset(img_labels_.data, 0, img_labels_.dataend - img_labels_.datastart); // Initialization
        ET.Setup();

        // First Scan
        int w(img_.cols);
        int h(img_.rows);

#define CONDITION_X img_row[c]>0
#define CONDITION_P img_row_prev[c-1]>0
#define CONDITION_Q img_row_prev[c]>0
#define CONDITION_R img_row_prev[c+1]>0

#define ACTION_1 // nothing to do 
#define ACTION_2 img_labels_row[c] = ET.NewLabel(); // new label
#define ACTION_3 img_labels_row[c] = img_labels_row_prev[c - 1]; // x <- p
#define ACTION_4 img_labels_row[c] = img_labels_row_prev[c]; // x <- q
#define ACTION_5 img_labels_row[c] = img_labels_row_prev[c + 1]; // x <- r
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 1]; // x <- s
#define ACTION_7 img_labels_row[c] = ET.Merge(img_labels_row_prev[c - 1], img_labels_row_prev[c + 1]); // x <- p + r
#define ACTION_8 img_labels_row[c] = ET.Merge(img_labels_row[c - 1], img_labels_row_prev[c + 1]); // x <- s + r

#define COLS w

        {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(0);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(0);

#include "labeling_grana_2016_forest_0.inc.h"
                
        }

        for (int r = 1; r < h; ++r) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            unsigned int* const img_labels_row = img_labels_.ptr<unsigned int>(r);
            unsigned int* const img_labels_row_prev = (unsigned int *)(((char *)img_labels_row) - img_labels_.step.p[0]);
        
#include "labeling_grana_2016_forest.inc.h"  

        }//End rows's for

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8

#undef CONDITION_X
#undef CONDITION_P
#undef CONDITION_Q
#undef CONDITION_R
    }
    void SecondScan(Labeling::StepsDuration& elapsed)
    {
	uint32_t height = this->img_.size.p[0];
	uint32_t width = this->img_.size.p[1];
	uint32_t size = height * width;

	// Second scan
	MEASURE_STEP_TIME(
	    this->n_labels_ = ET.Flatten(),
	    StepType::TRANSITIVE_CLOSURE, this->perf_, elapsed, this->samplers, size);

	MEASURE_STEP_TIME(
	    for (int r_i = 0; r_i < img_labels_.rows; ++r_i) {
		unsigned int *b = img_labels_.ptr<unsigned int>(r_i);
		unsigned int *e = b + img_labels_.cols;
		for (; b != e; ++b) {
		    *b = ET.GetLabel(*b);
		}
	    }, StepType::RELABELING, this->perf_, elapsed, this->samplers, size);
    }
};

#endif // YACCLAB_LABELING_GRANA_2016_H_
