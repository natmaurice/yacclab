// Copyright (c) 2022, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_PREDPP_H_
#define YACCLAB_LABELING_PREDPP_H_

#include <vector>

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver, typename ConfFeatures = ConfFeatures2DNone>
class PREDpp : public Labeling2D<Connectivity2D::CONN_8> {
protected:
    LabelsSolver ET;
public:
    PREDpp() {}

    void PerformLabeling()
    {
        this->img_labels_ = cv::Mat1i(this->img_.size(), 0); // Call to memset

        ET.Alloc(UPPER_BOUND_8_CONNECTIVITY);
        ET.Setup();

        // Rosenfeld Mask
        // +-+-+-+
        // |p|q|r|
        // +-+-+-+
        // |s|x|
        // +-+-+

        // First Scan
        int w(this->img_.cols);
        int h(this->img_.rows);

        //Conditions:
#define CONDITION_P img_row11[c - 1] > 0
#define CONDITION_Q img_row11[c] > 0
#define CONDITION_R img_row11[c + 1] > 0
#define CONDITION_S img_row00[c - 1] > 0
#define CONDITION_X img_row00[c] > 0


//Actions:
// Action 1: nothing
#define ACTION_1 
// Action 2: x<-newlabel
#define ACTION_2 img_labels_row00[c] = ET.NewLabel();
// Action 3: x<-p
#define ACTION_3 img_labels_row00[c] = img_labels_row11[c - 1];
// Action 4: x<-q
#define ACTION_4 img_labels_row00[c] = img_labels_row11[c];
// Action 5: x<-r
#define ACTION_5 img_labels_row00[c] = img_labels_row11[c + 1];
// Action 6: x<-p+r
#define ACTION_6 img_labels_row00[c] = ET.Merge(img_labels_row11[c + 1], img_labels_row11[c - 1]);
// Action 7: x<-s
#define ACTION_7 img_labels_row00[c] = img_labels_row00[c - 1];
// Action 8: x<-r+s
#define ACTION_8 img_labels_row00[c] = ET.Merge(img_labels_row00[c - 1], img_labels_row11[c + 1]);


#define COLS w

        {
            int c = -1;

            const unsigned char* const img_row00 = this->img_.template ptr<unsigned char>(0);

            // Row pointers for the output image 
            unsigned* const img_labels_row00 = this->img_labels_.template ptr<unsigned>(0);
            goto fl_tree_0;
#include "labeling_PREDpp_2021_fl_forest.inc.h"
        }

        for (int r = 1; r < h; ++r) {
            int c = -1;
            // Get rows pointer
    // Row pointers for the input image 
            const unsigned char* const img_row00 = this->img_.template ptr<unsigned char>(r);
            const unsigned char* const img_row11 = (unsigned char *)(((char *)img_row00) + this->img_.step.p[0] * -1);

            // Row pointers for the output image 
            unsigned* const img_labels_row00 = this->img_labels_.template ptr<unsigned>(r);
            unsigned* const img_labels_row11 = (unsigned *)(((char *)img_labels_row00) + this->img_labels_.step.p[0] * -1);
            goto cl_tree_0;
#include "labeling_PREDpp_2021_cl_forest.inc.h"

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
        this->n_labels_ = ET.template Flatten<ConfFeatures>(this->features);

        for (int r_i = 0; r_i < this->img_labels_.rows; ++r_i) {
            unsigned int *b = this->img_labels_.template ptr<unsigned int>(r_i);
            unsigned int *e = b + this->img_labels_.cols;
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


private:
    double Alloc()
    {
        // Memory allocation of the labels solver
        double ls_t = ET.Alloc(UPPER_BOUND_8_CONNECTIVITY, this->perf_);
        // Memory allocation for the output image
        this->perf_.start();
        this->img_labels_ = cv::Mat1i(this->img_.size());
        memset(this->img_labels_.data, 0, this->img_labels_.dataend - this->img_labels_.datastart);
        this->perf_.stop();
        double t = this->perf_.last();
        this->perf_.start();
        memset(this->img_labels_.data, 0, this->img_labels_.dataend - this->img_labels_.datastart);
        this->perf_.stop();
        double ma_t = t - this->perf_.last();
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
        memset(this->img_labels_.data, 0, this->img_labels_.dataend - this->img_labels_.datastart); // Initialization
        ET.Setup();

        // First Scan
        int w(this->img_.cols);
        int h(this->img_.rows);

#define CONDITION_P img_row11[c - 1] > 0
#define CONDITION_Q img_row11[c] > 0
#define CONDITION_R img_row11[c + 1] > 0
#define CONDITION_S img_row00[c - 1] > 0
#define CONDITION_X img_row00[c] > 0


        //Actions:
        // Action 1: nothing
#define ACTION_1 
// Action 2: x<-newlabel
#define ACTION_2 img_labels_row00[c] = ET.NewLabel();
// Action 3: x<-p
#define ACTION_3 img_labels_row00[c] = img_labels_row11[c - 1];
// Action 4: x<-q
#define ACTION_4 img_labels_row00[c] = img_labels_row11[c];
// Action 5: x<-r
#define ACTION_5 img_labels_row00[c] = img_labels_row11[c + 1];
// Action 6: x<-p+r
#define ACTION_6 img_labels_row00[c] = ET.Merge(img_labels_row11[c + 1], img_labels_row11[c - 1]);
// Action 7: x<-s
#define ACTION_7 img_labels_row00[c] = img_labels_row00[c - 1];
// Action 8: x<-r+s
#define ACTION_8 img_labels_row00[c] = ET.Merge(img_labels_row00[c - 1], img_labels_row11[c + 1]);

#define COLS w

        {
            int c = -1;
            // Get rows pointer
            // Row pointers for the input image 
            const unsigned char* const img_row00 = this->img_.template ptr<unsigned char>(0);

            // Row pointers for the output image 
            unsigned* const img_labels_row00 = this->img_labels_.template ptr<unsigned>(0);
            goto fl_tree_0;
#include "labeling_PREDpp_2021_fl_forest.inc.h"
        }

        for (int r = 1; r < h; ++r) {
            // Row pointers for the input image 
            const unsigned char* const img_row00 = this->img_.template ptr<unsigned char>(r);
            const unsigned char* const img_row11 = (unsigned char *)(((char *)img_row00) + this->img_.step.p[0] * -1);

            // Row pointers for the output image 
            unsigned* const img_labels_row00 = this->img_labels_.template ptr<unsigned>(r);
            unsigned* const img_labels_row11 = (unsigned *)(((char *)img_labels_row00) + this->img_labels_.step.p[0] * -1);
            int c = -1;
            goto cl_tree_0;
#include "labeling_PREDpp_2021_cl_forest.inc.h"

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
	    this->n_labels_ = ET.template Flatten<ConfFeatures>(this->features),
	    StepType::TRANSITIVE_CLOSURE, this->perf_, elapsed, this->samplers, size);
	
	MEASURE_STEP_TIME(
	    for (int r_i = 0; r_i < this->img_labels_.rows; ++r_i) {
		unsigned int *b = this->img_labels_.template ptr<unsigned int>(r_i);
		unsigned int *e = b + this->img_labels_.cols;
		for (; b != e; ++b) {
		    *b = ET.GetLabel(*b);
		}
	    }, StepType::RELABELING, this->perf_, elapsed, this->samplers, size);
    }
};

#endif // YACCLAB_LABELING_PREDPP_H_
