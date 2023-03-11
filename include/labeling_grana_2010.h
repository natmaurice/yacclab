// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_GRANA_2010_H_
#define YACCLAB_LABELING_GRANA_2010_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"

template <typename LabelsSolver>
class BBDT : public Labeling2D<Connectivity2D::CONN_8> {
protected:
    LabelsSolver ET;
public:
    BBDT() {}

    void PerformLabeling()
    {
        const int h = img_.rows;
        const int w = img_.cols;

        img_labels_ = cv::Mat1i(img_.size()); // Memory allocation for the output image

        ET.Alloc(UPPER_BOUND_8_CONNECTIVITY); // Memory allocation of the labels solver
        ET.Setup(); // Labels solver initialization

        // We work with 2x2 blocks
        // +-+-+-+
        // |P|Q|R|
        // +-+-+-+
        // |S|X|
        // +-+-+

        // The pixels are named as follows
        // +---+---+---+
        // |a b|c d|e f|
        // |g h|i j|k l|
        // +---+---+---+
        // |m n|o p|
        // |q r|s t|
        // +---+---+

        // Pixels a, f, l, q are not needed, since we need to understand the
        // the connectivity between these blocks and those pixels only matter
        // when considering the outer connectivities

        // A bunch of defines used to check if the pixels are foreground,
        // without going outside the image limits.

        // First scan
        for (int r = 0; r < h; r += 2) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            const unsigned char* const img_row_prev_prev = (unsigned char *)(((char *)img_row_prev) - img_.step.p[0]);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned* const img_labels_row_prev_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);

            for (int c = 0; c < w; c += 2) {

#define CONDITION_B c-1>=0 && r-2>=0 && img_row_prev_prev[c-1]>0
#define CONDITION_C r-2>=0 && img_row_prev_prev[c]>0
#define CONDITION_D c+1<w && r-2>=0 && img_row_prev_prev[c+1]>0
#define CONDITION_E c+2<w && r-2>=0 && img_row_prev_prev[c+2]>0

#define CONDITION_G c-2>=0 && r-1>=0 && img_row_prev[c-2]>0
#define CONDITION_H c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
#define CONDITION_I r-1>=0 && img_row_prev[c]>0
#define CONDITION_J c+1<w && r-1>=0 && img_row_prev[c+1]>0
#define CONDITION_K c+2<w && r-1>=0 && img_row_prev[c+2]>0

#define CONDITION_M c-2>=0 && img_row[c-2]>0
#define CONDITION_N c-1>=0 && img_row[c-1]>0
#define CONDITION_O img_row[c]>0
#define CONDITION_P c+1<w && img_row[c+1]>0

#define CONDITION_R c-1>=0 && r+1<h && img_row_fol[c-1]>0
#define CONDITION_S r+1<h && img_row_fol[c]>0
#define CONDITION_T c+1<w && r+1<h && img_row_fol[c+1]>0

                // Action 1: No action
#define ACTION_1 img_labels_row[c] = 0; continue; 
                // Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2 img_labels_row[c] = ET.NewLabel(); continue; 
                //Action 3: Assign label of block P
#define ACTION_3 img_labels_row[c] = img_labels_row_prev_prev[c - 2]; continue;
                // Action 4: Assign label of block Q 
#define ACTION_4 img_labels_row[c] = img_labels_row_prev_prev[c]; continue;
                // Action 5: Assign label of block R
#define ACTION_5 img_labels_row[c] = img_labels_row_prev_prev[c + 2]; continue;
                // Action 6: Assign label of block S
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 2]; continue; 
                // Action 7: Merge labels of block P and Q
#define ACTION_7 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]); continue;
                //Action 8: Merge labels of block P and R
#define ACTION_8 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]); continue;
                // Action 9 Merge labels of block P and S
#define ACTION_9 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]); continue;
                // Action 10 Merge labels of block Q and R
#define ACTION_10 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]); continue;
                // Action 11: Merge labels of block Q and S
#define ACTION_11 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c], img_labels_row[c - 2]); continue;
                // Action 12: Merge labels of block R and S
#define ACTION_12 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]); continue;
                // Action 13: not used
#define ACTION_13 
                // Action 14: Merge labels of block P, Q and S
#define ACTION_14 img_labels_row[c] = ET.Merge(ET.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row[c - 2]); continue;
                //Action 15: Merge labels of block P, R and S
#define ACTION_15 img_labels_row[c] = ET.Merge(ET.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]); continue;
                //Action 16: labels of block Q, R and S
#define ACTION_16 img_labels_row[c] = ET.Merge(ET.Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]); continue;

#include "labeling_grana_2010_tree.inc.h"
            }
        }

        // Second scan
        n_labels_ = ET.Flatten();

        int e_rows = img_labels_.rows & 0xfffffffe;
        bool o_rows = img_labels_.rows % 2 == 1;
        int e_cols = img_labels_.cols & 0xfffffffe;
        bool o_cols = img_labels_.cols % 2 == 1;

        int r = 0;
        for (; r < e_rows; r += 2) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
            int c = 0;
            for (; c < e_cols; c += 2) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = ET.GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                    if (img_row[c + 1] > 0)
                        img_labels_row[c + 1] = iLabel;
                    else
                        img_labels_row[c + 1] = 0;
                    if (img_row_fol[c] > 0)
                        img_labels_row_fol[c] = iLabel;
                    else
                        img_labels_row_fol[c] = 0;
                    if (img_row_fol[c + 1] > 0)
                        img_labels_row_fol[c + 1] = iLabel;
                    else
                        img_labels_row_fol[c + 1] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                    img_labels_row[c + 1] = 0;
                    img_labels_row_fol[c] = 0;
                    img_labels_row_fol[c + 1] = 0;
                }
            }
            // Last column if the number of columns is odd
            if (o_cols) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = ET.GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                    if (img_row_fol[c] > 0)
                        img_labels_row_fol[c] = iLabel;
                    else
                        img_labels_row_fol[c] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                    img_labels_row_fol[c] = 0;
                }
            }
        }
        // Last row if the number of rows is odd
        if (o_rows) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            int c = 0;
            for (; c < e_cols; c += 2) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = ET.GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                    if (img_row[c + 1] > 0)
                        img_labels_row[c + 1] = iLabel;
                    else
                        img_labels_row[c + 1] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                    img_labels_row[c + 1] = 0;
                }
            }
            // Last column if the number of columns is odd
            if (o_cols) {
                int iLabel = img_labels_row[c];
                if (iLabel > 0) {
                    iLabel = ET.GetLabel(iLabel);
                    if (img_row[c] > 0)
                        img_labels_row[c] = iLabel;
                    else
                        img_labels_row[c] = 0;
                }
                else {
                    img_labels_row[c] = 0;
                }
            }
        }

        ET.Dealloc();

#undef ACTION_1
#undef ACTION_2
#undef ACTION_3
#undef ACTION_4
#undef ACTION_5
#undef ACTION_6
#undef ACTION_7
#undef ACTION_8
#undef ACTION_9
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14
#undef ACTION_15
#undef ACTION_16


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
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
        const int h = img_.rows;
        const int w = img_.cols;

        ET.MemAlloc(UPPER_BOUND_8_CONNECTIVITY);
        ET.MemSetup();

        //Data structure for memory test
        MemMat<unsigned char> img(img_);
        MemMat<int> img_labels(img_.size(), 0);

        // We work with 2x2 blocks
        // +-+-+-+
        // |P|Q|R|
        // +-+-+-+
        // |S|X|
        // +-+-+

        // The pixels are named as follows
        // +---+---+---+
        // |a b|c d|e f|
        // |g h|i j|k l|
        // +---+---+---+
        // |m n|o p|
        // |q r|s t|
        // +---+---+

        // Pixels a, f, l, q are not needed, since we need to understand the
        // the connectivity between these blocks and those pixels only metter
        // when considering the outer connectivities

        // A bunch of defines used to check if the pixels are foreground,
        // without going outside the image limits.

        for (int r = 0; r < h; r += 2) {
            for (int c = 0; c < w; c += 2) {

#define CONDITION_B c-1>=0 && r-2>=0 && img(r-2, c-1)>0
#define CONDITION_C r-2>=0 && img(r-2, c)>0
#define CONDITION_D c+1<w && r-2>=0 && img(r-2, c+1)>0
#define CONDITION_E c+2<w && r-2>=0 && img(r-2, c+2)>0

#define CONDITION_G c-2>=0 && r-1>=0 && img(r-1, c-2)>0
#define CONDITION_H c-1>=0 && r-1>=0 && img(r-1, c-1)>0
#define CONDITION_I r-1>=0 && img(r-1, c)>0
#define CONDITION_J c+1<w && r-1>=0 && img(r-1, c+1)>0
#define CONDITION_K c+2<w && r-1>=0 && img(r-1, c+2)>0

#define CONDITION_M c-2>=0 && img(r, c-2)>0
#define CONDITION_N c-1>=0 && img(r, c-1)>0
#define CONDITION_O img(r,c)>0
#define CONDITION_P c+1<w && img(r,c+1)>0

#define CONDITION_R c-1>=0 && r+1<h && img(r+1, c-1)>0
#define CONDITION_S r+1<h && img(r+1, c)>0
#define CONDITION_T c+1<w && r+1<h && img(r+1, c+1)>0

                // Action 1: No action
#define ACTION_1 img_labels(r, c) = 0; continue; 
                // Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2 img_labels(r, c) = ET.MemNewLabel(); continue; 
                //Action 3: Assign label of block P
#define ACTION_3 img_labels(r, c) = img_labels(r - 2, c - 2); continue;
                // Action 4: Assign label of block Q 
#define ACTION_4 img_labels(r, c) = img_labels(r - 2, c); continue;
                // Action 5: Assign label of block R
#define ACTION_5 img_labels(r, c) = img_labels(r - 2, c + 2); continue;
                // Action 6: Assign label of block S
#define ACTION_6 img_labels(r, c) = img_labels(r, c - 2); continue; 
                // Action 7: Merge labels of block P and Q
#define ACTION_7 img_labels(r, c) = ET.MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c)); continue;
                //Action 8: Merge labels of block P and R
#define ACTION_8 img_labels(r, c) = ET.MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c + 2)); continue;
                // Action 9 Merge labels of block P and S
#define ACTION_9 img_labels(r, c) = ET.MemMerge(img_labels(r - 2, c - 2), img_labels(r, c - 2)); continue;
                // Action 10 Merge labels of block Q and R
#define ACTION_10 img_labels(r, c) = ET.MemMerge(img_labels(r - 2, c), img_labels(r - 2, c + 2)); continue;
                // Action 11: Merge labels of block Q and S
#define ACTION_11 img_labels(r, c) = ET.MemMerge(img_labels(r - 2, c), img_labels(r, c - 2)); continue;
                // Action 12: Merge labels of block R and S
#define ACTION_12 img_labels(r, c) = ET.MemMerge(img_labels(r - 2, c + 2), img_labels(r, c - 2)); continue;
                // Action 13: not used
#define ACTION_13 
                // Action 14: Merge labels of block P, Q and S
#define ACTION_14 img_labels(r, c) = ET.MemMerge(ET.MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c)), img_labels(r, c - 2)); continue;
                //Action 15: Merge labels of block P, R and S
#define ACTION_15 img_labels(r, c) = ET.MemMerge(ET.MemMerge(img_labels(r - 2, c - 2), img_labels(r - 2, c + 2)), img_labels(r, c - 2)); continue;
                //Action 16: labels of block Q, R and S
#define ACTION_16 img_labels(r, c) = ET.MemMerge(ET.MemMerge(img_labels(r - 2, c), img_labels(r - 2, c + 2)), img_labels(r, c - 2)); continue;

#include "labeling_grana_2010_tree.inc.h"
            }
        }

        n_labels_ = ET.MemFlatten();

        // Second scan
        for (int r = 0; r < h; r += 2) {
            for (int c = 0; c < w; c += 2) {
                int iLabel = img_labels(r, c);
                if (iLabel > 0) {
                    iLabel = ET.MemGetLabel(iLabel);
                    if (img(r, c) > 0)
                        img_labels(r, c) = iLabel;
                    else
                        img_labels(r, c) = 0;
                    if (c + 1 < w) {
                        if (img(r, c + 1) > 0)
                            img_labels(r, c + 1) = iLabel;
                        else
                            img_labels(r, c + 1) = 0;
                        if (r + 1 < h) {
                            if (img(r + 1, c) > 0)
                                img_labels(r + 1, c) = iLabel;
                            else
                                img_labels(r + 1, c) = 0;
                            if (img(r + 1, c + 1) > 0)
                                img_labels(r + 1, c + 1) = iLabel;
                            else
                                img_labels(r + 1, c + 1) = 0;
                        }
                    }
                    else if (r + 1 < h) {
                        if (img(r + 1, c) > 0)
                            img_labels(r + 1, c) = iLabel;
                        else
                            img_labels(r + 1, c) = 0;
                    }
                }
                else {
                    img_labels(r, c) = 0;
                    if (c + 1 < w) {
                        img_labels(r, c + 1) = 0;
                        if (r + 1 < h) {
                            img_labels(r + 1, c) = 0;
                            img_labels(r + 1, c + 1) = 0;
                        }
                    }
                    else if (r + 1 < h) {
                        img_labels(r + 1, c) = 0;
                    }
                }
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
#undef ACTION_9
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14
#undef ACTION_15
#undef ACTION_16


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
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
        const int h = img_.rows;
        const int w = img_.cols;

        ET.Setup(); // Labels solver initialization

        // We work with 2x2 blocks
        // +-+-+-+
        // |P|Q|R|
        // +-+-+-+
        // |S|X|
        // +-+-+

        // The pixels are named as follows
        // +---+---+---+
        // |a b|c d|e f|
        // |g h|i j|k l|
        // +---+---+---+
        // |m n|o p|
        // |q r|s t|
        // +---+---+

        // Pixels a, f, l, q are not needed, since we need to understand the
        // the connectivity between these blocks and those pixels only matter
        // when considering the outer connectivities

        // A bunch of defines used to check if the pixels are foreground,
        // without going outside the image limits.

        // First scan
        for (int r = 0; r < h; r += 2) {
            // Get rows pointer
            const unsigned char* const img_row = img_.ptr<unsigned char>(r);
            const unsigned char* const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
            const unsigned char* const img_row_prev_prev = (unsigned char *)(((char *)img_row_prev) - img_.step.p[0]);
            const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);
            unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
            unsigned* const img_labels_row_prev_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0] - img_labels_.step.p[0]);

            for (int c = 0; c < w; c += 2) {

#define CONDITION_B c-1>=0 && r-2>=0 && img_row_prev_prev[c-1]>0
#define CONDITION_C r-2>=0 && img_row_prev_prev[c]>0
#define CONDITION_D c+1<w && r-2>=0 && img_row_prev_prev[c+1]>0
#define CONDITION_E c+2<w && r-2>=0 && img_row_prev_prev[c+2]>0

#define CONDITION_G c-2>=0 && r-1>=0 && img_row_prev[c-2]>0
#define CONDITION_H c-1>=0 && r-1>=0 && img_row_prev[c-1]>0
#define CONDITION_I r-1>=0 && img_row_prev[c]>0
#define CONDITION_J c+1<w && r-1>=0 && img_row_prev[c+1]>0
#define CONDITION_K c+2<w && r-1>=0 && img_row_prev[c+2]>0

#define CONDITION_M c-2>=0 && img_row[c-2]>0
#define CONDITION_N c-1>=0 && img_row[c-1]>0
#define CONDITION_O img_row[c]>0
#define CONDITION_P c+1<w && img_row[c+1]>0

#define CONDITION_R c-1>=0 && r+1<h && img_row_fol[c-1]>0
#define CONDITION_S r+1<h && img_row_fol[c]>0
#define CONDITION_T c+1<w && r+1<h && img_row_fol[c+1]>0

                // Action 1: No action
#define ACTION_1 img_labels_row[c] = 0; continue; 
                // Action 2: New label (the block has foreground pixels and is not connected to anything else)
#define ACTION_2 img_labels_row[c] = ET.NewLabel(); continue; 
                //Action 3: Assign label of block P
#define ACTION_3 img_labels_row[c] = img_labels_row_prev_prev[c - 2]; continue;
                // Action 4: Assign label of block Q 
#define ACTION_4 img_labels_row[c] = img_labels_row_prev_prev[c]; continue;
                // Action 5: Assign label of block R
#define ACTION_5 img_labels_row[c] = img_labels_row_prev_prev[c + 2]; continue;
                // Action 6: Assign label of block S
#define ACTION_6 img_labels_row[c] = img_labels_row[c - 2]; continue; 
                // Action 7: Merge labels of block P and Q
#define ACTION_7 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]); continue;
                //Action 8: Merge labels of block P and R
#define ACTION_8 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]); continue;
                // Action 9 Merge labels of block P and S
#define ACTION_9 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c - 2], img_labels_row[c - 2]); continue;
                // Action 10 Merge labels of block Q and R
#define ACTION_10 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]); continue;
                // Action 11: Merge labels of block Q and S
#define ACTION_11 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c], img_labels_row[c - 2]); continue;
                // Action 12: Merge labels of block R and S
#define ACTION_12 img_labels_row[c] = ET.Merge(img_labels_row_prev_prev[c + 2], img_labels_row[c - 2]); continue;
                // Action 13: not used
#define ACTION_13 
                // Action 14: Merge labels of block P, Q and S
#define ACTION_14 img_labels_row[c] = ET.Merge(ET.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c]), img_labels_row[c - 2]); continue;
                //Action 15: Merge labels of block P, R and S
#define ACTION_15 img_labels_row[c] = ET.Merge(ET.Merge(img_labels_row_prev_prev[c - 2], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]); continue;
                //Action 16: labels of block Q, R and S
#define ACTION_16 img_labels_row[c] = ET.Merge(ET.Merge(img_labels_row_prev_prev[c], img_labels_row_prev_prev[c + 2]), img_labels_row[c - 2]); continue;

#include "labeling_grana_2010_tree.inc.h"
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
#undef ACTION_9
#undef ACTION_10
#undef ACTION_11
#undef ACTION_12
#undef ACTION_13
#undef ACTION_14
#undef ACTION_15
#undef ACTION_16


#undef CONDITION_B
#undef CONDITION_C
#undef CONDITION_D
#undef CONDITION_E

#undef CONDITION_G
#undef CONDITION_H
#undef CONDITION_I
#undef CONDITION_J
#undef CONDITION_K

#undef CONDITION_M
#undef CONDITION_N
#undef CONDITION_O
#undef CONDITION_P

#undef CONDITION_R
#undef CONDITION_S
#undef CONDITION_T
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
	    int e_rows = img_labels_.rows & 0xfffffffe;
	    bool o_rows = img_labels_.rows % 2 == 1;
	    int e_cols = img_labels_.cols & 0xfffffffe;
	    bool o_cols = img_labels_.cols % 2 == 1;
	
	    int r = 0;
	    for (; r < e_rows; r += 2) {
		// Get rows pointer
		const unsigned char* const img_row = img_.ptr<unsigned char>(r);
		const unsigned char* const img_row_fol = (unsigned char *)(((char *)img_row) + img_.step.p[0]);

		unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
		unsigned* const img_labels_row_fol = (unsigned *)(((char *)img_labels_row) + img_labels_.step.p[0]);
		int c = 0;
		for (; c < e_cols; c += 2) {
		    int iLabel = img_labels_row[c];
		    if (iLabel > 0) {
			iLabel = ET.GetLabel(iLabel);
			if (img_row[c] > 0)
			    img_labels_row[c] = iLabel;
			else
			    img_labels_row[c] = 0;
			if (img_row[c + 1] > 0)
			    img_labels_row[c + 1] = iLabel;
			else
			    img_labels_row[c + 1] = 0;
			if (img_row_fol[c] > 0)
			    img_labels_row_fol[c] = iLabel;
			else
			    img_labels_row_fol[c] = 0;
			if (img_row_fol[c + 1] > 0)
			    img_labels_row_fol[c + 1] = iLabel;
			else
			    img_labels_row_fol[c + 1] = 0;
		    }
		    else {
			img_labels_row[c] = 0;
			img_labels_row[c + 1] = 0;
			img_labels_row_fol[c] = 0;
			img_labels_row_fol[c + 1] = 0;
		    }
		}
		// Last column if the number of columns is odd
		if (o_cols) {
		    int iLabel = img_labels_row[c];
		    if (iLabel > 0) {
			iLabel = ET.GetLabel(iLabel);
			if (img_row[c] > 0)
			    img_labels_row[c] = iLabel;
			else
			    img_labels_row[c] = 0;
			if (img_row_fol[c] > 0)
			    img_labels_row_fol[c] = iLabel;
			else
			    img_labels_row_fol[c] = 0;
		    }
		    else {
			img_labels_row[c] = 0;
			img_labels_row_fol[c] = 0;
		    }
		}
	    }
	    // Last row if the number of rows is odd
	    if (o_rows) {
		// Get rows pointer
		const unsigned char* const img_row = img_.ptr<unsigned char>(r);
		unsigned* const img_labels_row = img_labels_.ptr<unsigned>(r);
		int c = 0;
		for (; c < e_cols; c += 2) {
		    int iLabel = img_labels_row[c];
		    if (iLabel > 0) {
			iLabel = ET.GetLabel(iLabel);
			if (img_row[c] > 0)
			    img_labels_row[c] = iLabel;
			else
			    img_labels_row[c] = 0;
			if (img_row[c + 1] > 0)
			    img_labels_row[c + 1] = iLabel;
			else
			    img_labels_row[c + 1] = 0;
		    }
		    else {
			img_labels_row[c] = 0;
			img_labels_row[c + 1] = 0;
		    }
		}
		// Last column if the number of columns is odd
		if (o_cols) {
		    int iLabel = img_labels_row[c];
		    if (iLabel > 0) {
			iLabel = ET.GetLabel(iLabel);
			if (img_row[c] > 0)
			    img_labels_row[c] = iLabel;
			else
			    img_labels_row[c] = 0;
		    }
		    else {
			img_labels_row[c] = 0;
		    }
		}
	    }, StepType::RELABELING, this->perf_, elapsed, this->samplers, size);
    }
};

#endif // !YACCLAB_LABELING_GRANA_2010_H_
