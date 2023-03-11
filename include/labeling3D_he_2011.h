// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING3D_HE_2011_H_
#define YACCLAB_LABELING3D_HE_2011_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"
#include "calc_features.hpp"


//Conditions:
#define CONDITION_V   img_slice00_row00[c] > 0
#define CONDITION_V1  c > 0 && img_slice00_row00[c - 1] > 0
#define CONDITION_V2  c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
#define CONDITION_V3  r > 0 && img_slice00_row11[c] > 0
#define CONDITION_V4  c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
#define CONDITION_V5  c > 0 && r > 0 && s > 0 && img_slice11_row11[c - 1] > 0
#define CONDITION_V6  r > 0 && s > 0 && img_slice11_row11[c] > 0
#define CONDITION_V7  c < w - 1 && r > 0 && s > 0 && img_slice11_row11[c + 1] > 0
#define CONDITION_V8  c > 0 && s > 0 && img_slice11_row00[c - 1] > 0
#define CONDITION_V9  s > 0 && img_slice11_row00[c] > 0
#define CONDITION_V10 c < w - 1 && s > 0 && img_slice11_row00[c + 1] > 0
#define CONDITION_V11 c > 0 && r < h - 1 && s > 0 && img_slice11_row01[c - 1] > 0
#define CONDITION_V12 r < h - 1 && s > 0 && img_slice11_row01[c] > 0
#define CONDITION_V13 c < w - 1 && r < h - 1 && s > 0 && img_slice11_row01[c + 1] > 0

// nothing
#define ACTION_1 img_labels_slice00_row00[c] = 0; 
// v <- v9
#define ACTION_2 img_labels_slice00_row00[c] = img_labels_slice11_row00[c];
// v <- v3 
#define ACTION_3 img_labels_slice00_row00[c] = img_labels_slice00_row11[c];
// merge(V3, v12)
#define ACTION_4 ET.Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c]);
// merge(V3, v11)
#define ACTION_5 ET.Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c - 1]);
// merge(V3, v13)
#define ACTION_6 ET.Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c + 1]);
// v <- v6
#define ACTION_7 img_labels_slice00_row00[c] = img_labels_slice11_row11[c];
// v <- v1
#define ACTION_8 img_labels_slice00_row00[c] = img_labels_slice00_row00[c - 1]; 
// merge(v1, v10)
#define ACTION_9 ET.Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row00[c + 1]); 
// merge(v1, v4)
#define ACTION_10 ET.Merge(img_labels_slice00_row00[c - 1], img_labels_slice00_row11[c + 1]);
// merge(v1, v7)
#define ACTION_11 ET.Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v1, v13)
#define ACTION_12 ET.Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row01[c + 1]);
// v <- v8
#define ACTION_13 img_labels_slice00_row00[c] = img_labels_slice11_row00[c - 1];
// v <- v10
#define ACTION_14 img_labels_slice00_row00[c] = img_labels_slice11_row00[c + 1];
// merge(v2, v10)
#define ACTION_15 ET.Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row00[c + 1]);
// merge(v5, v10)
#define ACTION_16 ET.Merge(img_labels_slice11_row00[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v11, v10)
#define ACTION_17 ET.Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row00[c + 1]);
// v <- v2
#define ACTION_18 img_labels_slice00_row00[c] = img_labels_slice00_row11[c - 1];
// merge(v4, v2)
#define ACTION_19 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice00_row11[c - 1]);
// merge(v7, v2)
#define ACTION_20 ET.Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v2, V12)
#define ACTION_21 ET.Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c]);
// merge(v2, V11)
#define ACTION_22 ET.Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c - 1]);
// merge(v2, V13)
#define ACTION_23 ET.Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c + 1]);
// merge(v6, V12)
#define ACTION_24 ET.Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c]);
// merge(v6, V11)
#define ACTION_25 ET.Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c]);
// merge(v6, V13)
#define ACTION_26 ET.Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c]);
// merge(v8, v10)
#define ACTION_27 ET.Merge(img_labels_slice11_row00[c + 1], img_labels_slice11_row00[c - 1]);
// merge(v8, v4)
#define ACTION_28 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row00[c - 1]);
// merge(v8, v7)
#define ACTION_29 ET.Merge(img_labels_slice11_row00[c - 1], img_labels_slice11_row11[c + 1]);
// v <- v5
#define ACTION_30 img_labels_slice00_row00[c] = img_labels_slice11_row11[c - 1];
// merge(v4, v5)
#define ACTION_31 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v7, v5)
#define ACTION_32 ET.Merge(img_labels_slice11_row11[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v5, V12)
#define ACTION_33 ET.Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c - 1]);
// merge(v5, V11)
#define ACTION_34 ET.Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c - 1]);
// merge(v5, V13)
#define ACTION_35 ET.Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c - 1]);
// v <- v12
#define ACTION_36 img_labels_slice00_row00[c] = img_labels_slice11_row01[c];
// merge(v12, v4)
#define ACTION_37 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c]);
// merge(v12, v7)
#define ACTION_38 ET.Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c + 1]);
// v <- v4
#define ACTION_39 img_labels_slice00_row00[c] = img_labels_slice00_row11[c + 1];
// merge(v11, v4)
#define ACTION_40 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c - 1]);
// merge(v13, v4)
#define ACTION_41 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c + 1]);
// v <- v7
#define ACTION_42 img_labels_slice00_row00[c] = img_labels_slice11_row11[c + 1];
// merge(v11, v7)
#define ACTION_43 ET.Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v13, v7)
#define ACTION_44 ET.Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c + 1]);
// v <- v11
#define ACTION_45 img_labels_slice00_row00[c] = img_labels_slice11_row01[c - 1];
// merge(v13, v11)
#define ACTION_46 ET.Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row01[c - 1]);
// v <- v13
#define ACTION_47 img_labels_slice00_row00[c] = img_labels_slice11_row01[c + 1];
// v <- newlabel
#define ACTION_48 img_labels_slice00_row00[c] = ET.NewLabel();
// merge(v8, v13)
#define ACTION_49 ET.Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row00[c - 1]);


template <typename LabelsSolver, bool DoRelabeling = true, typename ConfFeatures = ConfFeatures3DNone>
class LEB_3D : public Labeling3D<Connectivity3D::CONN_26, ConfFeatures> {
protected:
    LabelsSolver ET;
public:
    LEB_3D() {}
    
    bool UseRelabeling() const override {
	return DoRelabeling;
    }
    
    void PerformLabeling()
	{
	    Labeling::StepsDuration elapsed;
	    elapsed.Init();
	    
	    FirstScan();
	    SecondScan(elapsed);

	    CalcFeatures3DPixels::CalcFeatures<LabelsSolver, ConfFeatures, DoRelabeling>(
		this->img_labels_, ET, this->features);
	}

	void PerformLabelingWithSteps() {
	    Labeling::StepsDuration elapsed;
	    elapsed.Init();

	    
	    double alloc_timing = Alloc();

	    int32_t depth = this->img_.size.p[0];
	    int32_t height = this->img_.size.p[1];
	    int32_t width = this->img_.size.p[2];
	    int32_t size = depth * height * width;

	    MEASURE_STEP_TIME(FirstScan(), StepType::FIRST_SCAN, this->perf_, elapsed,
			      this->samplers, size);
       	    
	    SecondScan(elapsed);

	    this->perf_.start();
	    this->samplers.Start();
	    CalcFeatures3DPixels::CalcFeatures<LabelsSolver, ConfFeatures, DoRelabeling>(
		this->img_labels_, ET, this->features);
	    this->samplers.Stop();
	    this->perf_.stop();
	    this->samplers.Store(StepType::FEATURES, size);
	    elapsed.duration[StepType::FEATURES] = this->perf_.last();
	    
	    this->perf_.start();	    
	    Dealloc();
	    this->perf_.stop();
	    
	    this->perf_.store(Step(StepType::ALLOC_DEALLOC), this->perf_.last() + alloc_timing);


	    elapsed.CalcDerivedTime();
	    elapsed.StoreAll(this->perf_);
	    this->samplers.CalcDerived();
	}

	void PerformLabelingMem(std::vector<uint64_t>& accesses)
	{

		{
#undef CONDITION_V  
#undef CONDITION_V1 
#undef CONDITION_V2 
#undef CONDITION_V3 
#undef CONDITION_V4 
#undef CONDITION_V5 
#undef CONDITION_V6 
#undef CONDITION_V7 
#undef CONDITION_V8 
#undef CONDITION_V9 
#undef CONDITION_V10
#undef CONDITION_V11
#undef CONDITION_V12
#undef CONDITION_V13

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
#undef ACTION_17
#undef ACTION_18
#undef ACTION_19
#undef ACTION_20
#undef ACTION_21
#undef ACTION_22
#undef ACTION_23
#undef ACTION_24
#undef ACTION_25
#undef ACTION_26
#undef ACTION_27
#undef ACTION_28
#undef ACTION_29
#undef ACTION_30
#undef ACTION_31
#undef ACTION_32
#undef ACTION_33
#undef ACTION_34
#undef ACTION_35
#undef ACTION_36
#undef ACTION_37
#undef ACTION_38
#undef ACTION_39
#undef ACTION_40
#undef ACTION_41
#undef ACTION_42
#undef ACTION_43
#undef ACTION_44
#undef ACTION_45
#undef ACTION_46
#undef ACTION_47
#undef ACTION_48
#undef ACTION_49

//Conditions:
#define CONDITION_V   img(s, r, c) > 0
#define CONDITION_V1  c > 0 && img(s, r, c - 1) > 0
#define CONDITION_V2  c > 0 && r > 0 && img(s, r - 1, c - 1) > 0
#define CONDITION_V3  r > 0 && img(s, r - 1, c) > 0
#define CONDITION_V4  c < w - 1 && r > 0 && img(s, r - 1, c + 1) > 0
#define CONDITION_V5  c > 0 && r > 0 && s > 0 && img(s - 1, r - 1, c - 1) > 0
#define CONDITION_V6  r > 0 && s > 0 && img(s - 1, r - 1, c) > 0
#define CONDITION_V7  c < w - 1 && r > 0 && s > 0 && img(s - 1, r - 1, c + 1) > 0
#define CONDITION_V8  c > 0 && s > 0 && img(s - 1, r, c - 1) > 0
#define CONDITION_V9  s > 0 && img(s - 1, r, c) > 0
#define CONDITION_V10 c < w - 1 && s > 0 && img(s - 1, r, c + 1) > 0
#define CONDITION_V11 c > 0 && r < h - 1 && s > 0 && img(s - 1, r + 1, c - 1) > 0
#define CONDITION_V12 r < h - 1 && s > 0 && img(s - 1, r + 1, c) > 0
#define CONDITION_V13 c < w - 1 && r < h - 1 && s > 0 && img(s - 1, r + 1, c + 1) > 0

// nothing
#define ACTION_1 img_labels(s, r, c) = 0; 
// v <- v9
#define ACTION_2 img_labels(s, r, c) = img_labels(s - 1, r, c);
// v <- v3 
#define ACTION_3 img_labels(s, r, c) = img_labels(s, r - 1, c);
// merge(V3, v12)
#define ACTION_4 ET.MemMerge(img_labels(s, r - 1, c), img_labels(s - 1, r + 1, c));
// merge(V3, v11)
#define ACTION_5 ET.MemMerge(img_labels(s, r - 1, c), img_labels(s - 1, r + 1, c - 1));
// merge(V3, v13)
#define ACTION_6 ET.MemMerge(img_labels(s, r - 1, c), img_labels(s - 1, r + 1, c + 1));
// v <- v6
#define ACTION_7 img_labels(s, r, c) = img_labels(s - 1, r - 1, c);
// v <- v1
#define ACTION_8 img_labels(s, r, c) = img_labels(s, r, c - 1); 
// merge(v1, v10)
#define ACTION_9 ET.MemMerge(img_labels(s, r, c - 1), img_labels(s - 1, r, c + 1)); 
// merge(v1, v4)
#define ACTION_10 ET.MemMerge(img_labels(s, r, c - 1), img_labels(s, r - 1, c + 1));
// merge(v1, v7)
#define ACTION_11 ET.MemMerge(img_labels(s, r, c - 1), img_labels(s - 1, r - 1, c + 1));
// merge(v1, v13)
#define ACTION_12 ET.MemMerge(img_labels(s, r, c - 1), img_labels(s - 1, r + 1, c + 1));
// v <- v8
#define ACTION_13 img_labels(s, r, c) = img_labels(s - 1, r, c - 1);
// v <- v10
#define ACTION_14 img_labels(s, r, c) = img_labels(s - 1, r, c + 1);
// merge(v2, v10)
#define ACTION_15 ET.MemMerge(img_labels(s, r - 1, c - 1), img_labels(s - 1, r, c + 1));
// merge(v5, v10)
#define ACTION_16 ET.MemMerge(img_labels(s - 1, r, c + 1), img_labels(s - 1, r - 1, c - 1));
// merge(v11, v10)
#define ACTION_17 ET.MemMerge(img_labels(s - 1, r + 1, c - 1), img_labels(s - 1, r, c + 1));
// v <- v2
#define ACTION_18 img_labels(s, r, c) = img_labels(s, r - 1, c - 1);
// merge(v4, v2)
#define ACTION_19 ET.MemMerge(img_labels(s, r - 1, c + 1), img_labels(s, r - 1, c - 1));
// merge(v7, v2)
#define ACTION_20 ET.MemMerge(img_labels(s, r - 1, c - 1), img_labels(s - 1, r - 1, c + 1));
// merge(v2, V12)
#define ACTION_21 ET.MemMerge(img_labels(s, r - 1, c - 1), img_labels(s - 1, r + 1, c));
// merge(v2, V11)
#define ACTION_22 ET.MemMerge(img_labels(s, r - 1, c - 1), img_labels(s - 1, r + 1, c - 1));
// merge(v2, V13)
#define ACTION_23 ET.MemMerge(img_labels(s, r - 1, c - 1), img_labels(s - 1, r + 1, c + 1));
// merge(v6, V12)
#define ACTION_24 ET.MemMerge(img_labels(s - 1, r + 1, c), img_labels(s - 1, r - 1, c));
// merge(v6, V11)
#define ACTION_25 ET.MemMerge(img_labels(s - 1, r + 1, c - 1), img_labels(s - 1, r - 1, c));
// merge(v6, V13)
#define ACTION_26 ET.MemMerge(img_labels(s - 1, r + 1, c + 1), img_labels(s - 1, r - 1, c));
// merge(v8, v10)
#define ACTION_27 ET.MemMerge(img_labels(s - 1, r, c + 1), img_labels(s - 1, r, c - 1));
// merge(v8, v4)
#define ACTION_28 ET.MemMerge(img_labels(s, r - 1, c + 1), img_labels(s - 1, r, c - 1));
// merge(v8, v7)
#define ACTION_29 ET.MemMerge(img_labels(s - 1, r, c - 1), img_labels(s - 1, r - 1, c + 1));
// v <- v5
#define ACTION_30 img_labels(s, r, c) = img_labels(s - 1, r - 1, c - 1);
// merge(v4, v5)
#define ACTION_31 ET.MemMerge(img_labels(s, r - 1, c + 1), img_labels(s - 1, r - 1, c - 1));
// merge(v7, v5)
#define ACTION_32 ET.MemMerge(img_labels(s - 1, r - 1, c + 1), img_labels(s - 1, r - 1, c - 1));
// merge(v5, V12)
#define ACTION_33 ET.MemMerge(img_labels(s - 1, r + 1, c), img_labels(s - 1, r - 1, c - 1));
// merge(v5, V11)
#define ACTION_34 ET.MemMerge(img_labels(s - 1, r + 1, c - 1), img_labels(s - 1, r - 1, c - 1));
// merge(v5, V13)
#define ACTION_35 ET.MemMerge(img_labels(s - 1, r + 1, c + 1), img_labels(s - 1, r - 1, c - 1));
// v <- v12
#define ACTION_36 img_labels(s, r, c) = img_labels(s - 1, r + 1, c);
// merge(v12, v4)
#define ACTION_37 ET.MemMerge(img_labels(s, r - 1, c + 1), img_labels(s - 1, r + 1, c));
// merge(v12, v7)
#define ACTION_38 ET.MemMerge(img_labels(s - 1, r + 1, c), img_labels(s - 1, r - 1, c + 1));
// v <- v4
#define ACTION_39 img_labels(s, r, c) = img_labels(s, r - 1, c + 1);
// merge(v11, v4)
#define ACTION_40 ET.MemMerge(img_labels(s, r - 1, c + 1), img_labels(s - 1, r + 1, c - 1));
// merge(v13, v4)
#define ACTION_41 ET.MemMerge(img_labels(s, r - 1, c + 1), img_labels(s - 1, r + 1, c + 1));
// v <- v7
#define ACTION_42 img_labels(s, r, c) = img_labels(s - 1, r - 1, c + 1);
// merge(v11, v7)
#define ACTION_43 ET.MemMerge(img_labels(s - 1, r + 1, c - 1), img_labels(s - 1, r - 1, c + 1));
// merge(v13, v7)
#define ACTION_44 ET.MemMerge(img_labels(s - 1, r + 1, c + 1), img_labels(s - 1, r - 1, c + 1));
// v <- v11
#define ACTION_45 img_labels(s, r, c) = img_labels(s - 1, r + 1, c - 1);
// merge(v13, v11)
#define ACTION_46 ET.MemMerge(img_labels(s - 1, r + 1, c + 1), img_labels(s - 1, r + 1, c - 1));
// v <- v13
#define ACTION_47 img_labels(s, r, c) = img_labels(s - 1, r + 1, c + 1);
// v <- newlabel
#define ACTION_48 img_labels(s, r, c) = ET.MemNewLabel();
// merge(v8, v13)
#define ACTION_49 ET.MemMerge(img_labels(s - 1, r + 1, c + 1), img_labels(s - 1, r, c - 1));
		}

		ET.MemAlloc(UPPER_BOUND_26_CONNECTIVITY); // Equivalence solver

		MemVol<unsigned char> img(this->img_);
		MemVol<int> img_labels(this->img_.size.p);

		ET.MemSetup();

        //uint64_t accesses_count = 0;

		// First scan
		unsigned int d = this->img_.size.p[0];
		unsigned int h = this->img_.size.p[1];
		unsigned int w = this->img_.size.p[2];

		for (unsigned int s = 0; s < d; s++) {
			for (unsigned int r = 0; r < h; r++) {
				for (unsigned int c = 0; c < w; c++) {

#include "labeling3D_he_2011_tree.inc.h"
				}
			} // Rows cycle end
		} // Planes cycle end

		// Second scan
		ET.MemFlatten();

		for (unsigned int s = 0; s < d; s++) {
			for (unsigned int r = 0; r < h; r++) {
				for (unsigned int c = 0; c < w; c++) {
					img_labels(s, r, c) = ET.MemGetLabel(img_labels(s, r, c));
				}
			}
		}

		// Store total accesses in the output vector 'accesses'
		accesses = std::vector<uint64_t>((int)MD_SIZE, 0);

		accesses[MD_BINARY_MAT] = (unsigned long)img.GetTotalAccesses();
		accesses[MD_LABELED_MAT] = (unsigned long)img_labels.GetTotalAccesses();
		accesses[MD_EQUIVALENCE_VEC] = (unsigned long)ET.MemTotalAccesses();

		this->img_labels_ = img_labels.GetImage();

		ET.MemDealloc(); // Memory deallocation of the labels solver

		{
#undef CONDITION_V  
#undef CONDITION_V1 
#undef CONDITION_V2 
#undef CONDITION_V3 
#undef CONDITION_V4 
#undef CONDITION_V5 
#undef CONDITION_V6 
#undef CONDITION_V7 
#undef CONDITION_V8 
#undef CONDITION_V9 
#undef CONDITION_V10
#undef CONDITION_V11
#undef CONDITION_V12
#undef CONDITION_V13

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
#undef ACTION_17
#undef ACTION_18
#undef ACTION_19
#undef ACTION_20
#undef ACTION_21
#undef ACTION_22
#undef ACTION_23
#undef ACTION_24
#undef ACTION_25
#undef ACTION_26
#undef ACTION_27
#undef ACTION_28
#undef ACTION_29
#undef ACTION_30
#undef ACTION_31
#undef ACTION_32
#undef ACTION_33
#undef ACTION_34
#undef ACTION_35
#undef ACTION_36
#undef ACTION_37
#undef ACTION_38
#undef ACTION_39
#undef ACTION_40
#undef ACTION_41
#undef ACTION_42
#undef ACTION_43
#undef ACTION_44
#undef ACTION_45
#undef ACTION_46
#undef ACTION_47
#undef ACTION_48
#undef ACTION_49

			//Conditions:
#define CONDITION_V   img_slice00_row00[c] > 0
#define CONDITION_V1  c > 0 && img_slice00_row00[c - 1] > 0
#define CONDITION_V2  c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
#define CONDITION_V3  r > 0 && img_slice00_row11[c] > 0
#define CONDITION_V4  c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
#define CONDITION_V5  c > 0 && r > 0 && s > 0 && img_slice11_row11[c - 1] > 0
#define CONDITION_V6  r > 0 && s > 0 && img_slice11_row11[c] > 0
#define CONDITION_V7  c < w - 1 && r > 0 && s > 0 && img_slice11_row11[c + 1] > 0
#define CONDITION_V8  c > 0 && s > 0 && img_slice11_row00[c - 1] > 0
#define CONDITION_V9  s > 0 && img_slice11_row00[c] > 0
#define CONDITION_V10 c < w - 1 && s > 0 && img_slice11_row00[c + 1] > 0
#define CONDITION_V11 c > 0 && r < h - 1 && s > 0 && img_slice11_row01[c - 1] > 0
#define CONDITION_V12 r < h - 1 && s > 0 && img_slice11_row01[c] > 0
#define CONDITION_V13 c < w - 1 && r < h - 1 && s > 0 && img_slice11_row01[c + 1] > 0

// nothing
#define ACTION_1 img_labels_slice00_row00[c] = 0; 
// v <- v9
#define ACTION_2 img_labels_slice00_row00[c] = img_labels_slice11_row00[c];
// v <- v3 
#define ACTION_3 img_labels_slice00_row00[c] = img_labels_slice00_row11[c];
// merge(V3, v12)
#define ACTION_4 ET.Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c]);
// merge(V3, v11)
#define ACTION_5 ET.Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c - 1]);
// merge(V3, v13)
#define ACTION_6 ET.Merge(img_labels_slice00_row11[c], img_labels_slice11_row01[c + 1]);
// v <- v6
#define ACTION_7 img_labels_slice00_row00[c] = img_labels_slice11_row11[c];
// v <- v1
#define ACTION_8 img_labels_slice00_row00[c] = img_labels_slice00_row00[c - 1]; 
// merge(v1, v10)
#define ACTION_9 ET.Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row00[c + 1]); 
// merge(v1, v4)
#define ACTION_10 ET.Merge(img_labels_slice00_row00[c - 1], img_labels_slice00_row11[c + 1]);
// merge(v1, v7)
#define ACTION_11 ET.Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v1, v13)
#define ACTION_12 ET.Merge(img_labels_slice00_row00[c - 1], img_labels_slice11_row01[c + 1]);
// v <- v8
#define ACTION_13 img_labels_slice00_row00[c] = img_labels_slice11_row00[c - 1];
// v <- v10
#define ACTION_14 img_labels_slice00_row00[c] = img_labels_slice11_row00[c + 1];
// merge(v2, v10)
#define ACTION_15 ET.Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row00[c + 1]);
// merge(v5, v10)
#define ACTION_16 ET.Merge(img_labels_slice11_row00[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v11, v10)
#define ACTION_17 ET.Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row00[c + 1]);
// v <- v2
#define ACTION_18 img_labels_slice00_row00[c] = img_labels_slice00_row11[c - 1];
// merge(v4, v2)
#define ACTION_19 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice00_row11[c - 1]);
// merge(v7, v2)
#define ACTION_20 ET.Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v2, V12)
#define ACTION_21 ET.Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c]);
// merge(v2, V11)
#define ACTION_22 ET.Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c - 1]);
// merge(v2, V13)
#define ACTION_23 ET.Merge(img_labels_slice00_row11[c - 1], img_labels_slice11_row01[c + 1]);
// merge(v6, V12)
#define ACTION_24 ET.Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c]);
// merge(v6, V11)
#define ACTION_25 ET.Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c]);
// merge(v6, V13)
#define ACTION_26 ET.Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c]);
// merge(v8, v10)
#define ACTION_27 ET.Merge(img_labels_slice11_row00[c + 1], img_labels_slice11_row00[c - 1]);
// merge(v8, v4)
#define ACTION_28 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row00[c - 1]);
// merge(v8, v7)
#define ACTION_29 ET.Merge(img_labels_slice11_row00[c - 1], img_labels_slice11_row11[c + 1]);
// v <- v5
#define ACTION_30 img_labels_slice00_row00[c] = img_labels_slice11_row11[c - 1];
// merge(v4, v5)
#define ACTION_31 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v7, v5)
#define ACTION_32 ET.Merge(img_labels_slice11_row11[c + 1], img_labels_slice11_row11[c - 1]);
// merge(v5, V12)
#define ACTION_33 ET.Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c - 1]);
// merge(v5, V11)
#define ACTION_34 ET.Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c - 1]);
// merge(v5, V13)
#define ACTION_35 ET.Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c - 1]);
// v <- v12
#define ACTION_36 img_labels_slice00_row00[c] = img_labels_slice11_row01[c];
// merge(v12, v4)
#define ACTION_37 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c]);
// merge(v12, v7)
#define ACTION_38 ET.Merge(img_labels_slice11_row01[c], img_labels_slice11_row11[c + 1]);
// v <- v4
#define ACTION_39 img_labels_slice00_row00[c] = img_labels_slice00_row11[c + 1];
// merge(v11, v4)
#define ACTION_40 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c - 1]);
// merge(v13, v4)
#define ACTION_41 ET.Merge(img_labels_slice00_row11[c + 1], img_labels_slice11_row01[c + 1]);
// v <- v7
#define ACTION_42 img_labels_slice00_row00[c] = img_labels_slice11_row11[c + 1];
// merge(v11, v7)
#define ACTION_43 ET.Merge(img_labels_slice11_row01[c - 1], img_labels_slice11_row11[c + 1]);
// merge(v13, v7)
#define ACTION_44 ET.Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row11[c + 1]);
// v <- v11
#define ACTION_45 img_labels_slice00_row00[c] = img_labels_slice11_row01[c - 1];
// merge(v13, v11)
#define ACTION_46 ET.Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row01[c - 1]);
// v <- v13
#define ACTION_47 img_labels_slice00_row00[c] = img_labels_slice11_row01[c + 1];
// v <- newlabel
#define ACTION_48 img_labels_slice00_row00[c] = ET.NewLabel();
// merge(v8, v13)
#define ACTION_49 ET.Merge(img_labels_slice11_row01[c + 1], img_labels_slice11_row00[c - 1]);
		}

	}


private:
	double Alloc()
	{
		// Memory allocation of the labels solver
	    Dealloc();
	    this->samplers.Reset();
	    
	    double ls_t = ET.Alloc(UPPER_BOUND_26_CONNECTIVITY, this->perf_);
	    // Memory allocation for the output image
	    this->perf_.start();
	    this->img_labels_.create(3, this->img_.size.p);
	    memset(this->img_labels_.data, 0, this->img_labels_.dataend - this->img_labels_.datastart);
	    this->features.template Alloc<ConfFeatures>(UPPER_BOUND_26_CONNECTIVITY);
	    this->perf_.stop();
	    double t = this->perf_.last();

	    
	    this->perf_.start();
	    memset(this->img_labels_.data, 0, this->img_labels_.dataend - this->img_labels_.datastart);
	    this->features.template Touch<ConfFeatures>();
	    this->perf_.stop();
	    double ma_t = t - this->perf_.last();
	    // Return total time
	    return ls_t + ma_t;
	}
    void Dealloc() {
	ET.Dealloc();
	// No free for img_labels_ because it is required at the end of the algorithm 
    }
	void FirstScan() {
		ET.Setup(); // Labels solver initialization
		
		unsigned int d = this->img_.size.p[0];
		unsigned int h = this->img_.size.p[1];
		unsigned int w = this->img_.size.p[2];

		for (unsigned int s = 0; s < d; s++) {

			for (unsigned int r = 0; r < h; r++) {

				// Row pointers for the input image (current slice)
				const unsigned char* const img_slice00_row00 = this->img_.template ptr<unsigned char>(s, r);
				const unsigned char* const img_slice00_row11 = (unsigned char *)(((char *)img_slice00_row00) + this->img_.step.p[1] * -1);

				// Row pointers for the input image (previous slice)
				const unsigned char* const img_slice11_row11 = (unsigned char *)(((char *)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * -1);
				const unsigned char* const img_slice11_row00 = (unsigned char *)(((char *)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * 0);
				const unsigned char* const img_slice11_row01 = (unsigned char *)(((char *)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * 1);

				// Row pointers for the output image (current slice)
				unsigned* const img_labels_slice00_row00 = this->img_labels_.template ptr<unsigned>(s, r);
				unsigned* const img_labels_slice00_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + this->img_labels_.step.p[1] * -1);

				// Row pointers for the output image (previous slice)
				unsigned* const img_labels_slice11_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * -1);
				unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 0);
				unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 1);

				for (unsigned int c = 0; c < w; c++) {

#include "labeling3D_he_2011_tree.inc.h"

				}
			} // Rows cycle end
		} // Planes cycle end
	}
		

    void SecondScan(Labeling::StepsDuration& elapsed) {

	unsigned int d = this->img_.size.p[0];
	unsigned int h = this->img_.size.p[1];
	unsigned int w = this->img_.size.p[2];

	unsigned size = d * h * w;
	
	MEASURE_STEP_TIME(
	    this->n_labels_ = ET.Flatten(),
	    StepType::TRANSITIVE_CLOSURE, this->perf_, elapsed, this->samplers, size);

	MEASURE_STEP_TIME(
	    if (DoRelabeling) {
		int * img_row = reinterpret_cast<int*>(this->img_labels_.data);
		for (unsigned int s = 0; s < d; s++) {
		    for (unsigned int r = 0; r < h; r++) {
			for (unsigned int c = 0; c < w; c++) {
			    img_row[c] = ET.GetLabel(img_row[c]);
			}
			img_row += this->img_labels_.step.p[1] / sizeof(int);
		    }
		}, StepType::RELABELING, this->perf_, elapsed, this->samplers, size);
	    }
    }
};


#undef CONDITION_V  
#undef CONDITION_V1 
#undef CONDITION_V2 
#undef CONDITION_V3 
#undef CONDITION_V4 
#undef CONDITION_V5 
#undef CONDITION_V6 
#undef CONDITION_V7 
#undef CONDITION_V8 
#undef CONDITION_V9 
#undef CONDITION_V10
#undef CONDITION_V11
#undef CONDITION_V12
#undef CONDITION_V13

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
#undef ACTION_17
#undef ACTION_18
#undef ACTION_19
#undef ACTION_20
#undef ACTION_21
#undef ACTION_22
#undef ACTION_23
#undef ACTION_24
#undef ACTION_25
#undef ACTION_26
#undef ACTION_27
#undef ACTION_28
#undef ACTION_29
#undef ACTION_30
#undef ACTION_31
#undef ACTION_32
#undef ACTION_33
#undef ACTION_34
#undef ACTION_35
#undef ACTION_36
#undef ACTION_37
#undef ACTION_38
#undef ACTION_39
#undef ACTION_40
#undef ACTION_41
#undef ACTION_42
#undef ACTION_43
#undef ACTION_44
#undef ACTION_45
#undef ACTION_46
#undef ACTION_47
#undef ACTION_48
#undef ACTION_49

#endif // YACCLAB_LABELING3D_HE_2011_H_
