// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file, plus additional authors
// listed below. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional Authors:
// Maximilian Soechting
// Hasso Plattner Institute
// University of Potsdam, Germany

#ifndef YACCLAB_LABELING3D_EPDT_22c_H_
#define YACCLAB_LABELING3D_EPDT_22c_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"
#include "calc_features.hpp"

//Conditions:
#define CONDITION_KB c > 0 && r > 0 && s > 0 && img_slice11_row11[c - 1] > 0
#define CONDITION_LA r > 0 && s > 0 && img_slice11_row11[c] > 0
#define CONDITION_LB c < w - 1 && r > 0 && s > 0 && img_slice11_row11[c + 1] > 0
#define CONDITION_MA c < w - 2 && r > 0 && s > 0 && img_slice11_row11[c + 2] > 0
#define CONDITION_NB c > 0 && s > 0 && img_slice11_row00[c - 1] > 0
#define CONDITION_OA s > 0 && img_slice11_row00[c] > 0
#define CONDITION_OB c < w - 1 && s > 0 && img_slice11_row00[c + 1] > 0
#define CONDITION_PA c < w - 2 && s > 0 && img_slice11_row00[c + 2] > 0
#define CONDITION_QB c > 0 && r < h - 1 && s > 0 && img_slice11_row01[c - 1] > 0
#define CONDITION_RA r < h - 1 && s > 0 && img_slice11_row01[c] > 0
#define CONDITION_RB c < w - 1 && r < h - 1 && s > 0 && img_slice11_row01[c + 1] > 0
#define CONDITION_SA c < w - 2 && r < h - 1 && s > 0 && img_slice11_row01[c + 2] > 0
#define CONDITION_TB c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
#define CONDITION_UA r > 0 && img_slice00_row11[c] > 0
#define CONDITION_UB c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
#define CONDITION_VA c < w - 2 && r > 0 && img_slice00_row11[c + 2] > 0
#define CONDITION_WB c > 0 && img_slice00_row00[c - 1] > 0
#define CONDITION_XA img_slice00_row00[c] > 0
#define CONDITION_XB c < w - 1 && img_slice00_row00[c + 1] > 0
#define CONDITION_NA c > 1 && s > 0 && img_slice11_row00[c - 2] > 0
#define CONDITION_PB c < w - 3 && s > 0 && img_slice11_row00[c + 3] > 0
#define CONDITION_WA c > 1 && img_slice00_row00[c - 2] > 0

//Actions:
#include "labeling3D_EPDT_22c_action_def.inc.h"

template <typename LabelsSolver, bool DoRelabeling = true, typename ConfFeatures = ConfFeatures3DNone>
class EPDT_3D_22c : public Labeling3D<Connectivity3D::CONN_26, ConfFeatures> {
protected:
    LabelsSolver ET;
public:
	EPDT_3D_22c() {}

    bool UseRelabeling() const override {
	return DoRelabeling;
    }
    
	void PerformLabeling()
	{
		ET.Setup(); // Labels solver initialization

		// First scan
		unsigned int d = this->img_.size.p[0];
		unsigned int h = this->img_.size.p[1];
		unsigned int w = this->img_.size.p[2];

		for (unsigned int s = 0; s < d; s += 1) {

			for (unsigned int r = 0; r < h; r += 1) {

				const unsigned char* const img_slice00_row00 = this->img_.template ptr<unsigned char>(s, r);
				// T, W slice
				//const unsigned char* const img_slice00_row12 = (unsigned char *)(((char *)img_slice00_row00) + this->img_.step.p[1] * -2);
				const unsigned char* const img_slice00_row11 = (unsigned char*)(((char*)img_slice00_row00) + this->img_.step.p[1] * -1);
				// img_slice00_row00 defined above
				//const unsigned char* const img_slice00_row01 = (unsigned char *)(((char *)img_slice00_row00) + this->img_.step.p[1] * 1);

				// K, N, Q slice
				//const unsigned char* const img_slice11_row12 = (unsigned char *)(((char *)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * -2);
				const unsigned char* const img_slice11_row11 = (unsigned char*)(((char*)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * -1);
				const unsigned char* const img_slice11_row00 = (unsigned char*)(((char*)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * 0);
				const unsigned char* const img_slice11_row01 = (unsigned char*)(((char*)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * 1);
				//const unsigned char* const img_slice11_row02 = (unsigned char *)(((char *)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * 2);
				//const unsigned char* const img_slice11_row03 = (unsigned char *)(((char *)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * 3);


				// Row pointers for the output image (current slice)
				unsigned* const img_labels_slice00_row00 = this->img_labels_.template ptr<unsigned>(s, r);
				// T, W slice
				//unsigned* const img_labels_slice00_row12 = (unsigned *)(((char *)img_labels_slice00_row00) + this->img_labels_.step.p[1] * -2);
				unsigned* const img_labels_slice00_row11 = (unsigned*)(((char*)img_labels_slice00_row00) + this->img_labels_.step.p[1] * -1);
				// img_labels_slice00_row00 defined above
				//unsigned* const img_labels_slice00_row01 = (unsigned *)(((char *)img_labels_slice00_row00) + this->img_labels_.step.p[1] * 1);

				// K, N, Q slice
				//unsigned* const img_labels_slice11_row12 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * -2);
				unsigned* const img_labels_slice11_row11 = (unsigned*)(((char*)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * -1);
				unsigned* const img_labels_slice11_row00 = (unsigned*)(((char*)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 0);
				unsigned* const img_labels_slice11_row01 = (unsigned*)(((char*)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 1);
				//unsigned* const img_labels_slice11_row02 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 2);
				//unsigned* const img_labels_slice11_row03 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 3);

				// V -- old -- V
				//// Row pointers for the output image (current slice)
				//unsigned* const img_labels_slice00_row00 = this->img_labels_.template ptr<unsigned>(s, r);
				//unsigned* const img_labels_slice00_row12 = (unsigned *)(((char *)img_labels_slice00_row00) + this->img_labels_.step.p[1] * -2);
				//unsigned* const img_labels_slice00_row11 = (unsigned *)(((char *)img_labels_slice00_row00) + this->img_labels_.step.p[1] * -1);
				//unsigned* const img_labels_slice00_row01 = (unsigned *)(((char *)img_labels_slice00_row00) + this->img_labels_.step.p[1] * 1);

				//// Row pointers for the output image (previous slice)
				//unsigned* const img_labels_slice11_row12 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * -2);
				//unsigned* const img_labels_slice11_row11 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * -1);
				//unsigned* const img_labels_slice11_row00 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 0);
				//unsigned* const img_labels_slice11_row01 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 1);
				//unsigned* const img_labels_slice11_row02 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 2);
				for (unsigned int c = 0; c < w; c += 2) {
					if (!((CONDITION_XA) || (CONDITION_XB))) {
						ACTION_0;
					}
#include "labeling3D_EPDT_22c_tree.inc.h"
				}
			} // Rows cycle end
		} // Planes cycle end

		// Second scan
		this->n_labels_ = ET.Flatten();
		
		//unsigned* const img_labels_slice12_row12 = (unsigned *)(((char *)img_labels_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * -2);

		//char * img_labels_row = reinterpret_cast<char*>(this->img_labels_.data);
		//unsigned char * img_row = reinterpret_cast<unsigned char*>(this->img_.data);

		//const unsigned char* const img_row = this->img_.template ptr<unsigned char>(0);
		//unsigned* const img_labels_row = this->img_labels_.template ptr<unsigned>(0);

		//const unsigned char* const img_row = this->img_.template ptr<unsigned char>();
		//int* const img_labels_row = this->img_labels_.template ptr<int>();

		// NEW VERSION BELOW, OLD IN labeling3D_EPDT_19c.h
		int rows = h;
		int e_cols = w & 0xfffffffe;
		bool o_cols = w % 2 == 1;

		for (unsigned s = 0; s < d; s++) {
			int r = 0;
			for (; r < rows; r += 1) {
				// Get rows pointer
				const unsigned char* const img_row = this->img_.template ptr<unsigned char>(s, r);

				unsigned* const img_labels_row = this->img_labels_.template ptr<unsigned>(s, r);
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
		}
		
		CalcFeatures3DPixels::CalcFeatures<LabelsSolver, ConfFeatures, DoRelabeling>(
		    this->img_labels_, ET, this->features);
		
		
	}

	void PerformLabelingWithSteps() {
		double alloc_timing = Alloc();

		int32_t depth = this->img_.size.p[0];
		int32_t height = this->img_.size.p[1];
		int32_t width = this->img_.size.p[2];
		int32_t size = depth * height * width;

		Labeling::StepsDuration elapsed;
		elapsed.Init();
		

		MEASURE_STEP_TIME(FirstScan(),
				  StepType::FIRST_SCAN, this->perf_, elapsed, this->samplers, size);
		
		SecondScan(elapsed);

		
		this->perf_.start();
		this->samplers.Start();
		//CalcFeatures3DBlock_1x1x2::CalcFeatures<LabelsSolver, ConfFeatures, DoRelabeling>(
		//    this->img_, this->img_labels_, ET, this->features);
		CalcFeatures3DPixels::CalcFeatures<LabelsSolver, ConfFeatures>(this->img_labels_, ET, this->features);
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

	void PerformLabelingMem(std::vector<uint64_t>& accesses) {

		{
#undef CONDITION_KB 
#undef CONDITION_LA 
#undef CONDITION_LB 
#undef CONDITION_MA 
#undef CONDITION_NA 
#undef CONDITION_NB 
#undef CONDITION_OA 
#undef CONDITION_OB 
#undef CONDITION_PA 
#undef CONDITION_PB 
#undef CONDITION_QB 
#undef CONDITION_RA 
#undef CONDITION_RB 
#undef CONDITION_SA 
#undef CONDITION_TB 
#undef CONDITION_UA 
#undef CONDITION_UB 
#undef CONDITION_VA 
#undef CONDITION_WA 
#undef CONDITION_WB 
#undef CONDITION_XA 
#undef CONDITION_XB 

#include "labeling3D_EPDT_2829_action_undef.inc.h"

			//Conditions:
#define CONDITION_KB c > 0 && r > 0 && s > 0 && img(s - 1, r - 1, c - 1) > 0
#define CONDITION_LA r > 0 && s > 0 && img(s - 1, r - 1, c) > 0
#define CONDITION_LB c < w - 1 && r > 0 && s > 0 && img(s - 1, r - 1, c + 1) > 0
#define CONDITION_MA c < w - 2 && r > 0 && s > 0 && img(s - 1, r - 1, c + 2) > 0
#define CONDITION_NA c > 1 && s > 0 && img(s - 1, r, c - 2) > 0
#define CONDITION_NB c > 0 && s > 0 && img(s - 1, r, c - 1) > 0
#define CONDITION_OA s > 0 && img(s - 1, r, c) > 0
#define CONDITION_OB c < w - 1 && s > 0 && img(s - 1, r, c + 1) > 0
#define CONDITION_PA c < w - 2 && s > 0 && img(s - 1, r, c + 2) > 0
#define CONDITION_PB c < w - 3 && s > 0 && img(s - 1, r, c + 3) > 0
#define CONDITION_QB c > 0 && r < h - 1 && s > 0 && img(s - 1, r + 1, c - 1) > 0
#define CONDITION_RA r < h - 1 && s > 0 && img(s - 1, r + 1, c) > 0
#define CONDITION_RB c < w - 1 && r < h - 1 && s > 0 && img(s - 1, r + 1, c + 1) > 0
#define CONDITION_SA c < w - 2 && r < h - 1 && s > 0 && img(s - 1, r + 1, c + 2) > 0
#define CONDITION_TB c > 0 && r > 0 && img(s, r - 1, c - 1) > 0
#define CONDITION_UA r > 0 && img(s, r - 1, c) > 0
#define CONDITION_UB c < w - 1 && r > 0 && img(s, r - 1, c + 1) > 0
#define CONDITION_VA c < w - 2 && r > 0 && img(s, r - 1, c + 2) > 0
#define CONDITION_WA c > 1 && img(s, r, c - 2) > 0
#define CONDITION_WB c > 0 && img(s, r, c - 1) > 0
#define CONDITION_XA img(s, r, c) > 0
#define CONDITION_XB c < w - 1 && img(s, r, c + 1) > 0

#include "labeling3D_EPDT_22c_action_def_mem.inc.h"
		}

		ET.MemAlloc(UPPER_BOUND_26_CONNECTIVITY); // Equivalence solver

		MemVol<unsigned char> img(this->img_);
		MemVol<int> img_labels(this->img_.size.p);

		ET.MemSetup();

		// First scan
		unsigned int d = this->img_.size.p[0];
		unsigned int h = this->img_.size.p[1];
		unsigned int w = this->img_.size.p[2];

		for (unsigned int s = 0; s < d; s += 1) {
			for (unsigned int r = 0; r < h; r += 1) {
				for (unsigned int c = 0; c < w; c += 2) {
					if (!((CONDITION_XA) || (CONDITION_XB))) {
						ACTION_0;
					}
#include "labeling3D_EPDT_22c_tree.inc.h"
				}
			} // Rows cycle end
		} // Planes cycle end

		// Second scan
		ET.MemFlatten();

		// NEW VERSION BELOW, OLD IN labeling3D_EPDT_19c.h
		int rows = h;
		int e_cols = w & 0xfffffffe;
		bool o_cols = w % 2 == 1;

		for (unsigned s = 0; s < d; s++) {
			int r = 0;
			for (; r < rows; r += 1) {
				int c = 0;
				for (; c < e_cols; c += 2) {					
					int iLabel = img_labels(s, r, c);
					if (iLabel > 0) {
						iLabel = ET.MemGetLabel(iLabel);
						if (img(s, r, c) > 0)
							img_labels(s, r, c) = iLabel;
						else
							img_labels(s, r, c) = 0;
						if (img(s, r, c + 1) > 0)
							img_labels(s, r, c + 1) = iLabel;
						else
							img_labels(s, r, c + 1) = 0;
					}
					else {
						img_labels(s, r, c) = 0;
						img_labels(s, r, c + 1) = 0;
					}
				}
				// Last column if the number of columns is odd
				if (o_cols) {
					int iLabel = img_labels(s, r, c);
					if (iLabel > 0) {	// Useless controls
						iLabel = ET.MemGetLabel(iLabel);
						if (img(s, r, c) > 0)
							img_labels(s, r, c) = iLabel;
						else
							img_labels(s, r, c) = 0;
					}
					else {
						img_labels(s, r, c) = 0;
					}
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
#undef CONDITION_KB
#undef CONDITION_LA 
#undef CONDITION_LB 
#undef CONDITION_MA 
#undef CONDITION_NA 
#undef CONDITION_NB 
#undef CONDITION_OA 
#undef CONDITION_OB 
#undef CONDITION_PA 
#undef CONDITION_PB 
#undef CONDITION_QB 
#undef CONDITION_RA 
#undef CONDITION_RB 
#undef CONDITION_SA 
#undef CONDITION_TB 
#undef CONDITION_UA 
#undef CONDITION_UB 
#undef CONDITION_VA 
#undef CONDITION_WA 
#undef CONDITION_WB 
#undef CONDITION_XA 
#undef CONDITION_XB

#include "labeling3D_EPDT_2829_action_undef.inc.h"

			//Conditions:
#define CONDITION_KB c > 0 && r > 0 && s > 0 && img_slice11_row11[c - 1] > 0
#define CONDITION_LA r > 0 && s > 0 && img_slice11_row11[c] > 0
#define CONDITION_LB c < w - 1 && r > 0 && s > 0 && img_slice11_row11[c + 1] > 0
#define CONDITION_MA c < w - 2 && r > 0 && s > 0 && img_slice11_row11[c + 2] > 0
#define CONDITION_NB c > 0 && s > 0 && img_slice11_row00[c - 1] > 0
#define CONDITION_OA s > 0 && img_slice11_row00[c] > 0
#define CONDITION_OB c < w - 1 && s > 0 && img_slice11_row00[c + 1] > 0
#define CONDITION_PA c < w - 2 && s > 0 && img_slice11_row00[c + 2] > 0
#define CONDITION_QB c > 0 && r < h - 1 && s > 0 && img_slice11_row01[c - 1] > 0
#define CONDITION_RA r < h - 1 && s > 0 && img_slice11_row01[c] > 0
#define CONDITION_RB c < w - 1 && r < h - 1 && s > 0 && img_slice11_row01[c + 1] > 0
#define CONDITION_SA c < w - 2 && r < h - 1 && s > 0 && img_slice11_row01[c + 2] > 0
#define CONDITION_TB c > 0 && r > 0 && img_slice00_row11[c - 1] > 0
#define CONDITION_UA r > 0 && img_slice00_row11[c] > 0
#define CONDITION_UB c < w - 1 && r > 0 && img_slice00_row11[c + 1] > 0
#define CONDITION_VA c < w - 2 && r > 0 && img_slice00_row11[c + 2] > 0
#define CONDITION_WB c > 0 && img_slice00_row00[c - 1] > 0
#define CONDITION_XA img_slice00_row00[c] > 0
#define CONDITION_XB c < w - 1 && img_slice00_row00[c + 1] > 0
#define CONDITION_NA c > 1 && s > 0 && img_slice11_row00[c - 2] > 0
#define CONDITION_PB c < w - 3 && s > 0 && img_slice11_row00[c + 3] > 0
#define CONDITION_WA c > 1 && img_slice00_row00[c - 2] > 0

//Actions:
#include "labeling3D_EPDT_22c_action_def.inc.h"
		}
	}

private:
	double Alloc()
	{
	    this->samplers.Reset();
	    // Memory allocation of the labels solver
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

		// First scan
		unsigned int d = this->img_.size.p[0];
		unsigned int h = this->img_.size.p[1];
		unsigned int w = this->img_.size.p[2];

		for (unsigned int s = 0; s < d; s += 1) {
			for (unsigned int r = 0; r < h; r += 1) {

				const unsigned char* const img_slice00_row00 = this->img_.template ptr<unsigned char>(s, r);

				// T, W slice
				const unsigned char* const img_slice00_row11 = (unsigned char*)(((char*)img_slice00_row00) + this->img_.step.p[1] * -1);

				// K, N, Q slice
				const unsigned char* const img_slice11_row11 = (unsigned char*)(((char*)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * -1);
				const unsigned char* const img_slice11_row00 = (unsigned char*)(((char*)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * 0);
				const unsigned char* const img_slice11_row01 = (unsigned char*)(((char*)img_slice00_row00) - this->img_.step.p[0] + this->img_.step.p[1] * 1);

				// Row pointers for the output image (current slice)
				unsigned* const img_labels_slice00_row00 = this->img_labels_.template ptr<unsigned>(s, r);
				// T, W slice
				unsigned* const img_labels_slice00_row11 = (unsigned*)(((char*)img_labels_slice00_row00) + this->img_labels_.step.p[1] * -1);

				// K, N, Q slice
				unsigned* const img_labels_slice11_row11 = (unsigned*)(((char*)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * -1);
				unsigned* const img_labels_slice11_row00 = (unsigned*)(((char*)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 0);
				unsigned* const img_labels_slice11_row01 = (unsigned*)(((char*)img_labels_slice00_row00) - this->img_labels_.step.p[0] + this->img_labels_.step.p[1] * 1);

				for (unsigned int c = 0; c < w; c += 2) {
					if (!((CONDITION_XA) || (CONDITION_XB))) {
						ACTION_0;
					}
#include "labeling3D_EPDT_22c_tree.inc.h"
				}
			} // Rows cycle end
		} // Planes cycle end
	}

    void SecondScan(Labeling::StepsDuration& elapsed) {
	    // Second scan


	    unsigned int d = this->img_.size.p[0];
	    unsigned int h = this->img_.size.p[1];
	    unsigned int w = this->img_.size.p[2];
	    unsigned size = d * h * w;

	    MEASURE_STEP_TIME(this->n_labels_ = ET.Flatten(),
			      StepType::TRANSITIVE_CLOSURE, this->perf_, elapsed, this->samplers, size);
	    
	    //const unsigned char* const img_row = this->img_.template ptr<unsigned char>();
	    //int* const img_labels_row = this->img_labels_.template ptr<int>();

	    // NEW VERSION BELOW, OLD COMMENTED IN PerformLabeling
	    MEASURE_STEP_TIME(
		if (DoRelabeling) {
		    int rows = h;
		    int e_cols = w & 0xfffffffe;
		    bool o_cols = w % 2 == 1;

		    for (unsigned s = 0; s < d; s++) {
			int r = 0;
			for (; r < rows; r += 1) {
			    // Get rows pointer
			    const unsigned char* const img_row = this->img_.template ptr<unsigned char>(s, r);
			    unsigned* const img_labels_row = this->img_labels_.template ptr<unsigned>(s, r);
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
				    if (img_row[c] > 0) // Is this check needed ?
					img_labels_row[c] = iLabel;
				    else
					img_labels_row[c] = 0;
				}
				else {
				    img_labels_row[c] = 0;
				}
			    }
			}
		    }
		}, StepType::RELABELING, this->perf_, elapsed, this->samplers, size);
	}
};


#undef CONDITION_KB 
#undef CONDITION_LA 
#undef CONDITION_LB 
#undef CONDITION_MA 
#undef CONDITION_NA 
#undef CONDITION_NB 
#undef CONDITION_OA 
#undef CONDITION_OB 
#undef CONDITION_PA 
#undef CONDITION_PB 
#undef CONDITION_QB 
#undef CONDITION_RA 
#undef CONDITION_RB 
#undef CONDITION_SA 
#undef CONDITION_TB 
#undef CONDITION_UA 
#undef CONDITION_UB 
#undef CONDITION_VA 
#undef CONDITION_WA 
#undef CONDITION_WB 
#undef CONDITION_XA 
#undef CONDITION_XB 

//Actions:
#include "labeling3D_EPDT_2829_action_undef.inc.h"

#endif // YACCLAB_LABELING3D_EPDT_22c_H_
