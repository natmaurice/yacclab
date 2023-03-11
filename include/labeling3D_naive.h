// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING_NAIVE_3D_H_
#define YACCLAB_LABELING_NAIVE_3D_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"
#include "calc_features.hpp"

template <typename LabelsSolver, typename ConfFeatures = ConfFeatures3DNone>
class naive_3D : public Labeling3D<Connectivity3D::CONN_26, ConfFeatures> {
protected:
    LabelsSolver ET;
public:
	naive_3D() {}

	void PerformLabeling()
	{
		//img_labels_ = cv::Mat1i(this->img_.size(), 0); // Allocation + initialization of the output image
		this->img_labels_.create(3, this->img_.size.p);

		ET.Alloc(UPPER_BOUND_26_CONNECTIVITY); // Memory allocation of the labels solver
		this->features.template Alloc<ConfFeatures>(UPPER_BOUND_26_CONNECTIVITY);
		ET.Setup(); // Labels solver initialization

		// Rosenfeld Mask
		// +-+-+-+
		// |p|q|r|
		// +-+-+-+
		// |s|x|
		// +-+-+

		// First scan
		for (int z = 0; z < this->img_.size[0]; z++) {

			unsigned char const * const img_plane = this->img_.data + this->img_.step[0] * z;   //   this->img_.template ptr<unsigned char>(z, 0, 0);
			unsigned char const * const img_prev_plane = (z > 0) ? (img_plane - this->img_.step[0]) : nullptr;
			int * const labels_plane = reinterpret_cast<int*>(this->img_labels_.data) + (this->img_labels_.step[0] / sizeof(int)) * z;
			int * const labels_prev_plane = labels_plane - (this->img_labels_.step[0] / sizeof(int));

			for (int y = 0; y < this->img_.size[1]; y++) {

				// Prev plane row pointers
				unsigned char const * img_prev_plane_rows[3];
				int prev_plane_first_row, prev_plane_last_row;
				if (img_prev_plane != nullptr) {
					img_prev_plane_rows[1] = img_prev_plane + this->img_.step[1] * y;
					img_prev_plane_rows[0] = (y > 0) ? (prev_plane_first_row = 0, img_prev_plane_rows[1] - this->img_.step[1]) : (prev_plane_first_row = 1, nullptr);
					img_prev_plane_rows[2] = (y + 1 < this->img_.size[1]) ? (prev_plane_last_row = 2, img_prev_plane_rows[1] + this->img_.step[1]) : (prev_plane_last_row = 1, nullptr);
				}

				int * labels_prev_plane_rows[3];
				labels_prev_plane_rows[1] = labels_prev_plane + (this->img_labels_.step[1] / sizeof(int)) * y;
				labels_prev_plane_rows[0] = labels_prev_plane_rows[1] - (this->img_labels_.step[1] / sizeof(int));
				labels_prev_plane_rows[2] = labels_prev_plane_rows[1] + (this->img_labels_.step[1] / sizeof(int));

				// Cur plane row pointers
				unsigned char const * const img_row = img_plane + this->img_.step[1] * y;
				unsigned char const * const img_prev_row = (y > 0) ? (img_row - this->img_.step[1]) : nullptr;
				int * const labels_row = labels_plane + (this->img_labels_.step[1] / sizeof(int)) * y;
				int * const labels_prev_row = labels_row - (this->img_labels_.step[1] / sizeof(int));

				for (int x = 0; x < this->img_.size[2]; x++) {
					int label = 0;
					if (img_row[x] > 0) {

						int const first_neighbour_x = (x > 0) ? (x - 1) : x;
						int const last_neighbour_x = (x + 1 < this->img_.size[2]) ? (x + 1) : x;

						// Previous plane
						if (img_prev_plane != nullptr) {
							for (int r = prev_plane_first_row; r <= prev_plane_last_row; r++) {
								for (int c = first_neighbour_x; c <= last_neighbour_x; c++) {
									if (img_prev_plane_rows[r][c] > 0) {
										if (label == 0) {
											label = labels_prev_plane_rows[r][c];
										}
										else {
											ET.Merge(labels_prev_plane_rows[r][c], label);
										}
									}
								}
							}
						}

						// Previous row
						if (img_prev_row != nullptr) {
							for (int c = first_neighbour_x; c <= last_neighbour_x; c++) {
								if (img_prev_row[c] > 0) {
									if (label == 0) {
										label = labels_prev_row[c];
									}
									else {
										ET.Merge(labels_prev_row[c], label);
									}
								}
							}
						}

						// Previous col
						if (x > 0) {
							if (img_row[x - 1] > 0) {
								if (label == 0) {
									label = labels_row[x - 1];
								}
								else {
									ET.Merge(labels_row[x - 1], label);
								}
							}
						}

						if (label == 0) {
							label = ET.NewLabel();
						}
					}
					labels_row[x] = label;
				}
			} // Rows cycle end
		} // Planes cycle end

		// Second scan
		this->n_labels_ = ET.Flatten();

		
		int * img_row = reinterpret_cast<int*>(this->img_labels_.data);
		for (int z = 0; z < this->img_labels_.size[0]; z++) {
			for (int y = 0; y < this->img_labels_.size[1]; y++) {
				for (int x = 0; x < this->img_labels_.size[2]; x++) {
					img_row[x] = ET.GetLabel(img_row[x]);
				}
				img_row += this->img_labels_.step[1] / sizeof(int);
			}
		}

		calc_features3d_post<ConfFeatures>(this->img_labels_, this->n_labels_, this->features);
		
		ET.Dealloc(); // Memory deallocation of the labels solver

	}
	
	void PerformLabelingWithSteps()	{
	    	double alloc_timing = Alloc();

		int32_t depth = this->img_.size.p[0];
		int32_t height = this->img_.size.p[1];
		int32_t width = this->img_.size.p[2];
		int32_t size = depth * height * width;
	
		this->perf_.start();
		this->samplers.Start();
		
		FirstScan();
	
		this->samplers.Stop();
	
		this->perf_.stop();
		this->perf_.store(Step(StepType::FIRST_SCAN), this->perf_.last());
		this->samplers.Store(StepType::FIRST_SCAN, size);

	
		this->perf_.start();
		this->samplers.Start();
		
		SecondScan();
		
		this->samplers.Stop();
		this->perf_.stop();
		
		this->perf_.store(Step(StepType::RELABELING), this->perf_.last());
		this->samplers.Store(StepType::RELABELING, size);

		MEASURE_STEP(calc_features3d_post<ConfFeatures>(this->img_labels_, this->n_labels_,
								this->features),
			     StepType::FEATURES, this->perf_, this->samplers, size);		
		
		this->perf_.start();
		Dealloc();
		this->perf_.stop();
		this->perf_.store(Step(StepType::ALLOC_DEALLOC), this->perf_.last() + alloc_timing);

	}

	private:
	double Alloc()
	{
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

		// Rosenfeld Mask
		// +-+-+-+
		// |p|q|r|
		// +-+-+-+
		// |s|x|
		// +-+-+

		// First scan
		for (int z = 0; z < this->img_.size[0]; z++) {

			unsigned char const * const img_plane = this->img_.data + this->img_.step[0] * z;   //   this->img_.template ptr<unsigned char>(z, 0, 0);
			unsigned char const * const img_prev_plane = (z > 0) ? (img_plane - this->img_.step[0]) : nullptr;
			int * const labels_plane = reinterpret_cast<int*>(this->img_labels_.data) + (this->img_labels_.step[0] / sizeof(int)) * z;
			int * const labels_prev_plane = labels_plane - (this->img_labels_.step[0] / sizeof(int));

			for (int y = 0; y < this->img_.size[1]; y++) {

				// Prev plane row pointers
				unsigned char const * img_prev_plane_rows[3];
				int prev_plane_first_row, prev_plane_last_row;
				if (img_prev_plane != nullptr) {
					img_prev_plane_rows[1] = img_prev_plane + this->img_.step[1] * y;
					img_prev_plane_rows[0] = (y > 0) ? (prev_plane_first_row = 0, img_prev_plane_rows[1] - this->img_.step[1]) : (prev_plane_first_row = 1, nullptr);
					img_prev_plane_rows[2] = (y + 1 < this->img_.size[1]) ? (prev_plane_last_row = 2, img_prev_plane_rows[1] + this->img_.step[1]) : (prev_plane_last_row = 1, nullptr);
				}

				int * labels_prev_plane_rows[3];
				labels_prev_plane_rows[1] = labels_prev_plane + (this->img_labels_.step[1] / sizeof(int)) * y;
				labels_prev_plane_rows[0] = labels_prev_plane_rows[1] - (this->img_labels_.step[1] / sizeof(int));
				labels_prev_plane_rows[2] = labels_prev_plane_rows[1] + (this->img_labels_.step[1] / sizeof(int));

				// Cur plane row pointers
				unsigned char const * const img_row = img_plane + this->img_.step[1] * y;
				unsigned char const * const img_prev_row = (y > 0) ? (img_row - this->img_.step[1]) : nullptr;
				int * const labels_row = labels_plane + (this->img_labels_.step[1] / sizeof(int)) * y;
				int * const labels_prev_row = labels_row - (this->img_labels_.step[1] / sizeof(int));

				for (int x = 0; x < this->img_.size[2]; x++) {
					int label = 0;
					if (img_row[x] > 0) {

						int const first_neighbour_x = (x > 0) ? (x - 1) : x;
						int const last_neighbour_x = (x + 1 < this->img_.size[2]) ? (x + 1) : x;

						// Previous plane
						if (img_prev_plane != nullptr) {
							for (int r = prev_plane_first_row; r <= prev_plane_last_row; r++) {
								for (int c = first_neighbour_x; c <= last_neighbour_x; c++) {
									if (img_prev_plane_rows[r][c] > 0) {
										if (label == 0) {
											label = labels_prev_plane_rows[r][c];
										}
										else {
											ET.Merge(labels_prev_plane_rows[r][c], label);
										}
									}
								}
							}
						}

						// Previous row
						if (img_prev_row != nullptr) {
							for (int c = first_neighbour_x; c <= last_neighbour_x; c++) {
								if (img_prev_row[c] > 0) {
									if (label == 0) {
										label = labels_prev_row[c];
									}
									else {
										ET.Merge(labels_prev_row[c], label);
									}
								}
							}
						}

						// Previous col
						if (x > 0) {
							if (img_row[x - 1] > 0) {
								if (label == 0) {
									label = labels_row[x - 1];
								}
								else {
									ET.Merge(labels_row[x - 1], label);
								}
							}
						}

						if (label == 0) {
							label = ET.NewLabel();
						}
					}
					labels_row[x] = label;
				}
			} // Rows cycle end
		} // Planes cycle end
	}

	void SecondScan() {
		// Second scan
		this->n_labels_ = ET.Flatten();

		int * img_row = reinterpret_cast<int*>(this->img_labels_.data);
		for (int z = 0; z < this->img_labels_.size[0]; z++) {
			for (int y = 0; y < this->img_labels_.size[1]; y++) {
				for (int x = 0; x < this->img_labels_.size[2]; x++) {
					img_row[x] = ET.GetLabel(img_row[x]);
				}
				img_row += this->img_labels_.step[1] / sizeof(int);
			}
		}
	}
};

#endif // !YACCLAB_LABELING_NAIVE_3D_H_
