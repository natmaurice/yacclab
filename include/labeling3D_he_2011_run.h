// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_LABELING3D_HE_2011_RUN_H_
#define YACCLAB_LABELING3D_HE_2011_RUN_H_

#include <opencv2/core.hpp>

#include "labeling_algorithms.h"
#include "labels_solver.h"
#include "memory_tester.h"
#include "calc_features.hpp"



struct Run
{
    uint16_t start = 0;
    uint16_t end = 0;
    unsigned label = 0;
};

class Table3D
{
public:

    size_t d_;
    size_t h_;
    size_t max_runs_;
    Run *data_ = nullptr;       // This vector stores run data for each row of each slice
    uint16_t *sizes_ = nullptr; // This vector stores the number of runs actually contained in each row of each slice

    void Setup(size_t d, size_t h, size_t w)
    {
        d_ = d;
        h_ = h;
        max_runs_ = w / 2 + 1;
    }

    double Alloc(PerformanceEvaluator& perf)
    {
        perf.start();
        data_ = new Run[d_ * h_ * max_runs_];
        sizes_ = new uint16_t[d_ * h_];
        memset(data_, 0, d_ * h_ * max_runs_ * sizeof(Run));
        memset(sizes_, 0, d_ * h_ * sizeof(uint16_t));
        perf.stop();
        double t = perf.last();
        perf.start();
        memset(data_, 0, d_ * h_ * max_runs_ * sizeof(Run));
        memset(sizes_, 0, d_ * h_ * sizeof(uint16_t));
        perf.stop();
        return t - perf.last();
    }

    void Alloc()
    {
        Dealloc();
        data_ = new Run[d_ * h_ * max_runs_];
        sizes_ = new uint16_t[d_ * h_];
    }

    void Dealloc()
    {
        delete[] data_;
        delete[] sizes_;
	data_ = nullptr;
	sizes_ = nullptr;
    }
};

template <typename LabelsSolver, bool Relabeling = true, typename ConfFeatures = ConfFeatures3DNone>
class RBTS_3D : public Labeling3D<Connectivity3D::CONN_26, ConfFeatures>
{
protected:
    LabelsSolver ET;
public:
    RBTS_3D() {}

    Table3D runs;

    inline int ProcessRun(uint16_t row_index, uint16_t row_nruns, Run* row_runs, Run* cur_run, bool *new_label)
    {
        // Discard previous non connected runs (step "2" of the 2D algorithm)
        for (;
            row_index < row_nruns &&
            row_runs[row_index].end < cur_run->start - 1;
            ++row_index) {
        }

        // Get label (step "3A" of the 2D algorithm)
        if (row_index < row_nruns &&
            row_runs[row_index].start <= cur_run->end + 1) {
            if (*new_label) {
                cur_run->label = row_runs[row_index].label;
                *new_label = false;
            }
            else {
                ET.Merge(cur_run->label, row_runs[row_index].label);
            }
        }

        // Merge label (step "3B" of the 2D algorithm)
        for (;
            row_index < row_nruns &&
            row_runs[row_index].end <= cur_run->end;
            ++row_index) {
            ET.Merge(cur_run->label, row_runs[row_index].label);
        }

        // Get label without "removing the run" (step "4" of the 2D algorithm)
        // the skip step is not required in this case because this algorithm does not employ
        // a circular buffer.
        if (row_index < row_nruns &&
            row_runs[row_index].start <= cur_run->end + 1) {
            ET.Merge(cur_run->label, row_runs[row_index].label);
        }
        return row_index;
    }

    void PerformLabeling()
    {
        int d = this->img_.size.p[0];
        int h = this->img_.size.p[1];
        int w = this->img_.size.p[2];

	Labeling::StepsDuration elapsed;
	elapsed.Init();
	
	FirstScan();
	SecondScan(elapsed);

    }

    void PerformLabelingWithSteps()
    {
	double alloc_timing = Alloc();

	int32_t depth = this->img_.size.p[0];
	int32_t height = this->img_.size.p[1];
	int32_t width = this->img_.size.p[2];
	int32_t size = depth * height * width;
	
	Labeling::StepsDuration elapsed;
	elapsed.Init();
	
	MEASURE_STEP_TIME(FirstScan(), StepType::FIRST_SCAN, this->perf_, elapsed, this->samplers, size);
	
	SecondScan(elapsed); // Transitive Closure + Relabeling + Features
	
	this->perf_.start();
	Dealloc();
	this->perf_.stop();
	this->perf_.store(Step(StepType::ALLOC_DEALLOC), this->perf_.last() + alloc_timing);
	
	elapsed.CalcDerivedTime();
	elapsed.StoreAll(this->perf_);
	this->samplers.CalcDerived();
    }

    bool UseRelabeling() const override {
	return Relabeling;
    }
    
private:
    double Alloc()
    {
	Dealloc();
	
	this->samplers.Reset();
        // Memory allocation of the labels solver
        double ls_t = ET.Alloc(UPPER_BOUND_26_CONNECTIVITY, this->perf_);
        // Memory allocation of Table3D 
        runs.Setup(this->img_.size.p[0], this->img_.size.p[1], this->img_.size.p[2]);
        ls_t += runs.Alloc(this->perf_);
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
	// TODO: touch the runs
        this->perf_.stop();
        double ma_t = t - this->perf_.last();
        // Return total time
        return ls_t + ma_t;
    }
    
    void Dealloc()
    {
        ET.Dealloc();
        runs.Dealloc();
        // No free for img_labels_ because it is required at the end of the algorithm 
    }
    
    void FirstScan()
    {
        int d = this->img_.size.p[0];
        int h = this->img_.size.p[1];
        int w = this->img_.size.p[2];

        //memset(this->img_labels_.data, 0, this->img_labels_.dataend - this->img_labels_.datastart);

        ET.Setup(); // Labels solver initialization
        
        // First scan
        Run* run_slice00_row00 = runs.data_;
        uint16_t* nruns_slice00_row00 = runs.sizes_;
        for (int s = 0; s < d; s++) {

            for (int r = 0; r < h; r++) {
                // Row pointers for the input image
                const unsigned char* const img_slice00_row00 = this->img_.template ptr<unsigned char>(s, r);

                int slice11_row00_index = 0;
                int slice11_row11_index = 0;
                int slice11_row01_index = 0;
                int slice00_row11_index = 0;

                int nruns = 0;

                Run* run_slice11_row00 = run_slice00_row00 - (runs.max_runs_) * runs.h_;
                Run* run_slice11_row11 = run_slice11_row00 - (runs.max_runs_);
                Run* run_slice11_row01 = run_slice11_row00 + (runs.max_runs_);

                Run* run_slice00_row11 = run_slice00_row00 - (runs.max_runs_);

                uint16_t* nruns_slice11_row00 = nruns_slice00_row00 - h;
                uint16_t* nruns_slice11_row11 = nruns_slice11_row00 - 1;
                uint16_t* nruns_slice11_row01 = nruns_slice11_row00 + 1;

                uint16_t* nruns_slice00_row11 = nruns_slice00_row00 - 1;

                for (int c = 0; c < w; c++) {
                    // Is there a new run ?
                    if (img_slice00_row00[c] == 0) {
                        continue;
                    }

                    // Yes (new run)
                    bool new_label = true;
                    run_slice00_row00[nruns].start = c; // We start from 1 because 0 is a "special" run
                                                        // to store additional info
                    for (; c < w && img_slice00_row00[c] > 0; ++c) {}
                    run_slice00_row00[nruns].end = c - 1;

                    if (s > 0) {
                        if (r > 0) {
                            slice11_row11_index = ProcessRun(slice11_row11_index,        // uint16_t row_index
                                                             *nruns_slice11_row11,       // uint16_t row_nruns
                                                             run_slice11_row11,          // Run* row_runs
                                                             &run_slice00_row00[nruns],  // Run* cur_run
                                                             &new_label                  // bool *new_label
                                                            );
                        }
                        slice11_row00_index = ProcessRun(slice11_row00_index,        // uint16_t row_index
                                                         *nruns_slice11_row00,       // uint16_t row_nruns
                                                         run_slice11_row00,          // Run* row_runs
                                                         &run_slice00_row00[nruns],  // Run* cur_run
                                                         &new_label                  // bool *new_label
                                                        );
                        if (r < h - 1) {
                            slice11_row01_index = ProcessRun(slice11_row01_index,        // uint16_t row_index
                                                             *nruns_slice11_row01,       // uint16_t row_nruns
                                                             run_slice11_row01,          // Run* row_runs
                                                             &run_slice00_row00[nruns],  // Run* cur_run
                                                             &new_label                  // bool *new_label
                                                            );
                        }
                    }

                    if (r > 0) {
                        slice00_row11_index = ProcessRun(slice00_row11_index,        // uint16_t row_index
                                                         *nruns_slice00_row11,       // uint16_t row_nruns
                                                         run_slice00_row11,          // Run* row_runs
                                                         &run_slice00_row00[nruns],  // Run* cur_run
                                                         &new_label                  // bool *new_label
                                                        );
                    }

                    if (new_label) {
                        run_slice00_row00[nruns].label = ET.NewLabel();
                    }
                    nruns++;
                } // Columns cycle end

                run_slice00_row00 += (runs.max_runs_);
                (*nruns_slice00_row00++) = nruns;
            } // Rows cycle end
        } // Planes cycle end
    }
    void SecondScan(Labeling::StepsDuration& elapsed)
    {
        int d = this->img_.size.p[0];
        int h = this->img_.size.p[1];
        int w = this->img_.size.p[2];

	unsigned size = d * h * w;
	
	MEASURE_STEP_TIME(
	    this->n_labels_ = ET.Flatten(),
	    StepType::TRANSITIVE_CLOSURE, this->perf_, elapsed, this->samplers, size);

	
	if (!std::is_same<ConfFeatures, ConfFeatures3DNone>::value) {	
	    MEASURE_STEP_TIME(
		Run* run_row = runs.data_;
		uint16_t* nruns = runs.sizes_;	

		this->features.template Init<ConfFeatures>(this->n_labels_);
	    
		for (int s = 0; s < d; s++) {
		    for (int r = 0; r < h; r++) {
			for (int id = 0; id < *nruns; id++) {

			    int seg_start = run_row[id].start;
			    int seg_end = run_row[id].end;
			    int label = ET.GetLabel(run_row[id].label);
						
			    this->features.template AddSegment3D<ConfFeatures>(
				label, r, s, seg_start, seg_end + 1);
			}
			run_row += (runs.max_runs_);
			nruns++;
		    }
		}, StepType::FEATURES, this->perf_, elapsed, this->samplers, size);
	}


	if (UseRelabeling()) {
	    MEASURE_STEP_TIME(
		int* img_row = reinterpret_cast<int*>(this->img_labels_.data);
		Run* run_row = runs.data_;
		uint16_t* nruns = runs.sizes_;	
		for (int s = 0; s < d; s++) {
		    for (int r = 0; r < h; r++) {
			memset(img_row, 0, w * sizeof(int));
			for (int id = 0; id < *nruns; id++) {
			    for (int c = run_row[id].start; c <= run_row[id].end; ++c) {
				img_row[c] = ET.GetLabel(run_row[id].label);
			    }
			}
			run_row += (runs.max_runs_);
			img_row += this->img_labels_.step[1] / sizeof(int);
			nruns++;
		    }
		}, StepType::RELABELING, this->perf_, elapsed, this->samplers, size);
	}
    }
};

#endif // YACCLAB_LABELING3D_HE_2011_RUN_H_
