#ifndef YACCLAB_LABELING3D_LSL_GENERIC_DOUBLE_HPP
#define YACCLAB_LABELING3D_LSL_GENERIC_DOUBLE_HPP

#include "labeling3D_lsl_generic.hpp"

#include "algos/conf.hpp"



/*
 * Generic class for LSL algorithms that use a state machine
 * The purpose of this class is to reduce the amount of code duplication of the multiple 
 * versions of LSL.
 * To create a new LSL algorithms, register it in "src/<name>.cc" using the appropriate YACCLAB
 * registration methods and using the desired template parameters.
 * Here, we use the Curious Recurring Template Pattern. This allows for compile-time inheritence 
 * (ie: no vtable resolution overhead during runtime). This means that a new algorithms must have 
 * its name as the "Derived" template parameter. The second template parameter is the label solver.
 * This parameter is from YACCLAB and currently has no effect here (a custom UnionFind structure is 
 * used). The last parameter contains the properties of the algorithm (see conf.hpp). 
 * As of now, the properties are either "Conf::Double" (the algorithm uses the double line 
 * optimisation) and "Conf::Pipeline" (the double line unifications are pipeline). Note that 
 * Conf::Pipeline true requires Conf::Double to be true as well.
 */
template <class Derived, typename LabelsSolver, typename Conf>
class LSL3DZ_Generic_DoubleLine : public LSL3D_Generic<Derived, LabelsSolver> {
    static_assert(!Conf::Pipeline || Conf::Double);
    static_assert(Conf::ERA || (!Conf::ERA && !Conf::Pipeline && !Conf::Double),
		  "NoERA may not be used with Double or Pipeline");
public:
    LSL3DZ_Generic_DoubleLine() {
    }
protected:
    ERATable ERA;
    RLCTable rlc;
    algo::StateCounters counters; 
    
    int16_t* __restrict__ rlc_row0;
    int16_t* __restrict__ rlc_row1;
    int16_t* __restrict__ rlc_row2;
    int16_t* __restrict__ rlc_row3;

    int32_t* __restrict__ era_row0;
    int32_t* __restrict__ era_row1;
    int32_t* __restrict__ era_row2;
    int32_t* __restrict__ era_row3;

    int16_t len0, len1, len2, len3;

    int32_t uf_offset;
    int32_t uf_offset0, uf_offset1, uf_offset2, uf_offset3;

    
private:
    // These are set as private as they shouldn't be modified by a child class as they are used to
    // allocate/free memory
    // Use the "local" variants below instead
    int32_t* __restrict__ era_col0;
    int32_t* __restrict__ era_col1;
    int16_t* __restrict__ rlc_col0;
    int16_t* __restrict__ rlc_col1;

protected:
    // DoubleLines: these can safely be modified
    int16_t* __restrict__ l_rlc_col0;
    int16_t* __restrict__ l_rlc_col1;
    int32_t* __restrict__ l_era_col0;
    int32_t* __restrict__ l_era_col1;
    int16_t len_col0, len_col1;
    
    int16_t SizeOfRLC() const {
	constexpr int32_t DOUBLELINE_COUNT = 2; // Only 2 double-lines are necessary
	constexpr int32_t EXTRASPACE_PER_DOUBLELINE = 12;
	return (this->width + EXTRASPACE_PER_DOUBLELINE) * 2 * DOUBLELINE_COUNT;
    }

    // Ideally, we'd want to define class as friend of parent class to avoid making everything public
public:       
    void Touch() {
	rlc.Touch();

	if constexpr (Conf::ERA) {
	    ERA.Touch();
	}

	if constexpr (Conf::Double) {
	    touch(rlc_col0, SizeOfRLC());
	    touch(era_col0, this->width + 2);
	}
    }
    
    // Parent methods
    template<bool MeasureTime>
    double Alloc() {	
		
	//double ls_t = LabelsSolver::Alloc(UPPER_BOUND_26_CONNECTIVITY, perf_);
	double ls_t = 0, ma_t = 0;
	// Memory allocation for the output image

	if constexpr (MeasureTime) {
	    this->perf_.start();
	}

	rlc.Reserve(this->width + 2, this->height, this->depth);

	if constexpr (Conf::ERA) {	
	    ERA.Reserve(this->width + 2, this->height, this->depth);	
	}
	
	int32_t er_slice_pitch = this->width * (this->height + 1);
	int16_t er_row_pitch = this->width;
	
	if constexpr (Conf::Double) {
	    rlc_col0 = new int16_t[(this->width + 12) * 2 * 2];
	    era_col0 = new int32_t[this->width * 2];

	    rlc_col1 = rlc_col0 + 2 * this->width + 2;
	    era_col1 = era_col0 + this->width;
	}
	
	Touch();

	double t = 0;
	if constexpr (MeasureTime) {
	    this->perf_.stop();
	    t = this->perf_.last();
	    
	    this->perf_.start();
	    
	    Touch();

	    this->perf_.stop();
	    ma_t = t - this->perf_.last();
	}
	// Return total time
	return ma_t;
    }
    
    void Dealloc() {
	rlc.Free();
	
	if constexpr (Conf::ERA){	    
	    ERA.Free();
	}
	
	this->labels.Free();

	if constexpr (Conf::Double) {	    
	    delete[] rlc_col0;
	    delete[] era_col0;
	    rlc_col0 = nullptr;
	    era_col0 = nullptr;
	}
    }
    
    void Setup() {
	int16_t* data_ptr = rlc.Ptr(0);
	data_ptr[0] = INT16_MAX - 1; // Bigger than width: virtual line won't be taken into consideration
	data_ptr[1] = INT16_MAX - 1; // We substract 1 to avoid overflows if adding 1
		    
	rlc.SetLineSize( -1, 0, 0);
	rlc.SetPtr(data_ptr, -1,  0);
    
	for (int16_t i = -1; i <= this->height; i++) {
	    // First slice
	    rlc.SetPtr(data_ptr, i, -1);
	    rlc.SetLineSize(i, -1, 0);
	}
	for (int16_t i = -1; i < this->depth; i++) {
	    // First rows
	    rlc.SetPtr(data_ptr, -1, i);
	    rlc.SetLineSize(-1, i, 0);
	}    
	rlc.SetPtr(data_ptr + 2, 0,  0);

	
	rlc_row0 = rlc.Ptr(0, 0);
	rlc_row1 = rlc.Ptr(0, 0);
	rlc_row2 = rlc.Ptr(0, 0);
	rlc_row3 = rlc.Ptr(0, 0);

	if constexpr (Conf::ERA) {
	    era_row0 = ERA.Ptr(0, 0);
	    era_row1 = ERA.Ptr(0, 0);
	    era_row2 = ERA.Ptr(0, 0);
	    era_row3 = ERA.Ptr(0, 0);
	}
	
	len0 = 0;
	len1 = 0;
	len2 = 0;
	len3 = 0;

	if constexpr (Conf::Double){
	    l_rlc_col0 = rlc_col0;
	    l_rlc_col1 = rlc_col1;
	    l_era_col0 = era_col0;
	    l_era_col1 = era_col1;

	    len_col0 = 0;
	    len_col1 = 0;
	
	    l_rlc_col0[0] = INT16_MAX - 1;
	    l_rlc_col0[1] = INT16_MAX - 1;
	}

	uf_offset = 1;
	uf_offset0 = 1;
	uf_offset1 = 1;
	uf_offset2 = 1;
	uf_offset3 = 1;
    }
    
    template<bool MeasureSteps>
    void FirstScan(double& rle_t, double& unification_t, double& remaining_t) {

	assert(this->width > 0 && this->height > 0 && this->depth > 0);
	assert(this->size > 0);
	assert(this->uf.Capacity() > 0);
	assert(this->image.Width() == this->width);
	assert(this->image.Height() == this->height);
	assert(this->image.Depth() == this->depth);
	

	if constexpr (MeasureSteps) {	    
	    this->perf_.start();
	}

	Setup();
	

	int16_t segment_count = 0;
	uint8_t* __restrict__ image_row = this->image.GetLine(0, 0);
	int16_t* __restrict__ rlc_row = rlc.Ptr(0, 0);
	int32_t* __restrict__ era_row = nullptr;
	if constexpr (Conf::ERA) {
	    era_row = ERA.Ptr(0, 0);
	}
	segment_count = 0;

	int32_t slice_offset = 1;
	for (int16_t slice = 0; slice < this->depth; slice++) {
	    // Note: ERA and rlc are expected to provide border management	

	    if constexpr (Conf::Double) {
		len_col0 = 0;
		len_col1 = 0;

		l_rlc_col0[0] = INT16_MAX - 1;
		l_rlc_col0[1] = INT16_MAX - 1;
	    }

	    rlc_row0 = rlc.Ptr(-1, slice);
	    rlc_row1 = rlc.Ptr(0, slice - 1);
	    rlc_row2 = rlc.Ptr(-1, slice - 1);
	    rlc_row3 = rlc.Ptr(1, slice - 1);

	    if constexpr (Conf::ERA) {
		era_row0 = ERA.Ptr(-1, slice);
		era_row1 = ERA.Ptr(0, slice - 1);
		era_row2 = ERA.Ptr(-1, slice - 1);
		era_row3 = ERA.Ptr(1, slice - 1);
	    }
	    
	    len0 = rlc.Size(-1, slice);
	    len1 = rlc.Size(0, slice - 1);
	    len2 = rlc.Size(-1, slice - 1);
	    len3 = rlc.Size(1, slice - 1);

	    for (int16_t row = 0; row < this->height; row++) {

		// Register rotation: doing it at the end is not possible here
		// as it could be outside allocated memory
		len3 = rlc.Size(row + 1, slice - 1);
		if constexpr (Conf::ERA){
		    era_row3 = ERA.Ptr(row + 1, slice - 1);
		}
		rlc_row3 = rlc.Ptr(row + 1, slice - 1);

		uf_offset3 = uf_offset1 + len1 / 2;
		
		rlc_row = rlc_row + segment_count;
		era_row = era_row + segment_count / 2;	
		rlc.SetPtr(rlc_row, row, slice);
		
		if constexpr (Conf::ERA) {
		    ERA.SetPtr(era_row, row, slice);
		}
	    
		image_row = this->image.GetLine(row, slice);
	    	  	    
		assert(rlc_row != nullptr);

		if constexpr (MeasureSteps) {
		    this->perf_.stop();
		    remaining_t += this->perf_.last();
		    
		    this->perf_.start();
		}
		
		segment_count  = RLEScan(image_row, rlc_row, this->width);
		
		rlc_row[segment_count] = INT16_MAX - 1;
		rlc_row[segment_count + 1] = INT16_MAX - 1;
		
		rlc.SetLineSize(row, slice, segment_count);

		if constexpr (Conf::ERA){
		    ERA.SetLineSize(row, slice, segment_count / 2);
		}
		
		if constexpr (MeasureSteps) {
		    this->perf_.stop();
		    rle_t += this->perf_.last();
		
		    this->perf_.start();
		}
		
		assert(rlc_row1[len1] == INT16_MAX - 1);
		assert(rlc_row1[len1 + 1] == INT16_MAX - 1);
		
		if constexpr (!Conf::ERA) {		    
		    for (int32_t i = uf_offset; i < uf_offset + segment_count / 2; i++) {
			this->uf.m_parent[i]  = i;
		    }
		    this->uf.m_size += segment_count / 2; // unification_merge() will assign content
		}

		
		// Note: If ERA = false then function is expected to not use ERA.
		// In that case, the value that wil be passed is nullptr
		Unification(rlc_row, era_row, segment_count);

		if constexpr (MeasureSteps) {
		    this->perf_.stop();
		    unification_t += this->perf_.last();
		    this->perf_.start();
		}

		if constexpr (Conf::Double) {
		    std::swap(l_rlc_col0, l_rlc_col1);
		    std::swap(l_era_col0, l_era_col1);
		    std::swap(len_col0, len_col1);
		}
		
		rlc_row2 = rlc_row1;
		rlc_row1 = rlc_row3;
		rlc_row0 = rlc_row;

		if constexpr (Conf::ERA) {
		    era_row2 = era_row1;
		    era_row1 = era_row3;
		    era_row0 = era_row;
		}
		
		len2 = len1;
		len1 = len3;
		len0 = segment_count;
		// len3 done at the beginning of the loop
		
		uf_offset2 = uf_offset1;
		uf_offset1 = uf_offset3;
		//uf_offset3 += len3; // Done a the beginning of the loop
		uf_offset0 = uf_offset;
		uf_offset += segment_count / 2;
		
		rlc_row += 2;
	    }
	    // TODO: Add border management at beginning of slice
	}
	if constexpr (MeasureSteps) {
	    this->perf_.stop();
	    remaining_t += this->perf_.last();
	}

    }    
        
    int16_t RLEScan(uint8_t* __restrict__ image_row, int16_t* __restrict__ rlc_row,
		    const int16_t width) {
	return static_cast<Derived*>(this)->RLEScan(image_row, rlc_row, width);
    }
    
    void Unification(int16_t* __restrict__ rlc_row, int32_t* __restrict__ era_row,
		     int16_t segment_count) {
	static_cast<Derived*>(this)->Unification(rlc_row, era_row, segment_count);
    }
    
    void Relabeling() {
	static_cast<Derived*>(this)->Relabeling();
    }
};


#endif // YACCLAB_LABELING3D_LSL_GENERIC_DOUBLE_HPP
