#ifndef YACCLAB_LABELING3D_LSL_GENERIC_HPP
#define YACCLAB_LABELING3D_LSL_GENERIC_HPP


#include "labeling_algorithms.h"
#include "labels_solver.h"

#include <lsl3dlib/lsl3d/unification_common.hpp>
#include <lsl3dlib/conf.hpp>
#include <lsl3dlib/rle/rle.hpp>
#include <lsl3dlib/lsl3d/relabeling.hpp>
#include <lsl3dlib/features.hpp>
#include <lsl3dlib/lsl3d/lsl_features.hpp>
#include <lsl3dlib/utility.hpp>


#include <nrtype.h>
#include <nralloc2.h>
#include <nralloc3.h>
#include <nrset2.h>
#include <nrset3.h>


template <typename T>
inline void write_junk(T* restrict data, size_t len) {
    constexpr T junk_value = static_cast<T>(133);
    std::fill(data, data + len, junk_value);
}

#ifndef MEASURE_EACH_STEP
#define MEASURE_EACH_STEP 1
#endif // MEASURE_EACH_STEP

/*
 * This file defines several generic classes that can be used to prevent code duplication when 
 * implementing new LSL-based algorithms.
 */
template<typename LabelsSolver, class RLEAlgo, class UnifyAlgo, class RelabelingAlgo, 
	 class FeatureAlgo, typename ConfFeatures = ConfFeatures3DNone>
class LSL3D : public Labeling3D<Connectivity3D::CONN_26, ConfFeatures, false> {
protected:

    //static constexpr bool IsBitonal = RLEAlgo::IsBitonal; 

    struct Conf {
	static constexpr bool ER = RLEAlgo::Conf::ER;
	static constexpr bool ERA = UnifyAlgo::Conf::ERA;
	static constexpr bool Double = UnifyAlgo::Conf::Double;

	using Seg_t = typename RLEAlgo::Conf::Seg_t;
	using Label_t = typename UnifyAlgo::Conf::Label_t;	
    };
    
    static_assert(std::is_same<typename Conf::Seg_t, typename UnifyAlgo::Conf::Seg_t>::value,
		  "Incompatible segments types between RLE and unification");
    static_assert(std::is_same<typename Conf::Label_t, typename UnifyAlgo::Conf::Label_t>::value,
		  "Incompatible segments types between RLE and unification");

    static_assert(std::is_same<typename Conf::Seg_t, typename RelabelingAlgo::Conf::Seg_t>::value,
		  "Incompatible segments types between RLE and Relabeling");
    static_assert(std::is_same<typename Conf::Label_t, typename RelabelingAlgo::Conf::Label_t>::value,
		  "Incompatible segments types between RLE and Relabeling");


    LSL3D_CCL_t<Conf, LabelsSolver> LSL_context;
    
    //int32_t*** ERA = nullptr;
    //Seg_t*** rlc = nullptr;
    //int16_t** Lengths = nullptr;
    //LabelsSolver ET;
    
    // Only need to store 2 slices because ER is only used
    // in the RLE+Unification and we only access the current slice along
    // with the above slice.
    //int16_t* ER = nullptr; // Where allocation is stored
    int16_t *ERp0 = nullptr, *ERp1 = nullptr; // Pointers to each slice. ERp0 is the top slice and ERp1 is the bottom

        
    int16_t* restrict ERi = nullptr;

    AdjState<typename Conf::Seg_t, typename Conf::Label_t> state;
    
    size_t er_row_pitch = 0, er_slice_pitch = 0;
    
    typename Conf::Seg_t* restrict rlc_col0 = nullptr;
    typename Conf::Seg_t* restrict rlc_col1 = nullptr;
    typename Conf::Label_t* restrict era_col0 = nullptr;
    typename Conf::Label_t* restrict era_col1 = nullptr;

    static constexpr size_t ER_PADDING = 1;

	    
    static constexpr size_t ER_TOP_BORDER = 1;
    static constexpr size_t ER_BOTTOM_BORDER = 1;
    static constexpr size_t ER_VERT_BORDER = ER_TOP_BORDER + ER_BOTTOM_BORDER;
    static constexpr size_t ER_LEFT_BORDER  = 1;
    static constexpr size_t ER_RIGHT_BORDER = 1;
    static constexpr size_t ER_HORIZ_BORDER = ER_LEFT_BORDER + ER_RIGHT_BORDER;


    struct CubeDims {
	int x0, y0, z0;
	int x1, y1, z1;
    };

    CubeDims RLC_borders;
    CubeDims ERA_borders;
    
public:
    LSL3D() {
    }
    
    void PerformLabeling() override {
	
	int width, height, depth;
	GetMatSize(this->img_, width, height, depth);

	Labeling::StepsDuration elapsed;
	elapsed.Init();
	
	this->stats.tmp_labels = 0;
	this->stats.total_labels = 0;
	
        FirstScan<false>(elapsed);
        SecondScan<false>(elapsed);

	FeatureAlgo::template CalcFeatures<LabelsSolver, ConfFeatures>(
	    LSL_context.RLC, LSL_context.ERA, LSL_context.Lengths, LSL_context.ET,
	    this->features, this->n_labels_, depth, height, width);
	
	assert(this->stats.tmp_labels < width * height * depth);
	assert(this->stats.total_labels < width * height * depth);
    }
    
    void PerformLabelingWithSteps() {
	double alloc_timing = Alloc<true>();
	
	uint32_t depth = this->img_.size.p[0];
	uint32_t height = this->img_.size.p[1];
	uint32_t width = this->img_.size.p[2];
	uint32_t size = depth * height * width;

	Labeling::StepsDuration elapsed;
	elapsed.Init();
	
	this->samplers.Start();
	
	FirstScan<true>(elapsed);

	this->samplers.Stop();	
	this->samplers.Store(StepType::FIRST_SCAN, size);

	// Sampling is done inside SecondScan<true>()
	SecondScan<true>(elapsed);


	this->samplers.Start();
	this->perf_.start();
        FeatureAlgo::template CalcFeatures<LabelsSolver, ConfFeatures>(
	    LSL_context.RLC, LSL_context.ERA, LSL_context.Lengths, LSL_context.ET, this->features,
	    this->n_labels_, depth, height, width);
	
	this->perf_.stop();
	this->samplers.Stop();

	this->samplers.Store(StepType::FEATURES, size);
	elapsed.duration[StepType::FEATURES] = this->perf_.last();

	
	Dealloc();

	elapsed.CalcDerivedTime();
	elapsed.StoreAll(this->perf_);
	this->samplers.CalcDerived();
    }
    
    int16_t SizeOfRLC() const {
	constexpr uint32_t DOUBLELINE_COUNT = 2; // Only 2 double-lines are necessary
	constexpr uint32_t EXTRASPACE_PER_DOUBLELINE = 12;
	return (this->width + EXTRASPACE_PER_DOUBLELINE) * 2 * DOUBLELINE_COUNT;
    }

    constexpr uint32_t CalcMaxComponent(uint16_t width, uint16_t height, uint16_t depth) {
	return (width / 2 + 1) * (height / 2 + 1) * (depth / 2 + 1);	
    }

    bool UseRelabeling() const override {
	return RelabelingAlgo::Conf::DO_NOTHING == false;
    }
    
protected:    
    uint16_t width = 0, height = 0, depth = 0;
    uint32_t size = 0;

    // Ensure that memory will be physically allocated
    void Touch() {

	const size_t max_components = CalcMaxComponent(width, height, depth);
	
	memset(this->img_labels_.data, 0, this->img_labels_.dataend - this->img_labels_.datastart);

	zero_si16cube(LSL_context.RLC, RLC_borders.z0, RLC_borders.z1, RLC_borders.y0, RLC_borders.y1,
		      RLC_borders.x0, RLC_borders.x1);

	zero_si16matrix(LSL_context.Lengths, RLC_borders.z0, RLC_borders.z1, RLC_borders.y0, RLC_borders.y1);
	
	if (Conf::ERA) {
	    zero_si32cube(LSL_context.ERA, ERA_borders.z0, ERA_borders.z1, ERA_borders.y0, ERA_borders.y1,
			  ERA_borders.x0, ERA_borders.x1);
	}

	if (Conf::Double) {
	    touch(rlc_col0, SizeOfRLC());
	    touch(era_col0, this->width + 2);
	}
		
	if (Conf::ER) {
	    std::fill(LSL_context.ER, LSL_context.ER + 2 * er_slice_pitch, 0);
	}

	this->features.template Touch<ConfFeatures>();
    }


    double Alloc() {
	return Alloc<false>();
    }
    
    template<bool MeasureTime>
    double Alloc()  {

	this->samplers.Reset();
	
	Dealloc();
	
        width = this->img_.size.p[2];
	height = this->img_.size.p[1];
	depth = this->img_.size.p[0];
	size = width * height * depth;

	// Technically, it should be (width + 1 / 2) * (height / 2 + 1) * (depth / 2 + 1)
	const size_t max_components = ((width + 1) / 2) * ((height + 1) / 2) * ((depth + 1) / 2) + 1;
	assert(this->width > 0 && this->height > 0 && this->depth > 0);
	assert(size > 0);
	
	if (MeasureTime) {
	    this->perf_.start();
	}

	uint32_t row_pitch = this->img_.step[1];
	assert(row_pitch >= width);
	
	// If the image does not have borders then an SIMD operation may write beyond the image boundaries
	// While it's not an issues for most lines (data would get overwritten on the next row), it
	// remains a problem on the last few lines
	// The choice here is to add padding on *each line*. Adding padding only at the end of the
	// image or performing scalar relabeling on the last few lines would also work. 	
	create_mat_with_border<int32_t>(this->img_labels_, width, height, depth, 0, 0, 0,
					0, 0, 1);

	
	memset(this->img_labels_.data, 0, this->img_labels_.dataend - this->img_labels_.datastart);


	int16_t max_segments_per_line = (this->width + 1) * 2;
	
	RLC_borders.z0 = -1;
	RLC_borders.z1 = this->depth;
	RLC_borders.y0 = -1;
	RLC_borders.y1 = this->height;
	RLC_borders.x0 = 0;
	RLC_borders.x1 = roundup_kpow2(max_segments_per_line + 2, RLEAlgo::Conf::SIMD_WORDS * 8);

	
	LSL_context.RLC = si16cube(RLC_borders.z0, RLC_borders.z1, RLC_borders.y0, RLC_borders.y1,
		       RLC_borders.x0, RLC_borders.x1);
	LSL_context.Lengths = si16matrix(RLC_borders.z0, RLC_borders.z1, RLC_borders.y0, RLC_borders.y1);
	
	if (Conf::ERA) {
	    ERA_borders.z0 = -1;
	    ERA_borders.z1 = this->depth;
	    ERA_borders.y0 = -1;
	    ERA_borders.y1 = this->height;
	    ERA_borders.x0 = 0;
	    ERA_borders.x1 = max_segments_per_line / 2;

	    LSL_context.ERA = si32cube(ERA_borders.z0, ERA_borders.z1, ERA_borders.y0, ERA_borders.y1,
			   ERA_borders.x0, ERA_borders.x1);
	}
	
	if (Conf::Double) {
	    rlc_col0 = new int16_t[(this->width + 12) * 2 * 2];
	    era_col0 = new int32_t[this->width * 2];

	    rlc_col1 = rlc_col0 + 2 * this->width + 2;
	    era_col1 = era_col0 + this->width;
	}

	if (Conf::ER) {

	    constexpr size_t AVX512_ALIGN = 32;

	    
	    er_row_pitch = calc_stride(this->width + ER_HORIZ_BORDER, AVX512_ALIGN);
	    er_slice_pitch = er_row_pitch * (this->height + ER_VERT_BORDER);
	
	    LSL_context.ER = new int16_t[2 * er_slice_pitch];
	    ERp0 = LSL_context.ER + er_row_pitch + ER_LEFT_BORDER;
	    ERp1 = ERp0 + er_slice_pitch;
	}

	LSL_context.ET.Alloc(max_components, this->perf_);
	this->features.template Alloc<ConfFeatures>(max_components);

	// Perform shallow copy (does not copy data)
	LSL_context.image = this->img_;
	LSL_context.labels = this->img_labels_;

	LSL_context.width = this->width;
	LSL_context.height = this->height;
	LSL_context.depth = this->depth;
	
	double t = 0, ma_t = 0;
	if (MeasureTime) {
	    this->perf_.stop();
	    t = this->perf_.last();
	    
	    this->perf_.start();
	}
	
	Touch();
	
	if (MeasureTime) {
	    this->perf_.stop();
	    ma_t = t - this->perf_.last();
	}
	
	return ma_t;
    }
    
    void Dealloc() {

	if (LSL_context.RLC) {
	    free_si16cube(LSL_context.RLC, RLC_borders.z0, RLC_borders.z1, RLC_borders.y0, RLC_borders.y1,
			  RLC_borders.x0, RLC_borders.x1);
	}
	if (LSL_context.Lengths) {
	    free_si16matrix(LSL_context.Lengths, RLC_borders.z0, RLC_borders.z1, RLC_borders.y0, RLC_borders.y1);
	}
	
	LSL_context.RLC = nullptr;
	LSL_context.Lengths = nullptr;
	
	
	if (Conf::ERA){

	    if (LSL_context.ERA) {
		free_si32cube(LSL_context.ERA, ERA_borders.z0, ERA_borders.z1, ERA_borders.y0, ERA_borders.y1,
			      ERA_borders.x0, ERA_borders.x1);
 	    }
	    LSL_context.ERA = nullptr;
	}
	

	if (Conf::Double) {
	    delete[] rlc_col0;
	    delete[] era_col0;
	    rlc_col0 = nullptr;
	    era_col0 = nullptr;
	}

	if (Conf::ER){
	    delete[] LSL_context.ER;
	    LSL_context.ER = state.ER0 = state.ER1 = state.ER2 = state.ER3 = ERi = nullptr;
	    ERp0 = ERp1 = nullptr;
	}

	LSL_context.ET.Dealloc();
    }

    void Setup() {

	LSL_context.RLC[-1][-1][0] = std::numeric_limits<typename Conf::Seg_t>::max() - 1;
	LSL_context.RLC[-1][-1][1] = std::numeric_limits<typename Conf::Seg_t>::max() - 1;
	
	state.RLC0 = LSL_context.RLC[-1][-1];
	state.RLC1 = LSL_context.RLC[-1][-1];
	state.RLC2 = LSL_context.RLC[-1][-1];
	state.RLC3 = LSL_context.RLC[-1][-1];

	for (int row = RLC_borders.y0; row <= RLC_borders.y1; row++) {
	    LSL_context.RLC[-1][row] = LSL_context.RLC[-1][-1];
	    LSL_context.RLC[this->depth][row] = LSL_context.RLC[-1][-1];
	}
	for (int slice = RLC_borders.z0; slice <= RLC_borders.z1; slice++) {
	    LSL_context.RLC[slice][-1] = LSL_context.RLC[-1][-1];
	    LSL_context.RLC[slice][this->height] = LSL_context.RLC[-1][-1];
	}
	
	if (Conf::ERA) {
	    state.ERA0 = LSL_context.ERA[-1][-1];
	    state.ERA1 = LSL_context.ERA[-1][-1];
	    state.ERA2 = LSL_context.ERA[-1][-1];
	    state.ERA3 = LSL_context.ERA[-1][-1];
	}

	
	state.len0 = 0;
	state.len1 = 0;
	state.len2 = 0;
	state.len3 = 0;

	if (Conf::ER) {
	    std::fill(LSL_context.ER, LSL_context.ER + 2 * er_slice_pitch, 0);
	}

	if (Conf::Double){
	    state.l_RLC0 = rlc_col0;
	    state.l_RLC1 = rlc_col1;
	    state.l_ERA0 = era_col0;
	    state.l_ERA1 = era_col1;

	    state.l_len0 = 0;
	    state.l_len1 = 0;
	
	    state.l_RLC0[0] = INT16_MAX - 1;
	    state.l_RLC1[1] = INT16_MAX - 1;
	}
	LSL_context.ET.Setup();
    }
    
    template<bool MeasureTime>
    void FirstScan(Labeling::StepsDuration& elapsed) {

	int rowstride, slicestride;
	GetMatStrides<uint8_t>(this->img_, rowstride, slicestride);
	
	assert(this->width > 0 && this->height > 0 && this->depth > 0);
	assert(this->size > 0);

	if (MeasureTime) {	    
	    this->perf_.start();
	}
       
	Setup();

	int16_t segment_count = 0;
	uint8_t* restrict line = this->img_.template ptr<uint8_t>(0, 0);
	typename Conf::Seg_t* restrict RLCi = LSL_context.RLC[0][0];
	typename Conf::Label_t* restrict ERAi = nullptr;

	int old_label_count0 = 0;
	int old_label_count1 = 0;
	
	if (Conf::ERA) {
	    ERAi = LSL_context.ERA[0][0];
	}
	segment_count = 0;
	
	if (MeasureTime && MEASURE_EACH_STEP) {
	    this->perf_.stop();
	    elapsed.duration[StepType::SETUP] += this->perf_.last();
	    this->perf_.start();
	}
	
	int32_t slice_offset = 1;
	for (int16_t slice = 0; slice < this->depth; slice++) {
	    // Note: ERA and rlc are expected to provide border management
	    	    
	    if (Conf::Double) {
		state.l_len0 = 0;
		state.l_len1 = 0;

		state.l_RLC0[0] = INT16_MAX - 1;
		state.l_RLC0[1] = INT16_MAX - 1;
	    }

	    state.RLC0 = LSL_context.RLC[slice][-1];
	    state.RLC1 = LSL_context.RLC[slice - 1][0];
	    state.RLC2 = LSL_context.RLC[slice - 1][-1];
	    state.RLC3 = LSL_context.RLC[slice - 1][1];
	    
	    if (Conf::ER) {
		ERi = ERp1;
		state.ER0 = ERp1 - er_row_pitch;
		state.ER1 = ERp0;
		state.ER2 = ERp0 - er_row_pitch;
		state.ER3 = ERp0 + er_row_pitch;
	    }
	    
	    if (Conf::ERA) {
		state.ERA0 = LSL_context.ERA[slice][-1];
		state.ERA1 = LSL_context.ERA[slice - 1][0];
		state.ERA2 = LSL_context.ERA[slice - 1][-1];
		state.ERA3 = LSL_context.ERA[slice - 1][1];
	    }
	    
	    state.len0 = LSL_context.Lengths[slice][-1];
	    state.len1 = LSL_context.Lengths[slice - 1][0];
	    state.len2 = LSL_context.Lengths[slice - 1][-1];
	    state.len3 = LSL_context.Lengths[slice - 1][1];
	    
	    for (int16_t row = 0; row < this->height; row++) {

		// Register rotation: doing it at the end is not possible here
		// as it could be outside allocated memory
		state.len3 = LSL_context.Lengths[slice - 1][row + 1];
		
		if (Conf::ERA){
		    state.ERA3 = LSL_context.ERA[slice - 1][row + 1];
		    assert((long long)(state.ERA3) % 4 == 0 && "Wrong alignment");

		    ERAi = ERAi + segment_count / 2;

		    assert((long long)(ERAi) % 4 == 0 && "Wrong alignment");

		    LSL_context.ERA[slice][row] = ERAi;
		}
		
		state.RLC3 = LSL_context.RLC[slice - 1][row + 1];
		
		RLCi = RLCi + segment_count;
		LSL_context.RLC[slice][row] = RLCi;
		
		line = this->img_.template ptr<uint8_t>(slice, row);
		
		assert(RLCi != nullptr);

		//std::cout << "L" << row << "\n";
		if (Conf::ER) {
		    segment_count = RLEAlgo::template Line<0xff>(line, RLCi, ERi, width);
		    ERi[0] = 0;
		    ERi[width] = ERi[width - 1];
		    assert(ERi[-1] == 0);
		} else {
		    segment_count = RLEAlgo::template Line<0xff>(line, RLCi, nullptr, width);
		}
				
		LSL_context.Lengths[slice][row] = segment_count;
		
		
		if (MeasureTime && MEASURE_EACH_STEP) {
		    this->perf_.stop();
		    elapsed.duration[StepType::RLE_SCAN] += this->perf_.last();
		    
		    this->perf_.start();
		}

		if (!Conf::ER) {
		    assert(state.RLC1[state.len1] == INT16_MAX - 1);
		    assert(state.RLC1[state.len1 + 1] == INT16_MAX - 1);
		}		

		
		old_label_count0 = old_label_count1;
		old_label_count1 = LSL_context.ET.Size();


		assert((long long)(state.ERA3) % 4 == 0 && "Wrong alignment");

		// If feature compution is done on-the-fly then it is performed during unification
		// Otherwise, the regular (featureless) unification is called
		// The test with is_same is hacky and should be temporary
		if (std::is_same<FeatureComputation_OTF, FeatureAlgo>::value) {
		    UnifyAlgo::template Unify<LabelsSolver, ConfFeatures>(
			state, RLCi, ERAi, segment_count,
			LSL_context.ET, this->features, row, slice, this->width);
		} else {
		    UnifyAlgo::template Unify<LabelsSolver, ConfFeatures3DNone>(
			state, RLCi, ERAi, segment_count,
			LSL_context.ET, this->features, row, slice, this->width);
		}

		if (std::is_same<FeatureComputation_Line, FeatureAlgo>::value) {
		    for (int l = old_label_count0 + 1; l <= old_label_count1; l++) {
			this->features.template NewComponent3D<ConfFeatures>(l);
		    }
		    
		    FeatureComputation_Line::CalcFeatures<ConfFeatures>(state.RLC0, state.ERA0, state.len0,
									old_label_count0,
									this->features, slice, row - 1, width);
		}


		// Rotations
		if (Conf::Double) {
		    std::swap(state.l_RLC0, state.l_RLC1);
		    std::swap(state.l_ERA0, state.l_ERA1);
		    std::swap(state.l_len0, state.l_len1);
		}
		
		state.RLC2 = state.RLC1;
		state.RLC1 = state.RLC3;
		state.RLC0 = RLCi;		
		
		if (Conf::ERA) {
		    state.ERA2 = state.ERA1;
		    state.ERA1 = state.ERA3;
		    state.ERA0 = ERAi;
		}
		
		if (Conf::ER) {
		    state.ER0 = ERi;
		    ERi += er_row_pitch;
		    state.ER2 = state.ER1;
		    state.ER1 = state.ER3;
		    state.ER3 += er_row_pitch;
		}

		state.len2 = state.len1;
		state.len1 = state.len3;
		state.len0 = segment_count;
		// len3 done at the beginning of the loop
		
		RLCi += 2;
				
		if (MeasureTime && MEASURE_EACH_STEP) {
		    this->perf_.stop();
		    elapsed.duration[StepType::UNIFICATION] += this->perf_.last();
		    this->perf_.start();
		}
	    }


	    old_label_count0 = old_label_count1;
	    old_label_count1 = LSL_context.ET.Size();
	    
	    if (std::is_same<FeatureComputation_Line, FeatureAlgo>::value) {

		for (int l = old_label_count0 + 1; l <= old_label_count1; l++) {
		    //std::cout << "[" << l << "] new\n";
		    this->features.template NewComponent3D<ConfFeatures>(l);
		}
		
		FeatureComputation_Line::CalcFeatures<ConfFeatures>(state.RLC0, state.ERA0, state.len0,
								    old_label_count0,
								    this->features, slice, height - 1, width);

		old_label_count0 = old_label_count1 = LSL_context.ET.Size();
	    }
	    
	    std::swap(ERp0, ERp1);
	}
	
	if (MeasureTime) {
	    this->perf_.stop();
	    
	    if (!MEASURE_EACH_STEP) {
		elapsed.duration[StepType::FIRST_SCAN] = this->perf_.last();
	    }
	}
    }    

    template <bool MeasureSteps>
    void SecondScan(Labeling::StepsDuration& elapsed) {

	if (MeasureSteps) {
	    this->perf_.start();
	    this->samplers.Start();
	}

	unsigned long provisional_labels = LSL_context.ET.Size();

	if (std::is_same<FeatureAlgo, FeatureComputation_OTF>::value) {
	    this->n_labels_ = LSL_context.ET.template Flatten<ConfFeatures>(this->features);
	} else if (std::is_same<FeatureAlgo, FeatureComputation_Line>::value) {
	    this->n_labels_ = LSL_context.ET.template FlattenWithFeatures<ConfFeatures>(this->features);
	} else {
	    this->n_labels_ = LSL_context.ET.Flatten();
	}
	
	unsigned long final_labels = LSL_context.ET.Size();
	assert(provisional_labels >= final_labels);
	
	this->stats.total_labels = final_labels; 
	this->stats.tmp_labels = provisional_labels - final_labels;	

	assert(this->stats.total_labels >= 0);
	assert(this->stats.tmp_labels >= 0);
	
	if (MeasureSteps){
	    this->samplers.Stop();
	    this->perf_.stop();

	    elapsed.duration[StepType::TRANSITIVE_CLOSURE] = this->perf_.last();
	    this->samplers.Store(StepType::TRANSITIVE_CLOSURE, size);
	    
	    this->perf_.start();	
	    this->samplers.Start();
	}

        decltype(this->img_labels_) labels = this->img_labels_;
	RelabelingAlgo::template Relabel<Conf, LabelsSolver>(LSL_context);
	
	if (MeasureSteps) {
	    this->samplers.Stop();	
	    this->perf_.stop();

	    elapsed.duration[StepType::RELABELING] = this->perf_.last();
	    this->samplers.Store(StepType::RELABELING, size);
	}
    }
};

#endif // YACCLAB_LABELING3D_LSL_GENERIC_HPP
