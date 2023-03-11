#ifndef YACCLAB_CALC_FEATURES_HPP
#define YACCLAB_CALC_FEATURES_HPP

#include <lsl3dlib/features.hpp>
#include <lsl3dlib/compat.hpp>
#include <opencv2/core.hpp>

template <typename ConfFeatures>
void calc_features2d_post(const cv::Mat1i& labels, size_t label_count,
			Features& features) {

    int width, height;

    width = labels.cols;
    height = labels.rows;

    // Don't iterate over image if no feature to compute
    if (std::is_same<ConfFeatures, ConfFeatures2DNone>::value) {
	return;
    }
    
    std::fill(features.lo_col, features.lo_col + label_count, UINT16_MAX);
    std::fill(features.lo_row, features.lo_row + label_count, UINT16_MAX);
    std::fill(features.lo_col, features.lo_col + label_count, UINT16_MAX);
    std::fill(features.lo_row, features.lo_row + label_count, UINT16_MAX);
    
    for (int row = 0; row < height; row++) {
	const uint32_t* restrict line = labels.ptr<uint32_t>(row);
	for (int col = 0; col < width; col++) {
	    uint32_t l = line[col];
	    if (l > 0) {

		features.AddPoint2D<ConfFeatures>(l, col, row);
	    }
	}
    }
}


struct CalcFeatures3DPixels {

    struct Conf {
	using Label_t = int32_t;
    };
    
    template <typename LabelsSolver, typename ConfFeatures, bool RelabelingDone = true>
    static void CalcFeatures(const cv::Mat1i& labels, LabelsSolver& ET, Features& features) {

	// Do not iterate over the image if no feature to compute
	if (!std::is_same<ConfFeatures, ConfFeatures3DNone>::value) {
	    int width, height, depth;
	    GetMatSize(labels, width, height, depth);

	    int label_count = ET.Size();
	    features.Init<ConfFeatures>(label_count + 1);

	    for (int slice = 0; slice < depth; slice++) {
		for (int row = 0; row < height; row++) {
		    const Conf::Label_t* line = labels.ptr<Conf::Label_t>(slice, row);
		    for (int col = 0; col < width; col++) {
			Conf::Label_t l = line[col];

			if (!RelabelingDone) {
			    l = ET.GetLabel(l);
			}
		    
			if (l > 0) {
			    assert(label_count >= l);
			    features.AddPoint3D<ConfFeatures>(l, col, row, slice);
			}
		    }
		}
	    }
	}
    }
};

// Pixel-based features analysis:
// If RelabelingDone == true: make relabeling optional 
struct CalcFeatures3DBlock_1x1x2 {
    struct Conf {
	using Label_t = int32_t;
    };

    template <typename LabelsSolver, typename ConfFeatures, bool RelabelingDone = true>
    static void CalcFeatures(const cv::Mat1b& img, const cv::Mat1i& labels, LabelsSolver& ET, Features& features) {

	// Do not iterate over the image no feature to compute
	if (!std::is_same<ConfFeatures, ConfFeatures3DNone>::value) {
	    int width, height, depth;
	    GetMatSize(labels, width, height, depth);

	    int label_count = ET.Size();
	    features.Init<ConfFeatures>(label_count + 1);

	    for (int slice = 0; slice < depth; slice++) {
		for (int row = 0; row < height; row++) {
		    const Conf::Label_t* labels_line = labels.ptr<Conf::Label_t>(slice, row);
		    const uint8_t* img_line = img.ptr<uint8_t>(slice, row);
		    
		    for (int col = 0; col < width; col += 2) {
			Conf::Label_t label = labels_line[col];
			if (label > 0) {

			    if (!RelabelingDone) {
				label = ET.GetLabel(label);
			    }
			    
			    if (img_line[col] > 0) {
				features.AddPoint3D<ConfFeatures>(label, col, row, slice);
			    }

			    if (img_line[col + 1] > 0) {
				features.AddPoint3D<ConfFeatures>(label, col, row, slice);
			    }
			}
			
		    }
		    if (width % 2 == 1) { // If number of column is odd
			Conf::Label_t label = labels_line[width - 1];			
			if (label > 0) {
			    
			    if (!RelabelingDone) {
				label = ET.GetLabel(label);
			    }
			    
			    int col = width - 1;
			    features.AddPoint3D<ConfFeatures>(label, col, row, slice);
			}
		    }
		}
	    }
	}
    }    
};



template <typename LabelsSolver, typename ConfFeatures>
void calc_features3d_pixel(const cv::Mat1i& labels, LabelsSolver& ET, Features& features) {
    int width, height, depth;
    GetMatSize(labels, width, height, depth);

    int label_count = ET.Size();
    features.Init<ConfFeatures>(label_count + 1);

    for (int slice = 0; slice < depth; slice++) {
	for (int row = 0; row < height; row++) {
	    const uint32_t* restrict line = labels.ptr<uint32_t>(slice, row);
	    for (int col = 0; col < width; col++) {
		uint32_t l = line[col];
		l = ET.GetLabel(l);
		
		if (l > 0) {
		    assert(label_count >= l);
		    features.AddPoint3D<ConfFeatures>(l, col, row, slice);
		}
	    }
	}
    }
}

template <typename ConfFeatures>
void calc_features3d_post(const cv::Mat1i& labels, size_t label_count,
			Features& features) {

    int width, height, depth;
    GetMatSize(labels, width, height, depth);       
    
    features.Init<ConfFeatures>
	(label_count + 1); // Initialize features [0; label_count[
    
    for (int slice = 0; slice < depth; slice++) {
	for (int row = 0; row < height; row++) {
	    const uint32_t* restrict line = labels.ptr<uint32_t>(slice, row);
	    for (int col = 0; col < width; col++) {
		uint32_t l = line[col];
		if (l > 0) {

		    assert(label_count >= l);
		    
		    features.AddPoint3D<ConfFeatures>(l, col, row, slice);
		}
	    }
	}
    }    
}

#endif // YACCLAB_CALC_FEATURES_HPP
