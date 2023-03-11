#include "yacclab_tensor.h"

#include <cassert>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "file_manager.h"

#include <simdhelpers/term_utils.hpp>
#include <simdhelpers/utils.hpp>


using namespace cv;

Mat1b YacclabTensorInput2D::mat_;
Mat1b YacclabTensorInput3D::mat_;

constexpr size_t ALIGNMENT = 64;

bool YacclabTensorInput2D::ReadBinary(const std::string &filename, ImageMetadata& metadata) {
    if (!filesystem::exists(filesystem::path(filename))) {
        return false;
    }

    Mat1b original_mat = imread(filename, IMREAD_GRAYSCALE);   // Read the file

    // Check if image exist
    if (original_mat.empty()) {
        return false;
    }
    
    int height = original_mat.rows;
    int width = original_mat.cols;
    int stride = roundup_kpow2(width, ALIGNMENT);
    Mat1b parent(height, stride);
    memset(parent.data, 0, parent.dataend - parent.datastart);
    
    cv::Range rect[] = {
	cv::Range(0, height),
	cv::Range(0, width)
    };
    int type = original_mat.type();
    Mat1b img = parent(rect);

    for (int row = 0; row < height; row++) {
	uint8_t* srcline = original_mat.ptr<uint8_t>(row);
	uint8_t* dstline = img.ptr<uint8_t>(row);
	memcpy(dstline, srcline, width);
	memset(dstline + width, 0, stride - width);
    }

    metadata.width = width;
    metadata.height = height;
    metadata.depth = 1;
    
    mat_ = img;
    // Adjust the threshold to make it binary
    //threshold(original_mat, mat_, 100, 1, THRESH_BINARY);
    return true;
}

bool YacclabTensorInput3D::ReadBinary(const std::string &filename, ImageMetadata& metadata) {
    // Image load
    cv::Mat original_mat;
    bool is_dir;
    if (!filesystem::exists(filesystem::path(filename), is_dir))
        return false;
    if (!is_dir) {
        return false;
    }

    original_mat = volread(filename, IMREAD_GRAYSCALE);

    // Check if image exist
    if (original_mat.empty()) {
        return false;
    }

    size_t width = original_mat.size.p[2];
    size_t height = original_mat.size.p[1];
    size_t depth = original_mat.size.p[0];
    
    
    metadata.width = width;
    metadata.height = height;
    metadata.depth = depth;
    
    // Adjust the threshold to make it binary
    // threshold(image, binary_mat, 100, 1, THRESH_BINARY);			
    mat_ = original_mat;
    return true;
}

void print_diff_location(cv::Mat& diff) {
    int width = diff.size.p[2];
    int height = diff.size.p[1];
    int depth = diff.size.p[0];

    for (int slice = 0; slice < depth; slice++) {
	for (int row = 0; row < height; row++) {
	    const uint8_t* line = diff.ptr<uint8_t>(slice, row);
	    for (int col = 0; col < width; col++) {
		if (line[col] != 0) {
		    std::cout << "(" << col << ", " << row << ", " << slice << ")";
		    return;
		}
	    }
	}
    }
}

void print_2d_diff(const cv::Mat& ref, const cv::Mat& mat) {
    int width = ref.cols;
    int height = ref.rows;


    std::cout << "     ";
    for (int col = 0; col < width; col++) {
	std::cout << std::setw(2) << col << " ";
    }
    std::cout << "\n";
    
    for (int row = 0; row < height; row++) {
	const uint32_t* refline = ref.ptr<uint32_t>(row);
	const uint32_t* matline = mat.ptr<uint32_t>(row);
	std::cout << std::setw(2) << row << " | ";
	for (int col = 0; col < width; col++) {
	    int refval = refline[col];
	    int matval = matline[col];
	    if (refval != matval) {
		std::cout << TERM_RED << std::setw(2) << matval << " ";
	    } else {
		std::cout << TERM_GREEN << std::setw(2) << matval << " ";
	    }
	}
	std::cout << TERM_RESET << "\n";
    }
}

void print_3d_diff(const cv::Mat& ref, const cv::Mat& mat) {
    
    int width = ref.size.p[2];
    int height = ref.size.p[1];
    int depth = ref.size.p[0];

    for (int slice = 0; slice < depth; slice++) {
	
	std::cout << "[" << std::setw(2) << slice << "] ";
	for (int col = 0; col < width;  col++) {
	    std::cout << std::setw(2) << col << " ";
	}
	std::cout << "\n";
	
	for (int row = 0; row < height; row++) {
	    const int32_t* refline = ref.ptr<int32_t>(slice, row);
	    const int32_t* matline = mat.ptr<int32_t>(slice, row);

	    std::cout << std::setw(2) << row << " | ";	    
	    
	    for (int col = 0; col < width; col++) {
		int refval = refline[col];
		int matval = matline[col];
		if (refval != matval) {
		    std::cout << TERM_RED << std::setw(2) << matval << " ";
		} else {
		    std::cout << TERM_GREEN << std::setw(2) << matval << " ";
		}
	    }
	    std::cout << TERM_RESET << "\n";
	}
	std::cout << "\n";
    }
}

bool YacclabTensorOutput::Equals(YacclabTensorOutput *other) {
    
    cv::Mat diff = GetMat() != other->GetMat();
    // Equal if no elements disagree
    long zero_count = cv::countNonZero(diff) == 0;
    if (!zero_count) {
	//std::cout << "Diff: " << std::endl << diff << std::endl;	
	cv::Mat binary_diff;

	if (diff.dims == 2) {
	    binary_diff.create(diff.rows, diff.cols, diff.type());
	    cv::threshold(diff, binary_diff, 0, 255, cv::THRESH_BINARY);
	
	    std::cout << "diff" << std::endl;
	    
	    print_2d_diff(GetMat(), other->GetMat());
	    
	    cv::imwrite("diff.png", binary_diff);
	    cv::imwrite("expected.png", GetMat());
	    cv::imwrite("result.png", other->GetMat());
	} else {

	    long depth = diff.size.p[0];
	    
	    int sizes[3] = {diff.size.p[0], diff.size.p[1], diff.size.p[2]};
	    binary_diff.create(3, sizes, diff.type());
	    cv::threshold(diff, binary_diff, 0, 255, cv::THRESH_BINARY);
	
	    //print_diff_location(diff);
	    print_3d_diff(GetMat(), other->GetMat());	    
	    std::cout << "\n";
	    
	    volwrite("diff", diff);
	    volwrite("expected", GetMat());
	    volwrite("result", other->GetMat());
	}
    }
    return zero_count;
}

void YacclabTensorOutput2D::NormalizeLabels(bool label_background) {
    std::map<int, int> map_new_labels;
    int i_max_new_label = 0;

    for (int r = 0; r < mat_.rows; ++r) {
        unsigned * const mat_row = mat_.ptr<unsigned>(r);
        for (int c = 0; c < mat_.cols; ++c) {
            int iCurLabel = mat_row[c];
            if (label_background || iCurLabel > 0) {
                if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
                    map_new_labels[iCurLabel] = ++i_max_new_label;
                }
                mat_row[c] = map_new_labels.at(iCurLabel);
            }
        }
    }
}

void YacclabTensorOutput2D::NormalizeLabelsFeatures(Features& features, bool label_background, bool use_features) {
    std::map<int, int> map_new_labels;
    int i_max_new_label = 0;

    
    for (int r = 0; r < mat_.rows; ++r) {
        unsigned * const mat_row = mat_.ptr<unsigned>(r);
        for (int c = 0; c < mat_.cols; ++c) {
            int iCurLabel = mat_row[c];
            if (label_background || iCurLabel > 0) {
                if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
                    map_new_labels[iCurLabel] = ++i_max_new_label;
                }
                mat_row[c] = map_new_labels.at(iCurLabel);
            }
        }
    }

    if (use_features) {
	Features cpy = features.Copy();
	features.NormalizeFrom(cpy, map_new_labels);
    }
}


void YacclabTensorOutput2D::WriteColored(const std::string &filename) const {
    cv::Mat3b mat_colored(mat_.size());
    for (int r = 0; r < mat_.rows; ++r) {
        unsigned const * const mat_row = mat_.ptr<unsigned>(r);
        Vec3b * const  mat_colored_row = mat_colored.ptr<Vec3b>(r);
        for (int c = 0; c < mat_.cols; ++c) {
            mat_colored_row[c] = Vec3b(mat_row[c] * 131 % 255, mat_row[c] * 241 % 255, mat_row[c] * 251 % 255);
        }
    }
    imwrite(filename, mat_colored);
}

void YacclabTensorOutput3D::NormalizeLabels(bool label_background) {
    std::map<int, int> map_new_labels;
    int i_max_new_label = 0;

    if (mat_.dims == 3) {
        for (int z = 0; z < mat_.size[0]; z++) {
            for (int y = 0; y < mat_.size[1]; y++) {
                unsigned int * img_labels_row = reinterpret_cast<unsigned int *>(mat_.data + z * mat_.step[0] + y * mat_.step[1]);
                for (int x = 0; x < mat_.size[2]; x++) {
                    int iCurLabel = img_labels_row[x];
                    if (label_background || iCurLabel > 0) {
                        if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
                            map_new_labels[iCurLabel] = ++i_max_new_label;
                        }
                        img_labels_row[x] = map_new_labels.at(iCurLabel);
                    }
                }
            }
        }
    }
}

void YacclabTensorOutput3D::NormalizeLabelsFeatures(Features& features, bool label_background, bool use_features) {
    std::map<int, int> map_new_labels;
    int i_max_new_label = 0;

    if (mat_.dims == 3) {
        for (int z = 0; z < mat_.size[0]; z++) {
            for (int y = 0; y < mat_.size[1]; y++) {
                unsigned int * img_labels_row = reinterpret_cast<unsigned int *>(mat_.data + z * mat_.step[0] + y * mat_.step[1]);
                for (int x = 0; x < mat_.size[2]; x++) {
                    int iCurLabel = img_labels_row[x];
                    if (label_background || iCurLabel > 0) {
                        if (map_new_labels.find(iCurLabel) == map_new_labels.end()) {
                            map_new_labels[iCurLabel] = ++i_max_new_label;
                        }
                        img_labels_row[x] = map_new_labels.at(iCurLabel);
                    }
                }
            }
        }
    }
    
    if (use_features) {
	Features cpy = features.Copy();
	features.NormalizeFrom(cpy, map_new_labels);
    }
}


void YacclabTensorOutput3D::WriteColored(const std::string &filename) const {
    cv::Mat img_out(3, mat_.size.p, CV_8UC3);
    for (int z = 0; z < mat_.size[0]; z++) {
        for (int y = 0; y < mat_.size[1]; y++) {
			unsigned int const * const img_labels_row = mat_.ptr<unsigned int>(z, y);
            Vec3b * const img_out_row = img_out.ptr<Vec3b>(z, y);
            for (int x = 0; x < mat_.size[2]; x++) {
                img_out_row[x] = Vec3b(img_labels_row[x] * 131 % 255, img_labels_row[x] * 241 % 255, img_labels_row[x] * 251 % 255);
            }
        }
    }
    volwrite(filename, img_out);
}


#if defined YACCLAB_WITH_CUDA

cuda::GpuMat YacclabTensorInput2DCuda::d_mat_;

cuda::GpuMat3 YacclabTensorInput3DCuda::d_mat_;

bool YacclabTensorInput2DCuda::ReadBinary(const std::string &filename, ImageMetadata& metadata) {
    if (!YacclabTensorInput2D::ReadBinary(filename, metadata))
        return false;
    d_mat_.upload(YacclabTensorInput2D::mat_);
    return true;
}

bool YacclabTensorInput3DCuda::ReadBinary(const std::string &filename, ImageMetadata& metadata) {
    if (!YacclabTensorInput3D::ReadBinary(filename, metadata))
        return false;
    d_mat_.upload(YacclabTensorInput3D::mat_);
    return true;
}

#endif
