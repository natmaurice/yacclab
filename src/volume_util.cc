#include "volume_util.h"

#include <fstream>
#include <iomanip>

#include <opencv2/imgcodecs.hpp>

#include "file_manager.h"

#include <lsl3dlib/utility.hpp>

#include <iostream>


using namespace cv;
using namespace filesystem;

cv::Mat volread(const cv::String &filename, int flags) {

    constexpr int ALIGNMENT = 64;
    
    std::vector<cv::Mat> planes;
    path vol_path(filename);
    path files_path = vol_path / path("files.txt");		
    std::vector<std::string> planenames;

    {
	std::ifstream is(files_path.string());				// text mode shoud translate \r\n into \n only 
	if (!is) {
	    return Mat();
	}

	std::string cur_filename;
	while (std::getline(is, cur_filename)) {
	    planenames.push_back(cur_filename);
	}
    }
		
    int sz[3];
    int type;

    for (unsigned int plane = 0; plane < planenames.size(); plane++) {
	Mat tmp = imread((vol_path / path(planenames[plane])).string(), flags);
	if (tmp.empty()) {
	    return Mat();
	}
	if (plane == 0) {
	    // We can set volume sizes and type when we see the first plane
	    sz[0] = static_cast<int>(planenames.size());
	    sz[1] = tmp.rows;
	    sz[2] = tmp.cols;
	    type = tmp.type();
	}
	else {
	    if (tmp.rows != sz[1] || tmp.cols != sz[2]) {
		return Mat();
	    }
	}
	planes.push_back(std::move(tmp));
    }
	
    int width = sz[2];
    int height = sz[1];
    int depth = sz[0];    
    int padding = (ALIGNMENT - width % ALIGNMENT) % ALIGNMENT;
    
    int stride = roundup_kpow2(width, ALIGNMENT) + ALIGNMENT;
    //std::cout << "stride = " << stride << ", width = " << width << "\n";
    //int stride = width + padding;

    int psz[] = {
	depth,
	height,
	stride
    };

    size_t size = depth * height * stride;
    
    Mat pvolume(3, psz, type); // Create parent matrix    
    uchar *plane_data = pvolume.data;
    memset(plane_data, 0, size);
    
    for (int slice = 0; slice < depth; slice++) {
	Mat& plane = planes[slice];

	//if (!plane.isContinuous())
	//    return Mat();	
	
	for (int row = 0; row < height; row++) {
	    uint8_t* srcline = plane.ptr<uint8_t>(row);
	    uint8_t* dstline = pvolume.ptr<uint8_t>(slice, row);
	    memcpy(dstline, srcline, plane.size[1]);	    
	}
    }

    cv::Range rect[] = {
	cv::Range(0, depth),
	cv::Range(0, height),
	cv::Range(0, width)
    };
    
    Mat volume = pvolume(rect);
    return volume;
}

bool volwrite(const cv::String& filename, const cv::Mat& volume) {
    if (volume.empty() || volume.dims != 3)
	return false;

	int rows = volume.size[1];
	int cols = volume.size[2];

	std::vector<Mat> planes;

	size_t step = volume.step[1];

	for (int plane = 0; plane < volume.size[0]; plane++) {
		planes.emplace_back(rows, cols, volume.type(), volume.data + volume.step[0] * plane, step);
	}

	path vol_path(filename);

	if (!create_directories(vol_path))
		return false;

	path files_path = vol_path / path("files.txt");

	std::ofstream os(files_path.string(), std::ios::binary);
	if (!os)
		return false;

	for (unsigned i = 0; i < planes.size(); i++) {
		std::ostringstream plane_name;
		plane_name << std::setw(4) << std::setfill('0') << (i + 1) << ".png";						// this should be made general
		if (!imwrite((vol_path / path(plane_name.str())).string(), planes[i]))
			return false;
		os << plane_name.str() << '\n';
	}
	return true;
}
