// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "config_data.h"
#include "file_manager.h"
#include "labeling_algorithms.h"
#include "memory_tester.h"
#include "performance_evaluator.h"
#include "progress_bar.h"
#include "stream_demultiplexer.h"
#include "system_info.h"
#include "utilities.h"
#include "yacclab_tests.h"

#include <lsl3dlib/papi_helper.hpp>
#include <lsl3dlib/perf-helper.hpp>

#include <omp.h>

using namespace cv;
using namespace ::filesystem;

using string = std::string;
using error_code = std::error_code;

int main()
{
    // Redirect cv exceptions
#if OPENCV_VERSION_MAJOR >= 4
    redirectError(RedirectCvError);
#else
    cvRedirectError(RedirectCvError);
    cvRedirectError(RedirectCvError);
#endif
    
    // Hide cursor from console
    //HideConsoleCursor();

    // To handle filesystem errors
    error_code ec;

    // Create StreamDemultiplexer object in order
    // to print output on both stdout and log file
    string logfile = "log.txt";
    std::ofstream os(logfile);
    if (os.is_open()) {
        dmux::cout.AddStream(os);
    }

    OutputBox ob_setconf("Reading Configuration Parameters");
    // Read yaml configuration file
    const string config_file = "config.yaml";
    FileStorage fs;
    try {
        fs.open(config_file, FileStorage::READ);
    }
    catch (const cv::Exception&) {
        exit(EXIT_FAILURE);  // Error redirected,
                             // OpenCV redirected function will
                             // print the error on stdout
    }

    if (!fs.isOpened()) {
        ob_setconf.Cerror("Failed to open '" + config_file + "'");
        // EXIT_FAILURE
    }

    ob_setconf.Cmessage("Setting Configuration Parameters DONE");
    ob_setconf.CloseBox();

    // Load configuration data from yaml
    ConfigData cfg(fs);

    // Release FileStorage
    fs.release();


#if YACCLAB_USE_PAPI
    papih_init();
    papih_configure();
#endif // YACCLAB_USE_PAPI

#if YACCLAB_USE_PERF
    perf_helper_create();
#endif // YACCLAB_USE_PERF
    
    /*************************************************************************/
    /*  Configuration parameters check                                       */
    /*************************************************************************/


    for (auto &mode_cfg : cfg.mode_config_vector) {

	YacclabTests test_perf = YacclabTests(mode_cfg, cfg.global_config, ec);

        test_perf.InitialOperations();

	// Correctness test
	if (mode_cfg.perform_correctness) {
	    if (mode_cfg.perform_check_8connectivity_std) {
		test_perf.CheckPerformLabeling();
	    }

	    if (mode_cfg.perform_check_8connectivity_ws) {
		test_perf.CheckPerformLabelingWithSteps();
	    }

			if (mode_cfg.perform_check_8connectivity_mem) {
				test_perf.CheckPerformLabelingMem();
			}
		}

		// Average test
		if (mode_cfg.perform_average) {
			test_perf.AverageTest();
		}

		// Average with steps test
		if (mode_cfg.perform_average_ws) {
			test_perf.AverageTestWithSteps();
		}

		// Density test
		if (mode_cfg.perform_density) {
			test_perf.DensityTest();
		}

		// Granularity test
		if (mode_cfg.perform_granularity) {
			test_perf.GranularityTest();
		}
		// Memory test
		if (mode_cfg.perform_memory) {
			test_perf.MemoryTest();
		}

		if (mode_cfg.perform_parallel_average_ws) {
		    test_perf.ParallelAverageTestWithSteps();
		}

		if (mode_cfg.perform_parallel_granularity) {
		    test_perf.ParallelGranularityTest();
		}
		
        // There should be better places for this
		//LabelingMapSingleton::GetLabeling(mode_cfg.ccl_existing_algorithms[0])->GetInput()->Release();

		// Latex Generator
		if (mode_cfg.perform_average || mode_cfg.perform_average_ws || mode_cfg.perform_density || mode_cfg.perform_memory || mode_cfg.perform_granularity) {
			test_perf.LatexGenerator();
		}

		// Copy log file into output folder
		dmux::cout.flush();
		copy(path(logfile), cfg.global_config.glob_output_path / mode_cfg.mode_output_path / path(logfile), ec);
	    if (mode_cfg.perform_check_8connectivity_mem) {
		test_perf.CheckPerformLabelingMem();
	    }
	}

    ShowConsoleCursor();

#if YACCLAB_USE_PAPI
    papih_exit();
#endif // YACCLAB_USE_PAPI

#if YACCLAB_USE_PERF
    perf_helper_destroy();
#endif // YACCLAB_USE_PERF
    
    return EXIT_SUCCESS;
}
