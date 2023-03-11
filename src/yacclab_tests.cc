#include "yacclab_tests.h"

#include <cstdint>

#include <random>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sched.h>
#include <unistd.h>
#include <omp.h>

#include "memory_tester.h"
#include "step_types.h"

#ifndef YACCLAB_TIE_CPU_CORE
#define YACCLAB_TIE_CPU_CORE 1
#endif // TIE_CPU_CORE


void tie_to_cpu() {

#ifdef YACCLAB_TIE_CPU_CORE
    
#ifdef __linux__
    cpu_set_t cpuset;
    pid_t this_pid = getpid();
    constexpr int CPU_ID = YACCLAB_TIE_CPU_CORE;
    CPU_SET(CPU_ID, &cpuset);
    constexpr int CPU_COUNT = 1;
    
    int ret = sched_setaffinity(this_pid, CPU_COUNT, &cpuset);
    if (ret) {
	perror("Could not set CPU affinity");
    } else {
	std::cout << "Tied to CPU " << CPU_ID << "\n";
    }
    
#else
#warning "OS/Program does not support tying program to CPU"
#endif // __linux__

#endif // YACCLAB_TIE_CPU_CORE

    
}


// This function take a Mat1d of results and save it in the  specified output-stream
bool SaveBroadOutputResults(std::map<std::string, cv::Mat1d>& results, const std::string& o_filename, const cv::Mat1i& labels, 
	const std::vector<std::pair<std::string, bool>>& filenames, const std::vector<std::string>& ccl_algorithms) {
	std::ofstream os(o_filename);
	if (!os.is_open()) {
		return false;
	}

	// To set heading file format
	os << "#" << '\t';
	for (const auto& algo_name : ccl_algorithms) {

	    // Calculate the max of the columns to find unused steps
	    cv::Mat1d results_reduced(1, results.at(algo_name).cols);
#if OPENCV_VERSION_MAJOR >= 4
	    cv::reduce(results.at(algo_name), results_reduced, 0, cv::REDUCE_MAX);
#else
	    cv::reduce(results.at(algo_name), results_reduced, 0, CV_REDUCE_MAX);
	    cv::reduce(results.at(algo_name), results_reduced, 0, CV_REDUCE_MAX);
#endif

	    for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
		StepType step = static_cast<StepType>(step_number);
		double column_value(results_reduced(0, step_number));
		if (column_value != std::numeric_limits<double>::max()) {
		    os << algo_name + "_" << Step(step) << '\t';
		}
	    }
	}
	os << '\n';

	for (unsigned files = 0; files < filenames.size(); ++files) {
		if (filenames[files].second) {
			os << filenames[files].first << '\t';
			unsigned i = 0;
			for (const auto& algo_name : ccl_algorithms) {
				for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
					if (results.at(algo_name)(files, step_number) != std::numeric_limits<double>::max()) {
						os << results.at(algo_name)(files, step_number) << '\t';
					}
					else {
						// Step not held, skipped
						//os << 0 << '\t';
					}
				}
				++i;
			}
			os << '\n';
		}
	}

	return true;
}

bool SaveBroadOutputResultsMat(const cv::Mat1d& results, const std::string& o_filename, const cv::Mat1i& labels, 
	const std::vector<std::pair<std::string, bool>>& filenames, const std::vector<std::string>& ccl_algorithms) {
	std::ofstream os(o_filename);
	if (!os.is_open()) {
		return false;
	}

	// To set heading file format
	os << "#";
	for (const auto& algo_name : ccl_algorithms) {
		os << '\t' << algo_name;
	}
	os << '\n';
	// To set heading file format

	for (unsigned files = 0; files < filenames.size(); ++files) {
		if (filenames[files].second) {
			os << filenames[files].first << '\t';
			for (unsigned i = 0; i < ccl_algorithms.size(); ++i) {
				os << results(files, i) << '\t';

			}
			os << '\n';
		}
	}
	return true;
}


// To calculate average times and write it on the specified file
bool SaveBroadAverageWithStepsResults(
    std::map<std::string, cv::Mat1d>& results,
    const std::string& o_filename, const cv::Mat1i& labels, 
    const std::vector<std::pair<std::string, bool>>& filenames,
    const std::vector<std::string>& ccl_algorithms) {
    
    std::ofstream os(o_filename);
    if (!os.is_open()) {
	dmux::cout << "Unable to save broad average results" << '\n';
	return false;
    }

    // Write heading string in output stream
    os << "#Algorithm" << '\t' << "Image" << '\t';
    for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
	StepType step = static_cast<StepType>(step_number);
	os << Step(step) << '\t';
    }
    os << "Total" << '\n';


    for (const auto& algo_name : ccl_algorithms) {
	for (unsigned int files = 0; files < filenames.size(); files++) {

	    if (!filenames[files].second) {
		continue;
	    }
	    auto filename = filenames[files].first;
	    double cumulative_sum{ 0.0 };

	    // Gnuplot requires double-escaped name when underscores are encountered
	    //{
	    //    string algo_name_double_escaped{ algo_name };
	    //    std::size_t found = algo_name_double_escaped.find_first_of("_");
	    //    while (found != std::string::npos) {
	    //        algo_name_double_escaped.insert(found, "\\\\");
	    //        found = algo_name_double_escaped.find_first_of("_", found + 3);
	    //    }
	    //    os << algo_name_double_escaped << '\t';
	    //}
	    os << DoubleEscapeUnderscore(std::string(algo_name)) << '\t' << filename << "\t";
	
	    for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
		auto duration = results.at(algo_name)(files, step_number);
		if (duration == std::numeric_limits<double>::max()) {
		    duration = 0.0; // Step not performed => 0.0
		}
		cumulative_sum += duration;
		os << std::fixed << std::setprecision(8) << duration << '\t';
	    
	    }
	    os << std::fixed << std::setprecision(8) << cumulative_sum;
	
	    os << '\n';	
	}
    }
    os.close();
    return true;
}


// To calculate average times and write it on the specified file
void YacclabTests::SaveAverageWithStepsResults(const std::string& os_name,
					       const std::string& dataset_name, bool rounded) {
	std::ofstream os(os_name);
	if (!os.is_open()) {
		dmux::cout << "Unable to save average results" << '\n';
		return;
	}

	// Write heading string in output stream
	os << "#Algorithm" << '\t';
	for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
		StepType step = static_cast<StepType>(step_number);
		os << Step(step) << '\t';
	}
	os << "Total" << '\n';

	const auto& results = average_ws_results_.at(dataset_name);

	for (int r = 0; r < results.rows; ++r) {
		const auto& algo_name = mode_cfg_.ccl_average_ws_algorithms[r];
		double cumulative_sum{ 0.0 };

		// Gnuplot requires double-escaped name when underscores are encountered
		//{
		//    string algo_name_double_escaped{ algo_name };
		//    std::size_t found = algo_name_double_escaped.find_first_of("_");
		//    while (found != std::string::npos) {
		//        algo_name_double_escaped.insert(found, "\\\\");
		//        found = algo_name_double_escaped.find_first_of("_", found + 3);
		//    }
		//    os << algo_name_double_escaped << '\t';
		//}
		os << DoubleEscapeUnderscore(std::string(algo_name)) << '\t';

		for (int c = 0; c < results.cols; ++c) {
			if (rounded) {
				cumulative_sum += floor(results(r, c) * 100.00 + 0.5) / 100.00;
				os << std::fixed << std::setprecision(2) << results(r, c) << '\t';
			}
			else {
				cumulative_sum += results(r, c);
				os << std::fixed << std::setprecision(8) << results(r, c) << '\t';
			}
		}
		// Write cumulative_sum as total
		if (rounded) {
			os << std::fixed << std::setprecision(2) << cumulative_sum;
		}
		else {
			os << std::fixed << std::setprecision(8) << cumulative_sum;
		}
		os << '\n';
	}
	os.close();
}

void YacclabTests::SaveGranularityResults(const std::string& os_name,
					  const std::string& dataset_name,
					  const GranularityArgs& args) {
    
    // For GRANULARITY RESULT
    for (unsigned g = 1; g <= args.g; ++g) {
	std::string cur_granularity_os = os_name + "_" + std::to_string(g) + ".txt";
	std::ofstream granularity_os(cur_granularity_os);

	if (!granularity_os.is_open()) {
	    dmux::cout << "Unable to open '" << cur_granularity_os << "'\n";
	    return;
	}

	granularity_os << "# density";
	for (const auto& algo : mode_cfg_.ccl_average_algorithms) {
	    granularity_os  << '\t' << algo;
	}
	granularity_os << '\n';

	for (unsigned d = 0; d < args.density; ++d) {	    
	    granularity_os << std::fixed << std::setprecision(5) << /*real_densities[g - 1][d]*/ d << '\t';
	    for (unsigned a = 0; a < mode_cfg_.ccl_average_algorithms.size(); ++a) {
		if (args.dims == 2) {		    
		    granularity_os << std::fixed << std::setprecision(8)
				   << (granularity_results_[dataset_name].at<
				       cv::Vec<double, TestsConf::kGranularities2D>>(d, a)[g - 1]
				       / (float) TestsConf::kSamples2D) << '\t';
		}
		else if (args.dims == 3) {
		    granularity_os << std::fixed << std::setprecision(8)
				   << (granularity_results_[dataset_name].at<
				       cv::Vec<double, TestsConf::kGranularities3D>>(d, a)[g - 1]
				       / (float) TestsConf::kSamples3D) << '\t';
		}
	    }
	    granularity_os << '\n';
	}
	granularity_os.close();    
    }
}


struct ThreadResults {

    struct Result {
	int file_id;
	int algo_id;
	int thread_count;
	std::vector<double> thread_durations;

	Result(int file_id, int algo_id, int thread_count, const std::vector<double>& thread_durations) :
	    file_id(file_id), algo_id(algo_id), thread_count(thread_count),
	    thread_durations(thread_durations) {
	    // ctor
	}
	
    };

    std::vector<ThreadResults::Result> results;

    void Reserve(unsigned capacity) {
	results.reserve(capacity);	
    }
    
    void Write(int file, int algo, int thread_count, const std::vector<double>& thread_durations) {
	results.emplace_back(file, algo, thread_count, thread_durations);
    }
    
    bool Serialize(const std::string& o_filename,
		   const std::vector<std::pair<std::string, bool>>& filenames,
		   const std::vector<std::string>& algorithms) {

	std::ofstream os(o_filename);
	if (!os.is_open()) {
	    return false;
	}

	// Shape: (ex)
	// =========================================================
	// Image   | Algorithm      | ThreadCount | ThreadID | Time
	// =========================================================
	// test_gt | LSL3D_PARALLEL | 8           | 0        | 0.24
	// test_gt | LSL3D_PARALLEL | 8           | 1        | 0.38
	// ...
	// =========================================================
	// Separator: \t
	
	os << "Image\tAlgorithm\tThreadCount\tThreadID\tTime\n";
	
	for (unsigned i = 0; i < results.size(); i++) {
	    
	    const ThreadResults::Result& res = results[i];

	    const std::string image = filenames[res.file_id].first;
	    const std::string algo = algorithms[res.algo_id];
	    const int thread_count = res.thread_count;

	    if (!filenames[res.file_id].second) {
		continue; // Image not available
	    }
	    
	    for (unsigned tid = 0; tid < res.thread_durations.size(); tid++) {
		os << image << "\t" << algo << "\t" << thread_count << "\t" << tid << "\t"
		   << res.thread_durations[tid] << "\n";
	    }	    
	}
	return true;
    }
    

};

struct RawGranularityResults {

    struct Result {
	int file_id;
	int algo_id;
	int thread_count;
	
	short density_percent;
	short granularity;
        short image_width;
	short image_height;
	short image_depth;
	
	double value;
	int label_count;

	Result(int file_id, int algo_id, int thread_count, int density, int granularity,
	       int image_width, int image_height, int image_depth, double value, int label_count) :
	    file_id(file_id), algo_id(algo_id), thread_count(thread_count), density_percent(density),
	    granularity(granularity), image_width(image_width), image_height(image_height),
	    image_depth(image_depth), value(value), label_count(label_count) {	    
	    
	}
    };

    std::vector<RawGranularityResults::Result> results;

    void Reserve(int capacity) {
	results.reserve(capacity);
    }
    
    void Write(int file_id, int algo_id, int thread_count, int density, int granularity,
	       int image_width, int image_height, int image_depth, double value, int label_count) {
	results.emplace_back(file_id, algo_id, thread_count, density, granularity, image_width,
			     image_height, image_depth, value, label_count);
    }
    
    bool Serialize(const std::string& o_filename,
		   const std::vector<std::pair<std::string, bool>>& filenames,
		   const std::vector<std::string>& algorithms) {

	std::ofstream os(o_filename);
	if (!os.is_open()) {
	    return false;
	}

	// Shape: (ex)
	// =======================================================================================================
	// Image  | width | height | depth | size  | density | granularity | labels | Algorithm | Threads | Time |
	// =======================================================================================================
	// 160500 | 512   | 512    | 512   | 512^3 | 50      | 16          | 3      | LSL3D     | 2       |      |
	// 160502 | 512   | 512    | 512   | 512^3 | 50      | 16          | 3      | LSL3D     | 2       |      |
	// 160503 | 512   | 512    | 512   | 512^3 | 50      | 16          | 3      | LSL3D     | 2       |      |
	// 160501 | 512   | 512    | 512   | 512^3 | 50      | 16          | 3      | LSL3D     | 2       |      |

	os << "Image\twidth\theight\tdepth\tsize\tdensity\tgranularity\tlabels\tAlgorithm\tThreads\tTime\n";

	for (unsigned i = 0; i < results.size(); i++) {

	    const RawGranularityResults::Result& res = results[i];

	    const std::string image = filenames[res.file_id].first;
	    const std::string algo = algorithms[res.algo_id];

	    if (!filenames[res.file_id].second) {
		continue;
	    }

	    const long image_size = static_cast<long>(res.image_width) * res.image_height * res.image_depth;

	    os << image << "\t" << res.image_width << "\t" << res.image_height << "\t"
	       << res.image_depth << "\t" << image_size << "\t" << res.density_percent << "\t"
	       << res.granularity << "\t" << res.label_count << "\t" << algo << "\t" << res.value
	       << "\n";
	}
	
	return true;
    }
};

struct Results {
    std::map<std::string, cv::Mat1d> current_res;
    std::map<std::string, cv::Mat1d> current_res_norm;
    std::map<std::string, cv::Mat1d> cur_cycles_res;
    std::map<std::string, cv::Mat1d> cur_dcache_res;
    std::map<std::string, cv::Mat1d> cur_l2dm_res;
    std::map<std::string, cv::Mat1d> cur_l3m_res;

    std::map<std::string, cv::Mat1d> cur_icache_res;
    std::map<std::string, cv::Mat1d> cur_branchm_res;
    std::map<std::string, cv::Mat1d> cur_stalls_res;
    std::map<std::string, cv::Mat1d> min_res;
    std::map<std::string, cv::Mat1d> min_res_norm;
    std::map<std::string, cv::Mat1d> ccy_res;
    std::map<std::string, cv::Mat1d> wstalls_res;
    std::map<std::string, cv::Mat1d> instr_res;
    std::map<std::string, cv::Mat1d> brcnt_res;
    
    std::map<std::string, cv::Mat1d> temp_labels_res;

    cv::Mat1d events;
    
    size_t pixcount = 0;
    
    void Init(const std::vector<std::string>& algorithms, unsigned int filenames_size) {

	/*int sizes[] = {
	    MAX_PAPI_EVENTS,
	    static_cast<int>(filenames_size),
	    StepType::ST_SIZE
	};
	
	events.create(3, sizes);*/

	/*for (int evnt = 0; evnt < MAX_PAPI_EVENTS; evnt++) {
	    for (int alg = 0; alg < algorithms.size(); alg++) {
		for (int step = 0; step < StepType::ST_SIZE; step++) {
		    events(evnt, alg, step) = std::numeric_limits<double>::max();
		}
	    }
	    }*/
	
	for (const auto& algo_name : algorithms) {
	    current_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
					       std::numeric_limits<double>::max());
	    current_res_norm[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
						   std::numeric_limits<double>::max());
	    
	    min_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
					   std::numeric_limits<double>::max());
	    
	    min_res_norm[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
						std::numeric_limits<double>::max());
	    
	    cur_cycles_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
						  std::numeric_limits<double>::max());
	    
	    cur_dcache_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
						  std::numeric_limits<double>::max());

	    cur_l2dm_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
						std::numeric_limits<double>::max());
	    
	    cur_l3m_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
						std::numeric_limits<double>::max());
	    
	    cur_icache_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
						  std::numeric_limits<double>::max());
	    
	    cur_branchm_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
						   std::numeric_limits<double>::max());

	    cur_stalls_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
						   std::numeric_limits<double>::max());

	    ccy_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
					   std::numeric_limits<double>::max());
	    
	    wstalls_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
					   std::numeric_limits<double>::max());

	    instr_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
					       std::numeric_limits<double>::max());

	    brcnt_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE,
					     std::numeric_limits<double>::max());


	    temp_labels_res[algo_name] = cv::Mat1d(filenames_size, StepType::ST_SIZE, 0.0);
	}
    }

    void SaveBroadAverageSteps(const std::vector<std::string>& algorithms,
			  path& output_middle_results_path,
			  const std::string& dataset_name,
			  const std::string& middle_results_suffix,
			  unsigned int test,
			  const cv::Mat1i& labels,
			  const std::vector<std::pair<std::string, bool>>& filenames) {
	
	std::string output_middle_results_file =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test) + ".txt")).string();

	std::string output_norm_file =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test) + "_norm.txt")).string();
	
	std::string output_avg_file =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test) + "_avg.txt")).string();
	
	std::string middle_results_avg_norm_file =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test) + "_avg_norm.txt")).string();

	std::string output_middle_results_cycles_file =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test)
					       + "_cycles.txt")).string();
	std::string output_middle_results_cache_file =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test)
					       + "_l1dm.txt")).string();

	std::string output_middle_results_l2dm =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test)
					       + "_l2dm.txt")).string();
	std::string output_middle_results_l3m =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test)
					       + "_l3m.txt")).string();

	
	std::string output_middle_results_icache_file =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test)
					       + "_l1im.txt")).string();
	
	std::string output_middle_results_branchm_file =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test)
					       + "_branchm.txt")).string();

	std::string stalls_output =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test)
					       + "_stalls.txt")).string();
	
	std::string insr_output =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test)
					       + "_instr.txt")).string();

	
	std::string output_min_norm =
	    (output_middle_results_path / path(dataset_name + middle_results_suffix
					       + "_" + std::to_string(test)
					       + "_min_norm.txt")).string();
	
	std::string output_ccy = (output_middle_results_path / path(dataset_name + middle_results_suffix
								    + "_" + std::to_string(test)
								    + "_ccy.txt")).string();

	std::string output_wstalls = (output_middle_results_path / path(dataset_name + middle_results_suffix
								    + "_" + std::to_string(test)
								    + "_ccy.txt")).string();

	std::string output_brcnt = (output_middle_results_path / path(dataset_name + middle_results_suffix
								    + "_" + std::to_string(test)
								    + "_brnct.txt")).string();

	
	std::string output_tmplabels = (output_middle_results_path / path(dataset_name + middle_results_suffix
								    + "_" + std::to_string(test)
								    + "_tmplabels.txt")).string();

	
	SaveBroadOutputResults(current_res, output_middle_results_file, labels, filenames,
			       algorithms);

	SaveBroadOutputResults(current_res_norm, output_norm_file, labels, filenames,
			       algorithms);


	/*for (int evnt = 0; evnt < MAX_PAPI_EVENTS; evnt++) {
	    std::pair<int, int> event = g_Events[evnt];
	    int eventid = event.first;
	    int eventcode = event.second;
	    
	    if (eventcode < 0) {
		continue;
	    }
	    
	    const std::string eventname = find_event_abbrv(eventid);
	    const std::string o_filename = (output_middle_results_path
		/ path(dataset_name + middle_results_suffix + "_" + std::to_string(test) + "_"
		       + eventname + ".txt")).string();
	    
	    SaveBroadAverageWithStepsResults(events(evnt), o_filename, labels, filenames, algorithms);
	    }*/
	
	SaveBroadAverageWithStepsResults(current_res, output_avg_file, labels,
					 filenames, algorithms);

	SaveBroadAverageWithStepsResults(current_res_norm, middle_results_avg_norm_file, labels,
					 filenames, algorithms);
	
	SaveBroadAverageWithStepsResults(cur_cycles_res, output_middle_results_cycles_file, labels,
					 filenames, algorithms);
	
	SaveBroadAverageWithStepsResults(cur_dcache_res, output_middle_results_cache_file, labels,
					 filenames, algorithms);
	
	SaveBroadAverageWithStepsResults(cur_l2dm_res, output_middle_results_l2dm, labels,
					 filenames, algorithms);
	
	SaveBroadAverageWithStepsResults(cur_l3m_res, output_middle_results_l3m, labels,
					 filenames, algorithms);


	
	SaveBroadAverageWithStepsResults(cur_icache_res, output_middle_results_icache_file, labels,
					 filenames, algorithms);
	
        SaveBroadAverageWithStepsResults(cur_branchm_res, output_middle_results_branchm_file, labels,
					 filenames, algorithms);
	
	SaveBroadAverageWithStepsResults(cur_stalls_res, stalls_output, labels,
					 filenames, algorithms);
		
	SaveBroadAverageWithStepsResults(ccy_res, output_ccy, labels,
					 filenames, algorithms);
	
	SaveBroadAverageWithStepsResults(wstalls_res, output_wstalls, labels,
					 filenames, algorithms);

	SaveBroadAverageWithStepsResults(brcnt_res, output_brcnt, labels,
					 filenames, algorithms);

	SaveBroadAverageWithStepsResults(temp_labels_res, output_tmplabels, labels,
					 filenames, algorithms);

    }

    void SaveGranularity(const std::map<std::string, cv::Mat1d>& res,
			 const std::vector<std::string>& algorithms,
			 const std::string& current_output_path,
			 const std::vector<std::pair<std::string, bool>>& filenames,
			 const GranularityArgs& args) {
	

	
	for (int g = 1; g <= args.g; g++) {

	    const std::string cur_granularity_name = current_output_path + "_"
		+ std::to_string(g) + ".txt";
	    
	    std::string cur_granularity_os = current_output_path + "_" + std::to_string(g) + ".txt";
	    std::ofstream granularity_os(cur_granularity_os);
	    
	    if (!granularity_os.is_open()) {
		dmux::cout << "Unable to open '" << cur_granularity_os << "'\n";
		return;
	    }

	    granularity_os << "# density";
	    for (const auto& algo : algorithms) {
		granularity_os  << '\t' << algo;
	    }
	    granularity_os << '\n';

	    for (unsigned int file = 0; file < filenames.size(); file++) {		

		const auto& image = filenames[file];
		const std::string& image_name = image.first;
	        bool present = image.second;		
		int granularity = stoi(image_name.substr(0, 2));
		int density = stoi(image_name.substr(2, 3));

		if (!present) {
		    continue;
		}
		if (granularity != g) {
		    continue;
		}

		granularity_os << std::fixed << std::setprecision(5) << density << '\t';
		for (const auto& algo: algorithms) {
		    double minval = std::numeric_limits<double>::max();
		    const auto& img_it = res.find(algo);
		    if (img_it == res.end()) {
			assert(false);
			continue;
		    }			
		    
		    const cv::Mat1d& img_res = img_it->second;
		    const int step = StepType::ALL_SCANS;
		    double val = img_res.at<double>(file, step);
		    
		    minval = std::min(minval, val);
		    
		    granularity_os << std::fixed << std::setprecision(8)
				   << minval << '\t';		    
		}
		granularity_os << '\n';
	    }
	    
	    granularity_os.close();	
	}
    }

    void SaveGranularitySteps(const std::map<std::string, cv::Mat1d>& res,
			      const std::vector<std::string>& algorithms,
			      const std::string& current_output_path,
			      const std::vector<std::pair<std::string, bool>>& filenames,
			      const GranularityArgs& args) {	
	
	for (int g = 1; g <= args.g; g++) {

	    const std::string cur_granularity_name = current_output_path + "_"
		+ std::to_string(g) + ".txt";
	    
	    std::string cur_granularity_os = current_output_path + "_" + std::to_string(g) + ".txt";
	    std::ofstream granularity_os(cur_granularity_os);
	    
	    if (!granularity_os.is_open()) {
		dmux::cout << "Unable to open '" << cur_granularity_os << "'\n";
		return;
	    }

	    granularity_os << "# density" << "\t" << "step";
	    for (const auto& algo : algorithms) {
		granularity_os  << '\t' << algo;
	    }
	    granularity_os << '\n';

	    for (unsigned int file = 0; file < filenames.size(); file++) {		

		const auto& image = filenames[file];
		const std::string& image_name = image.first;
	        bool present = image.second;		
		int granularity = stoi(image_name.substr(0, 2));
		int density = stoi(image_name.substr(2, 3));

		if (!present) {
		    continue;
		}
		if (granularity != g) {
		    continue;
		}
		
		for (int step = 0; step < StepType::ST_SIZE; step++) {
		
		    granularity_os << std::fixed << std::setprecision(5) << density << '\t'
				   << Step(static_cast<StepType>(step)) << "\t";
		    
		    for (const auto& algo: algorithms) {
			const auto& img_it = res.find(algo);
			if (img_it == res.end()) {
			    assert(false);
			    continue;
			}			
		    
			const cv::Mat1d& img_res = img_it->second;
			double val = img_res.at<double>(file, step);

			granularity_os << std::fixed << std::setprecision(8)
				       << val << '\t';		    
		    }
		    granularity_os << '\n';
		}
	    }
	    granularity_os.close();	
	}
    }
    
    void SaveGranularityResults(const std::vector<std::string>& algorithms,
				const std::string& current_output_path,
				const std::string& dataset_name,
				const std::string& complete_results_suffix,
				const cv::Mat1i& labels,
				const std::vector<std::pair<std::string, bool>>& filenames,
				const GranularityArgs& args) {
	
	std::string output_file = current_output_path;

	const std::string output_norm_file = (current_output_path + "_norm");
	const std::string output_avg_file = current_output_path + "_avg";   
	const std::string avg_norm_file = current_output_path+ "_avg_norm";
	const std::string output_cycles_file = current_output_path + "_cycles";
	const std::string output_cache_file = current_output_path + "_l1dm";
	const std::string output_l2dm = current_output_path + "_l2dm";
	const std::string output_l3m = current_output_path + "_l3m";

	const std::string output_icache_file = current_output_path + "_l1im";	
	const std::string output_branchm_file = current_output_path + "_branchm";
	const std::string stalls_output = current_output_path + "_stalls";	
	const std::string output_min_norm = current_output_path + "_min_norm";
	const std::string output_ccy = current_output_path + "_ccy";
	const std::string output_wstalls = current_output_path + "_wstalls";
	const std::string output_instr = current_output_path + "_instr";
	const std::string output_brcnt = current_output_path + "_brcnt";

	const std::string output_tmplabels = current_output_path + "_tmplabels";
	
	const std::string output_avg_steps_file = current_output_path + "_avg_steps";   

	SaveGranularity(min_res, algorithms, output_avg_file, filenames, args);
	SaveGranularitySteps(min_res, algorithms, output_avg_steps_file, filenames, args);
	
	SaveGranularity(current_res_norm, algorithms, avg_norm_file, filenames, args);
	SaveGranularity(cur_cycles_res, algorithms, output_cycles_file, filenames, args);
	SaveGranularity(cur_dcache_res, algorithms, output_cache_file, filenames, args);
	SaveGranularity(cur_l2dm_res, algorithms, output_l2dm, filenames, args);
	SaveGranularity(cur_l3m_res, algorithms, output_l3m, filenames, args);

	SaveGranularity(cur_icache_res, algorithms, output_icache_file, filenames, args);
	SaveGranularity(cur_branchm_res, algorithms, output_branchm_file, filenames, args);
	SaveGranularity(cur_stalls_res, algorithms, stalls_output, filenames, args);
	SaveGranularity(ccy_res, algorithms, output_ccy, filenames, args);
	SaveGranularity(wstalls_res, algorithms, output_wstalls, filenames, args);
	SaveGranularity(instr_res, algorithms, output_instr, filenames, args);
	SaveGranularity(brcnt_res, algorithms, output_brcnt, filenames, args);

	
	SaveGranularity(temp_labels_res, algorithms, output_tmplabels, filenames, args);
    }

    
    bool SaveMinResults(const std::vector<std::string>& algorithms, const path& output_broad_path,
			const path& output_broad_path_norm, const cv::Mat1i& labels,
			const std::vector<std::pair<std::string, bool>>& filenames) {
	
	if (!SaveBroadOutputResults(min_res, output_broad_path.string(), labels, filenames,
				    algorithms)) {
	    return false;
	}
	return SaveBroadOutputResults(min_res_norm, output_broad_path_norm.string(), labels, filenames,
				      algorithms);

    }
};

void WriteAlgorithmSamplesSteps(Labeling* algorithm, const std::string& algo_name, const unsigned file, 
				Results& results) {
    
    size_t pixel_count = algorithm->pixel_count;
    // Save time results of all the steps
    for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
	std::string step = Step(static_cast<StepType>(step_number));
	const Samplers& samplers = algorithm->samplers;
	
	// Find if the current algorithm has the current step
	if (algorithm->perf_.find(step)) {
	    double duration = algorithm->perf_.get(step);
	    results.current_res[algo_name](file, step_number) = duration;
	    results.current_res_norm[algo_name](file, step_number) = duration / pixel_count;

	    double min_duration = results.min_res[algo_name](file, step_number);

	    if (duration < min_duration) {

		results.min_res[algo_name](file, step_number) = duration;
		results.min_res_norm[algo_name](file, step_number) = duration / pixel_count;
	    }
	}

	
	// Some metrics are only available with x86
#if defined(__i386__) || defined(__x86_64__)

	StepType step_type = static_cast<StepType>(step_number);

	/*for (int i = 0; i < PAPI_EVENT_CNT; i++) {
	    std::pair<int, int> event = g_Events[i];
	    int eventcode = event.second;
	    
	    double eventcnt = samplers.Get(step_type, eventcode);
	    double& saved_cnt = results.events(eventcode, file, step_number);
	    saved_cnt = std::min(eventcnt, saved_cnt);
	    }*/

	//std::cout << "SamplersEvents::CYCLES = " << SamplingEvents::CYCLES << "\n";
	//std::cout << "PERF_COUNT_HW_INSTRUCTIONS = " << PERF_COUNT_HW_INSTRUCTIONS << "\n";
	double cycles = samplers.Get(step_type, SamplingEvents::CYCLES);
	double& saved_cycles = results.cur_cycles_res[algo_name](file, step_number);
	saved_cycles = std::min(cycles, saved_cycles);
	
	double l1d_miss = samplers.Get(step_type, SamplingEvents::CACHE_L1D);
	double& saved_l1d = results.cur_dcache_res[algo_name](file, step_number);
	saved_l1d = std::min(l1d_miss, saved_l1d);

	double l2d_miss = samplers.Get(step_type, SamplingEvents::CACHE_L2D);
	double& saved_l2d = results.cur_l2dm_res[algo_name](file, step_number);
	saved_l2d = std::min(l2d_miss, saved_l2d);
	
	double l3_miss = samplers.Get(step_type, SamplingEvents::CACHE_L3);
	double& saved_l3 = results.cur_l3m_res[algo_name](file, step_number);
	saved_l3 = std::min(l3_miss, saved_l3);
	
	double icache_miss = samplers.Get(step_type, SamplingEvents::CACHE_L1I);
	double& saved_icache = results.cur_icache_res[algo_name](file, step_number);
	saved_icache = std::min(icache_miss, saved_icache);

	double branch_mispredict = samplers.Get(step_type, SamplingEvents::BRANCH_MISSES);
	double& saved_misp = results.cur_branchm_res[algo_name](file, step_number);
	//std::cout << "[" << file << "] - " << step << " " << algo_name << ", duration = "<< branch_mispredict << ", min = " << saved_misp << "\n";
	saved_misp = std::min(branch_mispredict, saved_misp);
	
	double stalls = samplers.Get(step_type, SamplingEvents::STALLED_CYC);
	double& saved_stalls = results.cur_stalls_res[algo_name](file, step_number);
	saved_stalls = std::min(stalls, saved_stalls);

	double instr = samplers.Get(step_type, SamplingEvents::TOTAL_INSTR);
	double& saved_instr = results.instr_res[algo_name](file, step_number);
	saved_instr = std::min(instr, saved_instr);

	
	double ccy_norm = samplers.Get(step_type, SamplingEvents::CYCLE_NIIS);
	double& saved_ccy = results.ccy_res[algo_name](file, step_number);
	saved_ccy = std::min(ccy_norm, saved_ccy);
	
	double wstalls_norm = samplers.Get(step_type, SamplingEvents::WSTALLED_CYC);
	double& saved_wstalls = results.wstalls_res[algo_name](file, step_number) = wstalls_norm;
	saved_wstalls = std::min(wstalls_norm, saved_wstalls);
	
	double branch_cnt = samplers.Get(step_type, SamplingEvents::BRANCH_INSTR);
	double& saved_brcnt = results.brcnt_res[algo_name](file, step_number) = branch_cnt;
	saved_brcnt = std::min(branch_cnt, saved_brcnt);
	
	
#endif // defined(__i386__) || defined(__x86_64__)
    }
    
    results.temp_labels_res[algo_name](file, StepType::TRANSITIVE_CLOSURE) = algorithm->stats.tmp_labels;
}


void YacclabTests::InitialOperations() {

    ob_ = OutputBox(mode_cfg_.mode + " - Performing initial operations");

    // Check if all the specified algorithms exist
    CheckAlgorithmsExistence();
    ob_.Cmessage("Checked algorithm existence");

    // Check if labeling methods of the specified algorithms exist
    Labeling *first_algo = LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_existing_algorithms[0]);
    ob_.Cmessage("Got first labeling algorithm");

    first_algo->GetInput()->Create();
    ob_.Cmessage("Created input");

    CheckMethodsExistence();
    ob_.Cmessage("Checked methods existence");

    // Check datasets existence
    CheckDatasets();

    CreateDirectories();

    ob_.CloseBox();
}

void YacclabTests::CheckAlgorithmsExistence(){
	for (auto& algo_name : mode_cfg_.ccl_algorithms) {
		if (!LabelingMapSingleton::Exists(algo_name)) {
			ob_.Cwarning("Unable to find the algorithm '" + algo_name + "'");
		}
		else {
			mode_cfg_.ccl_existing_algorithms.push_back(algo_name);
		}
	}
}

void YacclabTests::CheckMethodsExistence() {

    for (const auto& algo_name : mode_cfg_.ccl_existing_algorithms) {
	const auto& algorithm = LabelingMapSingleton::GetLabeling(algo_name);
	if (mode_cfg_.perform_average || mode_cfg_.perform_density || mode_cfg_.perform_granularity || (mode_cfg_.perform_correctness && mode_cfg_.perform_check_8connectivity_std)) {
	    try {
		algorithm->Alloc();
		algorithm->PerformLabeling();
		algorithm->Dealloc();
		mode_cfg_.ccl_average_algorithms.push_back(algo_name);
	    }
	    catch (const std::runtime_error& e) {
		ob_.Cwarning(algo_name + ": " + e.what());
	    }
	}
	if (mode_cfg_.perform_average_ws
	    || mode_cfg_.perform_parallel_average_ws
	    || (mode_cfg_.perform_correctness && mode_cfg_.perform_check_8connectivity_ws)) {
	    
	    try {
		algorithm->PerformLabelingWithSteps();
		mode_cfg_.ccl_average_ws_algorithms.push_back(algo_name);
		mode_cfg_.ccl_parallel_average_ws_algorithms.push_back(algo_name);
	    }
	    catch (const std::runtime_error& e) {
		ob_.Cwarning(algo_name + ": " + e.what());
	    }
	}
	if (mode_cfg_.perform_memory || (mode_cfg_.perform_correctness && mode_cfg_.perform_check_8connectivity_mem)) {
	    try {
		std::vector<uint64_t> temp;
		algorithm->PerformLabelingMem(temp);
		mode_cfg_.ccl_mem_algorithms.push_back(algo_name);
	    }
	    catch (const std::runtime_error& e) {
		ob_.Cwarning(algo_name + ": " + e.what());
	    }
	}
	algorithm->FreeLabelingData();
    }


    if ((mode_cfg_.perform_average
	 || (mode_cfg_.perform_correctness && mode_cfg_.perform_check_8connectivity_std))
	&& mode_cfg_.ccl_average_algorithms.size() == 0) {
	
	ob_.Cwarning("There are no 'algorithms' with valid 'PerformLabeling()' method, related tests will be skipped");
	mode_cfg_.perform_average = false;
	mode_cfg_.perform_check_8connectivity_std = false;
    }

    if ((mode_cfg_.perform_average_ws
	 || (mode_cfg_.perform_correctness && mode_cfg_.perform_check_8connectivity_ws))
	&& mode_cfg_.ccl_average_ws_algorithms.size() == 0) {
	
	ob_.Cwarning("There are no 'algorithms' with valid 'PerformLabelingWithSteps()' method, related tests will be skipped");
	mode_cfg_.perform_average_ws = false;
	mode_cfg_.perform_check_8connectivity_ws = false;
    }

    if ((mode_cfg_.perform_parallel_average_ws
	 || (mode_cfg_.perform_correctness && mode_cfg_.perform_check_8connectivity_std))
	 && mode_cfg_.ccl_parallel_average_ws_algorithms.size() == 0) {
	
	ob_.Cwarning("There are no 'algorithms' with valid 'PerformLabelingWithSteps()' method, related tests will be skipped");
	mode_cfg_.perform_average = false;
	mode_cfg_.perform_check_8connectivity_std = false;	
    }
    
    if ((mode_cfg_.perform_memory || (mode_cfg_.perform_correctness && mode_cfg_.perform_check_8connectivity_mem)) && mode_cfg_.ccl_mem_algorithms.size() == 0) {
	ob_.Cwarning("There are no 'algorithms' with valid 'PerformLabelingMem()' method, related tests will be skipped");
	mode_cfg_.perform_memory = false;
	mode_cfg_.perform_check_8connectivity_mem = false;
    }

    if (mode_cfg_.perform_average && (mode_cfg_.average_tests_number < 1 || mode_cfg_.average_tests_number > 999)) {
	ob_.Cwarning("'average test' repetitions cannot be less than 1 or more than 999, skipped");
	mode_cfg_.perform_average = false;
    }

    if (mode_cfg_.perform_density && (mode_cfg_.density_tests_number < 1 || mode_cfg_.density_tests_number > 999)) {
	ob_.Cwarning("'density test' repetitions cannot be less than 1 or more than 999, skipped");
	mode_cfg_.perform_density = false;
    }

    if (mode_cfg_.perform_average_ws && (mode_cfg_.average_ws_tests_number < 1 || mode_cfg_.average_ws_tests_number > 999)) {
	ob_.Cwarning("'average with steps test' repetitions cannot be less than 1 or more than 999, skipped");
	mode_cfg_.perform_average_ws = false;
    }

    if ((mode_cfg_.perform_correctness) && mode_cfg_.check_datasets.size() == 0) {
	ob_.Cwarning("There are no datasets specified for 'correctness test', skipped");
	mode_cfg_.perform_correctness = false;
    }

    if ((mode_cfg_.perform_average) && mode_cfg_.average_datasets.size() == 0) {
	ob_.Cwarning("There are no datasets specified for 'average test', skipped");
	mode_cfg_.perform_average = false;
    }

    if ((mode_cfg_.perform_average_ws) && mode_cfg_.average_ws_datasets.size() == 0) {
	ob_.Cwarning("There are no datasets specified for 'average with steps test', skipped");
	mode_cfg_.perform_average_ws = false;
    }

    if ((mode_cfg_.perform_memory) && mode_cfg_.memory_datasets.size() == 0) {
	ob_.Cwarning("There are no datasets specified for 'memory test', skipped");
	mode_cfg_.perform_memory = false;
    }

    if (!mode_cfg_.perform_average && !mode_cfg_.perform_correctness &&
	!mode_cfg_.perform_density && !mode_cfg_.perform_memory &&
	!mode_cfg_.perform_average_ws && !mode_cfg_.perform_granularity) {
	ob_.Cwarning("There are no tests to perform");
    }
}

class CheckDatasetsExistence {

private:
	const path &input_path_;
	const std::string &input_txt_;
	OutputBox &ob_;
	std::error_code& ec_;

public:
	CheckDatasetsExistence(const path &input_path, const std::string &input_txt, OutputBox &ob, std::error_code& ec) :
		input_path_(input_path), input_txt_(input_txt), ob_(ob), ec_(ec) {}

	bool operator()(const std::vector<cv::String>& dataset, bool print_message) {
		// Check if all the datasets' files.txt exist
		bool exists_one_dataset = false;
		for (auto& x : dataset) {
			path p = input_path_ / path(x) / path(input_txt_);
			if (!exists(p, ec_)) {
				if (print_message) {
					ob_.Cwarning("There is no dataset '" + x + "' (no files.txt available), skipped");
				}
			}
			else {
				exists_one_dataset = true;
			}
		}
		return exists_one_dataset;
	}
};
void YacclabTests::CheckDatasets() {

	std::vector<std::string> ds;
	if (mode_cfg_.perform_correctness) {
		ds.insert(ds.end(), mode_cfg_.check_datasets.begin(), mode_cfg_.check_datasets.end());
	}
	if (mode_cfg_.perform_memory) {
		ds.insert(ds.end(), mode_cfg_.memory_datasets.begin(), mode_cfg_.memory_datasets.end());
	}
	if (mode_cfg_.perform_average) {
		ds.insert(ds.end(), mode_cfg_.average_datasets.begin(), mode_cfg_.average_datasets.end());
	}
	if (mode_cfg_.perform_average) {
		ds.insert(ds.end(), mode_cfg_.average_ws_datasets.begin(), mode_cfg_.average_ws_datasets.end());
	}
	std::sort(ds.begin(), ds.end());
	ds.erase(unique(ds.begin(), ds.end()), ds.end());
	CheckDatasetsExistence check_datasets_existence = CheckDatasetsExistence(glob_cfg_.input_path, glob_cfg_.input_txt, ob_, ec_); // To check single dataset

	if (mode_cfg_.perform_correctness) {
		if (!check_datasets_existence(mode_cfg_.check_datasets, false)) {
			ob_.Cwarning("There are no valid datasets for 'correctness test', skipped");
			mode_cfg_.perform_correctness = false;
		}
	}

	if (mode_cfg_.perform_average) {
		if (!check_datasets_existence(mode_cfg_.average_datasets, false)) {
			ob_.Cwarning("There are no valid datasets for 'average test', skipped");
			mode_cfg_.perform_average = false;
		}
	}

	if (mode_cfg_.perform_average_ws) {
		if (!check_datasets_existence(mode_cfg_.average_ws_datasets, false)) {
			ob_.Cwarning("There are no valid datasets for 'average with steps test', skipped");
			mode_cfg_.perform_average_ws = false;
		}
	}

	if (mode_cfg_.perform_memory) {
		if (!check_datasets_existence(mode_cfg_.memory_datasets, false)) {
			ob_.Cwarning("There are no valid datasets for 'memory test', skipped");
			mode_cfg_.perform_memory = false;
		}
	}

}

void YacclabTests::CreateDirectories() {

	if (mode_cfg_.perform_average || mode_cfg_.perform_average_ws || mode_cfg_.perform_density || mode_cfg_.perform_memory || mode_cfg_.perform_granularity) {
		// Set and create current output directory
		if (!create_directories(output_path, ec_)) {
			ob_.Cerror("Unable to create output directory '" + output_path.string() + "' - " + ec_.message());
		}

		// Create the directory for latex reports
		if (!create_directories(output_path / glob_cfg_.latex_path, ec_)) {
			ob_.Cerror("Unable to create output directory '" + (output_path / glob_cfg_.latex_path).string() + "' - " + ec_.message());
		}
	}

}

// Load a list of image names from a specified file (files_path) and store them into a vector of
// pairs (filenames). Each pairs contains the name of the file (first) and a bool (second)
// representing file state.
bool YacclabTests::LoadFileList(std::vector<std::pair<std::string, bool>>& filenames, const path& files_path) {

	// Open files_path (files.txt)
	std::ifstream is(files_path.string());
	if (!is.is_open()) {
		return false;
	}

	std::string cur_filename;
	while (getline(is, cur_filename)) {
		// To delete possible carriage return in the file name
		// (especially designed for windows file newline format)
		RemoveCharacter(cur_filename, '\r');
		filenames.push_back(make_pair(cur_filename, true));
	}

	is.close();
	return true;
}

// Check if all the files in a list of pair (filename, state) exists and set the state of every file
// opportunely. The function returns true if all the files exist, false otherwise.
bool YacclabTests::CheckFileList(const path& base_path, std::vector<std::pair<std::string, bool>>& filenames)
{
	bool ret = true;
	for (size_t i = 0; i < filenames.size(); ++i) {
		std::error_code ec;
		filenames[i].second = filesystem::exists(base_path / path(filenames[i].first), ec);
		if (!filenames[i].second) {
			ret = false;
		}
	}
	return ret;
}

void YacclabTests::AverageTest() {
	OutputBox ob(mode_cfg_.mode + " Average Test");

	std::string complete_results_suffix = "_results.txt",
	    complete_results_norm_suffix = "_results_norm.txt",
	    middle_results_suffix = "_run",
	    average_results_suffix = "_average.txt";

	// Initialize results container
	average_results_ = cv::Mat1d(static_cast<unsigned>(mode_cfg_.average_datasets.size()), static_cast<unsigned>(mode_cfg_.ccl_average_algorithms.size()), std::numeric_limits<double>::max());

	std::random_device rd;
	std::mt19937 g(rd());

	for (unsigned d = 0; d < mode_cfg_.average_datasets.size(); ++d) { // For every dataset in the average list

		std::string dataset_name(mode_cfg_.average_datasets[d]),
			output_average_results = dataset_name + average_results_suffix,
			output_graph = dataset_name + kTerminalExtension,
			output_graph_bw = dataset_name + "_bw" + kTerminalExtension;

		path dataset_path(glob_cfg_.input_path / path(dataset_name)),
		    is_path = dataset_path / path(glob_cfg_.input_txt), // files.txt path
		    current_output_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / path(glob_cfg_.average_folder) / path(dataset_name)),
		    current_latex_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / glob_cfg_.latex_path),
		    output_broad_path = current_output_path / path(dataset_name + complete_results_suffix),
		    output_broad_path_norm = current_output_path / path(dataset_name + complete_results_suffix),
		    output_colored_images_path = current_output_path / path(glob_cfg_.colors_folder),
		    output_middle_results_path = current_output_path / path(glob_cfg_.middle_folder),
		    average_os_path = current_output_path / path(output_average_results);
		
		if (!create_directories(current_output_path)) {
			ob.Cwarning("Unable to find/create the output path '" + current_output_path.string() + "', skipped", dataset_name);
			continue;
		}

		// TODO: remove color labels from this test
		if (glob_cfg_.average_color_labels) {
			if (!create_directories(output_colored_images_path)) {
				ob.Cwarning("Unable to find/create the output path '" + output_colored_images_path.string() + "', colored images won't be saved");
			}
		}

		// For AVERAGE
		std::ofstream average_os(average_os_path.string());
		if (!average_os.is_open()) {
			ob.Cwarning("Unable to open '" + average_os_path.string() + "', skipped", dataset_name);
			continue;
		}

		// To save list of filename on which CLLAlgorithms must be tested
		std::vector<std::pair<std::string, bool>> filenames;  // first: filename, second: state of filename (find or not)
		if (!LoadFileList(filenames, is_path)) {
			ob.Cwarning("Unable to open '" + is_path.string() + "', skipped", dataset_name);
			continue;
		}

		// Number of files
		unsigned filenames_size = static_cast<unsigned>(filenames.size());

		// To save middle/min and average results;
		cv::Mat1d min_res(filenames_size, static_cast<unsigned>(
				      mode_cfg_.ccl_average_algorithms.size()),
				  std::numeric_limits<double>::max());
		
		cv::Mat1d min_res_norm(filenames_size, static_cast<unsigned>(
					   mode_cfg_.ccl_average_algorithms.size()),
				       std::numeric_limits<double>::max());
		
		cv::Mat1d current_res(filenames_size, static_cast<unsigned>(
					  mode_cfg_.ccl_average_algorithms.size()),
				      std::numeric_limits<double>::max());
		
		cv::Mat1i labels(filenames_size, static_cast<unsigned>(
				     mode_cfg_.ccl_average_algorithms.size()), 0);
		
		std::vector<std::pair<double, uint16_t>> supp_average(
		    mode_cfg_.ccl_average_algorithms.size(), std::make_pair(0.0, 0));
		

		
		// Start output message box
		ob.StartRepeatedBox(dataset_name, filenames_size, mode_cfg_.average_tests_number);

		if (mode_cfg_.average_ws_save_middle_tests) {
			if (!create_directories(output_middle_results_path)) {
				ob.Cwarning("Unable to find/create the output path '" + output_middle_results_path.string() + "', middle results won't be saved");
			}
		}

		// To save middle/min and average results;
		Results results;
		results.Init(mode_cfg_.ccl_average_algorithms, filenames_size);
		
		std::map<std::string, size_t> algo_pos;
		for (size_t i = 0; i < mode_cfg_.ccl_average_algorithms.size(); ++i)
			algo_pos[mode_cfg_.ccl_average_algorithms[i]] = i;
		auto shuffled_ccl_average_algorithms = mode_cfg_.ccl_average_algorithms;

		// Test is executed n_test times
		for (unsigned test = 0; test < mode_cfg_.average_tests_number; ++test) {
			// For every file in list
			for (unsigned file = 0; file < filenames.size(); ++file) {
				// Display output message box
				ob.UpdateRepeatedBox(file);

				std::string filename = filenames[file].first;
				path filename_path = dataset_path / path(filename);

				ImageMetadata metadata;
				// Read and load image
				if (!LabelingMapSingleton::GetLabeling(
					mode_cfg_.ccl_average_algorithms[0])->GetInput()
				    ->ReadBinary(filename_path.string(), metadata)) {
				    
					ob.Cmessage("Unable to open '" + filename + "', skipped");
					continue;
				}

				shuffle(begin(shuffled_ccl_average_algorithms), end(shuffled_ccl_average_algorithms), g);

				// For all the Algorithms in the array
				for (const auto& algo_name : shuffled_ccl_average_algorithms) {
					Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
					unsigned i = static_cast<unsigned>(algo_pos[algo_name]);

					try {
						// Perform current algorithm on current image and save result.
					    algorithm->Alloc();
					    algorithm->perf_.start();
					    algorithm->PerformLabeling();
					    algorithm->perf_.stop();
					    algorithm->Dealloc();
					    algorithm->perf_.store(Step(StepType::ALL_SCANS),
								   algorithm->perf_.last());
					}
					catch (const std::exception& e) {
						algorithm->FreeLabelingData();
						ob.Cerror("Something wrong with " + algo_name + ": " + e.what()); // You should check your algorithms' implementation before performing YACCLAB tests  
					}

					WriteAlgorithmSamplesSteps(algorithm, algo_name, file, results);

					YacclabTensorOutput* img_out = algorithm->GetOutput();
					// If 'at_colorLabels' is enabled only the first time (test == 0) the output is saved
					if (glob_cfg_.average_color_labels && test == 0) {
						// Remove gnuplot escape character from output filename
					    img_out->PrepareForCheck();
					    img_out->NormalizeLabels();
					    std::string colored_image = (output_colored_images_path / path(filename + "_" + algo_name + ".png")).string();
					    img_out->WriteColored(colored_image);
					}
					algorithm->FreeLabelingData();
				} // END ALGORITHMS FOR
			} // END FILES FOR.
			ob.StopRepeatedBox(false);

			// Save middle results if necessary (flag 'average_save_middle_tests' enabled)
			if (mode_cfg_.average_save_middle_tests) {
				std::string output_middle_results_file = (output_middle_results_path / path(dataset_name + middle_results_suffix + "_" + std::to_string(test) + ".txt")).string();
				if (!SaveBroadOutputResultsMat(current_res, output_middle_results_file, labels, filenames, mode_cfg_.ccl_average_algorithms)) {
					ob.Cwarning("Unable to save middle results for 'average test'");
				}

				results.SaveBroadAverageSteps(mode_cfg_.ccl_average_algorithms,
							 output_middle_results_path,
							 dataset_name, middle_results_suffix, test,
							 labels, filenames);
			}
		} // END TEST FOR

		// To write in a file min results
		if (!SaveBroadOutputResultsMat(min_res, output_broad_path.string(), labels, filenames,
					   mode_cfg_.ccl_average_algorithms)) {
			ob.Cwarning("Unable to save min results for 'average test'");
		}

		if (!SaveBroadOutputResultsMat(min_res_norm, output_broad_path_norm.string(),
					       labels, filenames, mode_cfg_.ccl_average_algorithms)) {
			ob.Cwarning("Unable to save normalized min results for 'average test'");
		}


		// To calculate average times and write it on the specified file
		for (int r = 0; r < min_res.rows; ++r) {
			for (int c = 0; c < min_res.cols; ++c) {
				if (min_res(r, c) != std::numeric_limits<double>::max()) {
					supp_average[c].first += min_res(r, c);
					supp_average[c].second++;
				}
			}
		}

		average_os << "#Algorithm" << '\t' << "Average" << '\t' << "Round Average for Graphs" << '\n';
		for (unsigned i = 0; i < mode_cfg_.ccl_average_algorithms.size(); ++i) {
			// For all the Algorithms in the array
			const auto& algo_name = mode_cfg_.ccl_average_algorithms[i];

			// Gnuplot requires double-escaped name in presence of underscores
			{
				std::string algo_name_double_escaped = algo_name;
				std::size_t found = algo_name_double_escaped.find_first_of("_");
				while (found != std::string::npos) {
					algo_name_double_escaped.insert(found, "\\\\");
					found = algo_name_double_escaped.find_first_of("_", found + 3);
				}
				average_os << algo_name_double_escaped << '\t';
			}

			// Save all the results
			average_results_(d, i) = supp_average[i].first / supp_average[i].second;
			average_os << std::fixed << std::setprecision(8) << supp_average[i].first / supp_average[i].second << '\t';
			// TODO numberOfDecimalDigitToDisplayInGraph in configuration file ?
			average_os << std::fixed << std::setprecision(2) << supp_average[i].first / supp_average[i].second << '\n';
		}

		{ // GNUPLOT SCRIPT
			std::string compiler_name(SystemInfo::compiler_name());
			std::string compiler_version(SystemInfo::compiler_version());
			//replace the . with _ for compiler strings
			std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');

			path script_os_path = current_output_path / path(dataset_name + glob_cfg_.gnuplot_script_extension);

			std::ofstream script_os(script_os_path.string());
			if (!script_os.is_open()) {
				ob.Cwarning("Unable to create " + script_os_path.string());
			}

			script_os << "# This is a gnuplot (http://www.gnuplot.info/) script!" << '\n';
			script_os << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << dataset_name + glob_cfg_.gnuplot_script_extension << "' if you want to run it" << '\n' << '\n';

			script_os << "reset" << '\n';
			script_os << "# cd '" << current_output_path.string() << "\'" << '\n';
			script_os << "set grid ytic" << '\n';
			script_os << "set grid" << '\n' << '\n';

			script_os << "# " << dataset_name << "(COLORS)" << '\n';
			script_os << "set output '" + (current_output_path / path(output_graph)).string() + "'" << '\n';

			script_os << "set title " << LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])->GetTitle() << '\n' << '\n';

			script_os << "# " << kTerminal << " colors" << '\n';
			script_os << "set terminal " << kTerminal << " enhanced color font ',15'" << '\n' << '\n';

			script_os << "# Graph style" << '\n';
			script_os << "set style data histogram" << '\n';
			script_os << "set style histogram cluster gap 1" << '\n';
			script_os << "set style fill solid 0.25 border -1" << '\n';
			script_os << "set boxwidth 0.9" << '\n' << '\n';

			script_os << "# Get stats to set labels" << '\n';
			script_os << "stats '" << (current_output_path / path(output_average_results)).string() << "' using 3 nooutput" << '\n';
			script_os << "ymax = STATS_max + (STATS_max/100)*10" << '\n';
			script_os << "xw = 0" << '\n';
			script_os << "yw = (ymax)/22.0" << '\n' << '\n';

			script_os << "# Axes labels" << '\n';
			script_os << "set xtic rotate by -45 scale 0" << '\n';
			script_os << "set ylabel \"Execution Time [ms]\"" << '\n' << '\n';

			script_os << "# Axes range" << '\n';
			script_os << "set yrange[0:ymax]" << '\n';
			script_os << "set xrange[*:*]" << '\n' << '\n';

			script_os << "# Legend" << '\n';
			script_os << "set key off" << '\n' << '\n';

			script_os << "# Plot" << '\n';
			script_os << "plot \\" << '\n';

			script_os << "'" + (current_output_path / path(output_average_results)).string() + "' using 3:xtic(1), '" 
                << (current_output_path / path(output_average_results)).string() << "' using ($0 - xw) : ($3 + yw) : (stringcolumn(3)) with labels" << '\n' << '\n';

			script_os << "# Replot in latex folder" << '\n';
			script_os << "set title \"\"" << '\n' << '\n';
			script_os << "set output \'" << (current_latex_path / path(compiler_name + compiler_version + "_" + output_graph)).string() << "\'" << '\n';
			script_os << "replot" << '\n' << '\n';

			script_os << "# " << dataset_name << "(BLACK AND WHITE)" << '\n';
			script_os << "set output '" + (current_output_path / path(output_graph_bw)).string() + "'" << '\n';

			script_os << "set title " << LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])->GetTitle() << '\n' << '\n';

			script_os << "# " << kTerminal << " black and white" << '\n';
			script_os << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << '\n' << '\n';

			script_os << "replot" << '\n' << '\n';

			script_os << "exit gnuplot" << '\n';

			average_os.close();
			script_os.close();
		} // GNUPLOT SCRIPT

		if (0 != std::system(("gnuplot \"" + (current_output_path / path(dataset_name + glob_cfg_.gnuplot_script_extension)).string() + "\" 2> gnuplot_errors.txt").c_str())) {
			ob.Cwarning("Unable to run gnuplot script");
		}
		ob.CloseBox();
	} // END DATASET FOR
	// LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])->ReleaseInput();
}

void YacclabTests::ParallelGranularityTest() {

    OutputBox ob(mode_cfg_.mode + " Parallel Granularity Test");


    std::string complete_results_suffix = "_results.txt",
	middle_results_suffix = "_run",
	granularity_results_suffix = "_granularity";

    
    std::random_device rd;
    std::mt19937 g(rd());

    const unsigned THREAD_MAX = glob_cfg_.parallel_max_threads;
    
    for (const auto& dataset: mode_cfg_.granularity_datasets) {

	std::string dataset_name = dataset,
	    output_granularity_results = dataset_name + granularity_results_suffix,
	    output_granularity_graph = dataset_name + "_granularity" + kTerminalExtension,
	    output_granularity_bw_graph = dataset_name + "_granularity_bw" + kTerminalExtension;

	path father_dir_path;
	unsigned int dims = (LabelingMapSingleton::GetLabeling(
				 mode_cfg_.ccl_average_algorithms[0]))->GetInput()->Dims();

	if (dims == 2) {
	    father_dir_path = "random";
	} else if (dims == 3) {
	    father_dir_path = "random3D";
	}


	path dataset_path(glob_cfg_.input_path / father_dir_path / path(dataset_name)),
	    is_path = dataset_path / path(glob_cfg_.input_txt), // files.txt path
	    current_output_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / path(glob_cfg_.granularity_folder) / path(dataset_name)),
	    current_latex_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / glob_cfg_.latex_path),
	    output_broad_path = current_output_path / path(dataset_name + complete_results_suffix),
	    output_middle_results_path = current_output_path / path(glob_cfg_.middle_folder),
	    granularity_os_path = current_output_path / path(output_granularity_results);

	if (!create_directories(current_output_path)) {
	    ob.Cwarning("Unable to find/create the output path '" + current_output_path.string() + ", skipped", dataset_name);
	    continue;
	}
	
	// To save list of filename on which CLLAlgorithms must be tested
	std::vector<std::pair<std::string, bool>> filenames;  // first: filename, second: state of filename (find or not)
	if (!LoadFileList(filenames, is_path)) {
	    ob.Cwarning("Unable to open '" + is_path.string() + "', skipped", dataset_name);
	    continue;
	}
	if (!CheckFileList(dataset_path, filenames)) {
	    ob.Cwarning("Missing some files in dataset folder", dataset_name);
	}

	unsigned filenames_size = static_cast<unsigned>(filenames.size());

	constexpr uint8_t DENSITY = 101;
	uint8_t granularity;
        if (dims == 2) {
            granularity = TestsConf::kGranularities2D; // For granularity tests granularity ranges from 1 to 16 with step 1
        }
        if (dims == 3) {
	    granularity = TestsConf::kGranularities3D;
	}	

	// Start output message box
	ob.StartRepeatedBox(dataset_name, filenames_size, mode_cfg_.granularity_tests_number);
	
	if (mode_cfg_.granularity_save_middle_tests) {
	    if (!create_directories(output_middle_results_path)) {
		ob.Cwarning("Unable to find/create the output path '" + output_middle_results_path.string() + "', middle results won't be saved", dataset_name);
	    }
	}


	
	std::map<std::string, size_t> algo_pos;
	size_t algo_count = mode_cfg_.ccl_average_algorithms.size();
	for (size_t i = 0; i < algo_count; ++i)
	    algo_pos[mode_cfg_.ccl_average_algorithms[i]] = i;
	auto shuffled_ccl_average_algorithms = mode_cfg_.ccl_average_algorithms;

	shuffle(begin(filenames), end(filenames), g);



	size_t test_count = mode_cfg_.granularity_tests_number;
	
	RawGranularityResults results;
	results.Reserve(algo_count * filenames_size * test_count * THREAD_MAX);


	for (unsigned test = 0; test < test_count; ++test) {
	    for (unsigned file = 0; file < filenames.size(); file++) {
		// Display output message box
		ob.UpdateRepeatedBox(file);

		std::string filename = filenames[file].first;
		path filename_path = dataset_path / path(filename);

		ImageMetadata metadata;
		if (!LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])
		    ->GetInput()->ReadBinary(filename_path.string(), metadata)) {
		    
		    ob.Cwarning("Unable to open '" + filename + "'");
		    continue;
		}

		shuffle(begin(shuffled_ccl_average_algorithms), end(shuffled_ccl_average_algorithms), g);
		
		std::string image_name = filename_path.stem().string();
		int granularity = stoi(image_name.substr(0, 2));
		int density = stoi(image_name.substr(2, 3));		

		for (unsigned thread_count = 1; thread_count <= THREAD_MAX; thread_count++) {
		    for (size_t algo_id = 0; algo_id < shuffled_ccl_average_algorithms.size(); algo_id++) {
			const auto& algo_name = shuffled_ccl_average_algorithms[algo_id];
			Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
			unsigned i = static_cast<unsigned>(algo_pos[algo_name]);
			
			algorithm->UpdatePixelCount();
			algorithm->SetThreadCount(thread_count);
		    
			size_t img_size = metadata.depth * metadata.height * metadata.width;

			try {
			    algorithm->PerformLabelingWithSteps();
			} catch (const std::exception& e) {
			    algorithm->FreeLabelingData();
			    ob.Cerror("Something wrong with " + algo_name + ": " + e.what()); // You should check your algorithms' implementation before performing YACCLAB tests
			}

			long label_count = algorithm->n_labels_;

			double duration = algorithm->perf_.get(Step(StepType::ALL_SCANS));
			results.Write(file, algo_id, thread_count, density, granularity,
				      metadata.width, metadata.height, metadata.depth, duration,
				      label_count);

		    
		    } // END ALGORITHMS FOR
		} // END THREADS FOR		
	    } // END FILES FOR
	    ob.StopRepeatedBox(false);
	}
    }
}

void YacclabTests::ParallelAverageTestWithSteps() {

    OutputBox ob(mode_cfg_.mode + " Parallel Average Test With Steps");

    std::string complete_results_suffix = "_results.txt",
	complete_results_norm_suffix = "_results_norm.txt",
	average_results_suffix = "_results_avg.txt",
	middle_results_suffix = "_run",
	average_results_rounded_suffix = "_average_rounded.txt",
	thread_results_suffix = "_threads.txt";

    std::random_device rd;
    std::mt19937 g(rd());

    const unsigned THREAD_MAX = glob_cfg_.parallel_max_threads;

    ThreadResults thread_results;
    
    unsigned dataset_size = mode_cfg_.parallel_average_ws_datasets.size();
    for (unsigned d = 0; d < dataset_size; d++) {
	std::string dataset_name(mode_cfg_.parallel_average_ws_datasets[d]),
	    output_average_results_rounded = dataset_name + average_results_rounded_suffix,
	    output_average_results = dataset_name + average_results_suffix,
	    output_thread_results = dataset_name + thread_results_suffix;

	path dataset_path(glob_cfg_.input_path / path(dataset_name)),
	    is_path = dataset_path / path(glob_cfg_.input_txt), // files.txt path
	    current_output_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path
				/ path(glob_cfg_.parallel_average_ws_folder) / path(dataset_name)),
	    
	    current_latex_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / glob_cfg_.latex_path),
	    output_broad_path = current_output_path / path(dataset_name + complete_results_suffix),
	    output_broad_path_norm = current_output_path / path(dataset_name + complete_results_norm_suffix),
	    output_middle_results_path = current_output_path / path(glob_cfg_.middle_folder),
	    average_os_path = current_output_path / path(output_average_results),
	    average_rounded_os_path = current_output_path / path(output_average_results_rounded),
	    output_thread_results_path = current_output_path / path(output_thread_results);

	
	if (!create_directories(current_output_path)) {
	    ob.Cwarning("Unable to find/create the output path '" + current_output_path.string() + "', skipped", dataset_name);
	    continue;
	}

	// Initialize results container
	average_ws_results_[dataset_name] = cv::Mat1d(
	    static_cast<unsigned>(mode_cfg_.ccl_average_ws_algorithms.size()),
	    StepType::ST_SIZE, std::numeric_limits<double>::max());

	// To save list of filename on which CLLAlgorithms must be tested
	std::vector<std::pair<std::string, bool>> filenames;  // first: filename, second: state of filename (find or not)
	if (!LoadFileList(filenames, is_path)) {
	    ob.Cwarning("Unable to open '" + is_path.string() + "'", dataset_name);
	    continue;
	}


	// Number of files
	unsigned int filenames_size = static_cast<unsigned>(filenames.size());

	// To save middle/min and average results;
	Results results;
		
	cv::Mat1i labels(filenames_size, static_cast<unsigned>(mode_cfg_.ccl_parallel_average_ws_algorithms.size()), 0);

	
	// Start output message box
	ob.StartRepeatedBox(dataset_name, filenames_size, mode_cfg_.average_ws_tests_number);


	if (mode_cfg_.parallel_average_ws_save_middle_tests) {
	    if (!create_directories(output_middle_results_path)) {
		ob.Cwarning("Unable to find/create the output path '" + output_middle_results_path.string() + "', middle results won't be saved");
	    }
	}	
	
	const auto& algorithms = mode_cfg_.ccl_parallel_average_ws_algorithms;


	std::vector<std::string> all_algorithms;
	all_algorithms.reserve(algorithms.size() * THREAD_MAX);
	for (const auto& algo: algorithms) {
	    for (unsigned thread_count = 1; thread_count <= THREAD_MAX; thread_count++) {
		all_algorithms.push_back(algo + "_t" + std::to_string(thread_count));
	    }
	}
	results.Init(all_algorithms, filenames_size);

	
	std::map<std::string, size_t> algo_pos;
	for (size_t i = 0; i < algorithms.size(); ++i)
	    algo_pos[algorithms[i]] = i;
	auto shuffled_algorithms = mode_cfg_.ccl_parallel_average_ws_algorithms;

	std::vector<double> thread_durations;
	thread_durations.resize(THREAD_MAX);


	
	const unsigned test_count = mode_cfg_.parallel_average_ws_tests_number;

	thread_results.Reserve(all_algorithms.size() * test_count * filenames.size());

	
	for (unsigned test = 0; test < test_count; test++) {
	    for (unsigned file = 0; file < filenames.size(); file++) {

		ob.UpdateRepeatedBox(file);

		const std::string filename = filenames[file].first;
		path filename_path = dataset_path / path(filename);

		ImageMetadata metadata;
		if (!LabelingMapSingleton::GetLabeling(algorithms[0])->GetInput()->ReadBinary(filename_path.string(), metadata)) {
		    ob.Cwarning("Unable to open '" + filename + "'");
		    continue;
		}

		shuffle(begin(shuffled_algorithms), end(shuffled_algorithms), g);

		for (unsigned thread_count = 1; thread_count <= THREAD_MAX; thread_count++) {

		    //omp_set_num_threads(thread_count);
		    

		    
		    for (unsigned algo_num = 0; algo_num < shuffled_algorithms.size(); algo_num++) {
			const auto& algo_name = shuffled_algorithms[algo_num];
			Labeling* algorithm = LabelingMapSingleton::GetLabeling(algo_name);

			algorithm->UpdatePixelCount();
			algorithm->SetThreadCount(thread_count);
			
			const auto algo_name_thread = algo_name + "_t" + std::to_string(thread_count);
			
			try {
			    algorithm->PerformLabelingWithSteps();
			    
			} catch (const std::exception& e) {
			    algorithm->FreeLabelingData();
			    // You should check your algorithms' implementation before performing YACCLAB tests  
			    ob.Cerror("Something wrong with " + algo_name + ": " + e.what()); 

			}

			algorithm->GetThreadDurations(thread_durations);

			WriteAlgorithmSamplesSteps(algorithm, algo_name_thread, file, results);
			thread_results.Write(file, algo_num, thread_count, thread_durations);
			
			algorithm->FreeLabelingData();
		    } // END ALGORITHMS FOR
		} // END THREADS FOR
	    } // END FILES FOR
	    ob.StopRepeatedBox(false);

		
	    if (mode_cfg_.parallel_average_ws_save_middle_tests) {
		results.SaveBroadAverageSteps(all_algorithms,
					      output_middle_results_path, dataset_name,
					      middle_results_suffix, test, labels, filenames);
	    }	    
	} // END TESTS FOR

	results.SaveMinResults(all_algorithms, output_broad_path,
			       output_broad_path_norm, labels, filenames);
	
	thread_results.Serialize(output_thread_results_path.string(), filenames, shuffled_algorithms);

	
	// Write the results stored in average_ws_results_ in file
	SaveAverageWithStepsResults(average_os_path.string(), dataset_name, false);
	SaveAverageWithStepsResults(average_rounded_os_path.string(), dataset_name, true);


    }	    
}

void YacclabTests::AverageTestWithSteps() {
	// Initialize output message box
	OutputBox ob(mode_cfg_.mode + " Average Test With Steps");

	std::string complete_results_suffix = "_results.txt",
	    complete_results_norm_suffix = "_results_norm.txt",
	    middle_results_suffix = "_run",
	    average_results_suffix = "_average.txt",
	    average_results_rounded_suffix = "_average_rounded.txt";

	std::random_device rd;
	std::mt19937 g(rd());

	tie_to_cpu();
	
	for (unsigned d = 0; d < mode_cfg_.average_ws_datasets.size(); ++d) { // For every dataset in the average list

	    std::string dataset_name(mode_cfg_.average_ws_datasets[d]),
		output_average_results = dataset_name + average_results_suffix,
		output_average_results_rounded = dataset_name + average_results_rounded_suffix,
		output_graph = dataset_name + kTerminalExtension,
		output_graph_bw = dataset_name + "_bw" + kTerminalExtension;
		
	    path dataset_path(glob_cfg_.input_path / path(dataset_name)),
		is_path = dataset_path / path(glob_cfg_.input_txt), // files.txt path
		current_output_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / path(glob_cfg_.average_ws_folder) / path(dataset_name)),
		current_latex_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / glob_cfg_.latex_path),
		output_broad_path = current_output_path / path(dataset_name + complete_results_suffix),
		output_broad_path_norm = current_output_path / path(dataset_name + complete_results_norm_suffix),
		output_middle_results_path = current_output_path / path(glob_cfg_.middle_folder),
		average_os_path = current_output_path / path(output_average_results),
		average_rounded_os_path = current_output_path / path(output_average_results_rounded);
		
	    if (!create_directories(current_output_path)) {
		ob.Cwarning("Unable to find/create the output path '" + current_output_path.string() + "', skipped", dataset_name);
		continue;
	    }

	    // Initialize results container
	    average_ws_results_[dataset_name] = cv::Mat1d(static_cast<unsigned>(mode_cfg_.ccl_average_ws_algorithms.size()), StepType::ST_SIZE, std::numeric_limits<double>::max());

	    // To save list of filename on which CLLAlgorithms must be tested
	    std::vector<std::pair<std::string, bool>> filenames;  // first: filename, second: state of filename (find or not)
	    if (!LoadFileList(filenames, is_path)) {
		ob.Cwarning("Unable to open '" + is_path.string() + "'", dataset_name);
		continue;
	    }

	    // Number of files
	    unsigned int filenames_size = static_cast<unsigned>(filenames.size());

	    // To save middle/min and average results;
	    Results results;
		
	    cv::Mat1i labels(filenames_size, static_cast<unsigned>(mode_cfg_.ccl_average_ws_algorithms.size()), 0);

	    results.Init(mode_cfg_.ccl_average_ws_algorithms, filenames_size);
		
	    // Start output message box
	    ob.StartRepeatedBox(dataset_name, filenames_size, mode_cfg_.average_ws_tests_number);

	    if (mode_cfg_.average_ws_save_middle_tests) {
		if (!create_directories(output_middle_results_path)) {
		    ob.Cwarning("Unable to find/create the output path '" + output_middle_results_path.string() + "', middle results won't be saved");
		}
	    }

	    std::map<std::string, size_t> algo_pos;
	    for (size_t i = 0; i < mode_cfg_.ccl_average_ws_algorithms.size(); ++i)
		algo_pos[mode_cfg_.ccl_average_ws_algorithms[i]] = i;
	    auto shuffled_ccl_average_ws_algorithms = mode_cfg_.ccl_average_ws_algorithms;

	    // Test is executed n_test times
	    for (unsigned test = 0; test < mode_cfg_.average_ws_tests_number; ++test) {
		// For every file in list
		for (unsigned file = 0; file < filenames.size(); ++file) {
		    // Display output message box
		    ob.UpdateRepeatedBox(file);

		    std::string filename = filenames[file].first;
		    path filename_path = dataset_path / path(filename);

		    // Read and load image
		    //if (!GetBinaryImage(filename_path, Labeling::img_)) {
		    //	ob.Cwarning("Unable to open '" + filename + "'");
		    //	continue;
		    //}
		    ImageMetadata metadata;
		    if (!LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_ws_algorithms[0])
			->GetInput()->ReadBinary(filename_path.string(), metadata)) {
			
			ob.Cwarning("Unable to open '" + filename + "'");
			continue;
		    }

		    shuffle(begin(shuffled_ccl_average_ws_algorithms), end(shuffled_ccl_average_ws_algorithms), g);

		    // For all the Algorithms in the array
		    for (const auto& algo_name : shuffled_ccl_average_ws_algorithms) {
			Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
			// unsigned i = static_cast<unsigned>(algo_pos[algo_name]);
			algorithm->UpdatePixelCount();

			try {
			    // Perform current algorithm on current image and save result.
			    algorithm->PerformLabelingWithSteps();
			}
			catch (const std::exception& e) {
			    algorithm->FreeLabelingData();
			    ob.Cerror("Something wrong with " + algo_name + ": " + e.what()); // You should check your algorithms' implementation before performing YACCLAB tests  
			}

			WriteAlgorithmSamplesSteps(algorithm, algo_name, file, results);

			algorithm->FreeLabelingData();
		    } // END ALGORITHMS FOR
		} // END FILES FOR.
		ob.StopRepeatedBox(false);

		// Save middle results if necessary (flag 'average_save_middle_tests' enabled)
		if (mode_cfg_.average_ws_save_middle_tests) {

		    results.SaveBroadAverageSteps(mode_cfg_.ccl_average_ws_algorithms,
					     output_middle_results_path,
					     dataset_name, middle_results_suffix, test,
					     labels, filenames);
		}
	    }// END TESTS FOR

	    // To write in a file min results
	    results.SaveMinResults(mode_cfg_.ccl_average_ws_algorithms, output_broad_path,
				   output_broad_path_norm, labels, filenames);

	    // If true the i-th step is used by all the algorithms
	    std::vector<bool> steps_presence(StepType::ST_SIZE, false);

	    // To calculate average times and write it on the specified file
	    for (unsigned a = 0; a < mode_cfg_.ccl_average_ws_algorithms.size(); ++a) {
		const auto& algo_name(mode_cfg_.ccl_average_ws_algorithms[a]);
		std::vector<std::pair<double, uint16_t>> supp_average(StepType::ST_SIZE, std::make_pair(0.0, 0));

		for (int r = 0; r < results.min_res.at(algo_name).rows; ++r) {
		    for (int c = 0; c < results.min_res.at(algo_name).cols; ++c) {
			if (results.min_res.at(algo_name)(r, c) != std::numeric_limits<double>::max()) {
			    supp_average[c].first += results.min_res.at(algo_name)(r, c);
			    supp_average[c].second++;
			}
		    }
		}

		// Matrix reduce done, save the results into the average file
		for (int step_number = 0; step_number != StepType::ST_SIZE; ++step_number) {
		    double avg{ 0.0 };
		    if (supp_average[step_number].first > 0.0 && supp_average[step_number].second > 0) {
			steps_presence[step_number] = true;
			avg = supp_average[step_number].first / supp_average[step_number].second;
		    }
		    else {
			// The current step is not threated by the current algorithm, write 0
		    }
		    average_ws_results_[dataset_name](a, step_number) = avg;
		}
	    }

	    // Write the results stored in average_ws_results_ in file
	    SaveAverageWithStepsResults(average_os_path.string(), dataset_name, false);
	    SaveAverageWithStepsResults(average_rounded_os_path.string(), dataset_name, true);

	    // GNUPLOT SCRIPT
	    {
		std::string compiler_name(SystemInfo::compiler_name());
		std::string compiler_version(SystemInfo::compiler_version());
		//replace the . with _ for compiler strings
		std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');

		path script_os_path = current_output_path / path(dataset_name + glob_cfg_.gnuplot_script_extension);

		std::ofstream script_os(script_os_path.string());
		if (!script_os.is_open()) {
		    ob.Cwarning("Unable to create " + script_os_path.string());
		}

		script_os << "# This is a gnuplot (http://www.gnuplot.info/) script!" << '\n';
		script_os << "# comment fifth line, open gnuplot's teminal, move to script's path and launch 'load " << dataset_name + glob_cfg_.gnuplot_script_extension << "' if you want to run it" << '\n' << '\n';

		script_os << "reset" << '\n';
		script_os << "# cd '" << current_output_path.string() << "\'" << '\n';
		script_os << "set grid ytic" << '\n';
		script_os << "set grid" << '\n' << '\n';

		script_os << "# " << dataset_name << "(COLORS)" << '\n';
		script_os << "set output '" + (current_output_path / path(output_graph)).string() + "'" << '\n';

		script_os << "set title " << LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_ws_algorithms[0])->GetTitle() << '\n' << '\n';

		script_os << "# " << kTerminal << " colors" << '\n';
		script_os << "set terminal " << kTerminal << " enhanced color font ',15'" << '\n' << '\n';

		script_os << "# Graph style" << '\n';
		script_os << "set style data histogram" << '\n';
		script_os << "set style histogram cluster gap 1" << '\n';
		script_os << "set style histogram rowstacked" << '\n';
		script_os << "set style fill solid 0.25 border -1" << '\n';
		script_os << "set boxwidth 0.6" << '\n' << '\n';

		script_os << "# Get stats to set labels" << '\n';
		script_os << "stats '" << (current_output_path / path(output_average_results_rounded)).string() << "' using " << StepType::ST_SIZE + 2 << " nooutput" << '\n';
		script_os << "ymax = STATS_max + (STATS_max/100)*10" << '\n';
		script_os << "xw = 0" << '\n';
		script_os << "yw = (ymax)/22.0" << '\n' << '\n';

		script_os << "# Axes labels" << '\n';
		script_os << "set xtic rotate by -45 scale 0" << '\n';
		script_os << "set ylabel \"Execution Time [ms]\"" << '\n' << '\n';

		script_os << "# Axes range" << '\n';
		script_os << "set yrange[0:ymax]" << '\n';
		script_os << "set xrange[*:*]" << '\n' << '\n';

		script_os << "# Legend" << '\n';
		script_os << "set key inside right font ', 8'" << '\n' << '\n';

		script_os << "# Plot" << '\n';
		script_os << "plot \\" << '\n';

		script_os << "'" + (current_output_path / path(output_average_results_rounded)).string() + "' using 2:xtic(1) title '" << 
		    Step(static_cast<StepType>(0)) << "', \\" << '\n';
		unsigned i = 3;
		// Start from the second step
		for (int step_number = 1; step_number != StepType::ST_SIZE; ++step_number, ++i) {
		    StepType step = static_cast<StepType>(step_number);
		    // Add the step only if it used by almost an algorithm
		    if (steps_presence[step_number]) {
			script_os << "'' u " << i << " title '" << Step(step) << "', \\" << '\n';
		    }
		}
		const unsigned start_n = 2;
		i = 2;
		for (; i <= StepType::ST_SIZE + 1; ++i) {
		    script_os << "'' u ($0) : ((";
		    for (unsigned j = i; j >= start_n; --j) {
			script_os << "$" << j;
			if (j > start_n) {
			    script_os << "+";
			}
		    }
		    script_os << ") - ($" << i << "/2)) : ($" << i << "!=0.0 ? sprintf(\"%4.2f\",$" << i << "):'') w labels font 'Tahoma, 8' title '', \\" << '\n';
		}
		script_os << "'' u ($0) : ($" << i << " + yw) : ($" << i << "!=0.0 ? sprintf(\"%4.2f\",$" << i << "):'') w labels font 'Tahoma' title '', \\" << '\n';

		script_os << "# Replot in latex folder" << '\n';
		script_os << "set title \"\"" << '\n' << '\n';
		script_os << "set output \'" << (current_latex_path / path(compiler_name + compiler_version + "_with_steps_" + output_graph)).string() << "\'" << '\n';
		script_os << "replot" << '\n' << '\n';

		script_os << "# " << dataset_name << "(BLACK AND WHITE)" << '\n';
		script_os << "set output '" + (current_output_path / path(output_graph_bw)).string() + "'" << '\n';

		script_os << "set title " << LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_ws_algorithms[0])->GetTitle() << '\n' << '\n';

		script_os << "# " << kTerminal << " black and white" << '\n';
		script_os << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << '\n' << '\n';

		script_os << "replot" << '\n' << '\n';

		script_os << "exit gnuplot" << '\n';

		script_os.close();
	    } // End GNUPLOT SCRIPT

	    //if (0 != std::system(("gnuplot \"" + (current_output_path / path(dataset_name + glob_cfg_.gnuplot_script_extension)).string() + "\" 2> gnuplot_errors.txt").c_str())) {
	    //	ob.Cwarning("Unable to run gnuplot script");
	    //}
	    ob.CloseBox();
	} // END DATASET FOR
	// LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_ws_algorithms[0])->ReleaseInput();
}

void YacclabTests::DensityTest() {
	OutputBox ob(mode_cfg_.mode + " Density Test");

	std::string complete_results_suffix = "_results.txt",
	    complete_results_norm_suffix = "_results_norm.txt",
	    middle_results_suffix = "_run",
	    density_results_suffix = "_density.txt",
	    normalized_density_results_suffix = "_normalized_density.txt",
	    size_results_suffix = "_size.txt",
	    null_results_suffix = "_null_results.txt";

	// Initialize results container
	density_results_ = cv::Mat1d(static_cast<unsigned>(mode_cfg_.density_datasets.size()), static_cast<unsigned>(mode_cfg_.ccl_average_algorithms.size()), std::numeric_limits<double>::max());

	std::random_device rd;
	std::mt19937 g(rd());

	for (unsigned d = 0; d < mode_cfg_.density_datasets.size(); ++d) { // For every dataset in the density list
		std::string dataset_name(mode_cfg_.density_datasets[d]),
			output_density_results = dataset_name + density_results_suffix,
			output_size_results = dataset_name + size_results_suffix,
			output_density_graph = dataset_name + "_density" + kTerminalExtension,
			output_density_bw_graph = dataset_name + "_density_bw" + kTerminalExtension,
			output_size_graph = dataset_name + "_size" + kTerminalExtension,
			output_size_bw_graph = dataset_name + "_size_bw" + kTerminalExtension,
			output_null = dataset_name + null_results_suffix;

		path father_dir_path;
		unsigned int dims = LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])->GetInput()->Dims();
		if (dims == 2) {
			father_dir_path = "random";
		}
		else if (dims == 3) {
			father_dir_path = "random3D";
		}

		path dataset_path(glob_cfg_.input_path / father_dir_path / path(dataset_name)),
		    is_path = dataset_path / path(glob_cfg_.input_txt), // files.txt path
		    current_output_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / path(glob_cfg_.density_folder) / path(dataset_name)),
		    current_latex_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / glob_cfg_.latex_path),
		    output_broad_path = current_output_path / path(dataset_name + complete_results_suffix),
		    output_broad_path_norm = current_output_path / path(dataset_name + complete_results_norm_suffix),
		    output_colored_images_path = current_output_path / path(glob_cfg_.colors_folder),
		    output_middle_results_path = current_output_path / path(glob_cfg_.middle_folder),
		    density_os_path = current_output_path / path(output_density_results),
		    size_os_path = current_output_path / path(output_size_results),
		    null_os_path = current_output_path / path(output_null);
		
		if (!create_directories(current_output_path)) {
			ob.Cwarning("Unable to find/create the output path '" + current_output_path.string() + ", skipped", dataset_name);
			continue;
		}

		// TODO: remove color labels from this test
		if (glob_cfg_.density_color_labels) {
			if (!create_directories(output_colored_images_path)) {
				ob.Cwarning("Unable to find/create the output path '" + output_colored_images_path.string() + "', colored images won't be saved");
			}
		}

		// To save list of filename on which CLLAlgorithms must be tested
		std::vector<std::pair<std::string, bool>> filenames;  // first: filename, second: state of filename (find or not)
		if (!LoadFileList(filenames, is_path)) {
			ob.Cwarning("Unable to open '" + is_path.string() + "'", dataset_name);
			continue;
		}
		if (!CheckFileList(dataset_path, filenames)) {
			ob.Cwarning("Missing some files in dataset folder, skipped", dataset_name);
			continue;
		}

		// Number of files
		unsigned int filenames_size = static_cast<unsigned>(filenames.size());

		// To save middle/min and average results;
		cv::Mat1d min_res(filenames_size, static_cast<unsigned>(
				      mode_cfg_.ccl_average_algorithms.size()),
				  std::numeric_limits<double>::max());
		
		cv::Mat1d current_res(filenames_size, static_cast<unsigned>(
					  mode_cfg_.ccl_average_algorithms.size()),
				      std::numeric_limits<double>::max());
		
		cv::Mat1i labels(filenames_size, static_cast<unsigned>(
				     mode_cfg_.ccl_average_algorithms.size()), 0);
		
		cv::Mat1d min_res_norm(filenames_size, static_cast<unsigned>(
					   mode_cfg_.ccl_average_algorithms.size()),
					   std::numeric_limits<double>::max());
		/*
		Note that number of random_images is less than 800, this is why the second element of the
		pair has uint16_t data type. Extern std::vector represent the algorithms, inner std::vector represent
		density for "supp_density" variable and dimension for "supp_dimension" one. In particular:

		FOR "supp_density" VARIABLE:
		INNER_std::vector[0] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
		INNER_VECTPR[1] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.2_DENSITY, COUNT_OF_THAT_IMAGES }
		.. and so on;

		SO:
		  supp_density[0][0] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
		  for algorithm in position 0;

		  supp_density[0][1] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.2_DENSITY, COUNT_OF_THAT_IMAGES }
		  for algorithm in position 0;

		  supp_density[1][0] represent the pair { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_0.1_DENSITY, COUNT_OF_THAT_IMAGES }
		  for algorithm in position 1;
		.. and so on

		FOR "SUP_DIMENSION VARIABLE":
		INNER_std::vector[0] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_32*32_DIMENSION, COUNT_OF_THAT_IMAGES }
		INNER_std::vector[1] = { SUM_OF_TIME_FOR_CCL_OF_IMAGES_WITH_64*64_DIMENSION, COUNT_OF_THAT_IMAGES }

		view "supp_density" explanation for more details;
	   */

		uint8_t density = 9 /*[0.1,0.9]*/, size = 8 /*[32,64,128,256,512,1024,2048,4096]*/;													//////////////////////////////////////////////////
		if (dims == 3) {
			size = 6;
		}

		using vvp = std::vector<std::vector<std::pair<double, uint16_t>>>;
		vvp supp_density(mode_cfg_.ccl_average_algorithms.size(),
				 std::vector<std::pair<double, uint16_t>>(density, std::make_pair(0, 0)));
		
		vvp supp_size(mode_cfg_.ccl_average_algorithms.size(),
			      std::vector<std::pair<double, uint16_t>>(size, std::make_pair(0, 0)));

		// Start output message box
		ob.StartRepeatedBox(dataset_name, filenames_size, mode_cfg_.density_tests_number);

		if (mode_cfg_.density_save_middle_tests) {
			if (!create_directories(output_middle_results_path)) {
				ob.Cwarning("Unable to find/create the output path '" + output_middle_results_path.string() + "', middle results won't be saved");
			}
		}

		std::map<std::string, size_t> algo_pos;
		for (size_t i = 0; i < mode_cfg_.ccl_average_algorithms.size(); ++i)
			algo_pos[mode_cfg_.ccl_average_algorithms[i]] = i;
		auto shuffled_ccl_average_algorithms = mode_cfg_.ccl_average_algorithms;

		// Test is executed n_test times
		for (unsigned test = 0; test < mode_cfg_.density_tests_number; ++test) {
			// For every file in list
			for (unsigned file = 0; file < filenames.size(); ++file) {
				// Display output message box
				ob.UpdateRepeatedBox(file);

				std::string filename = filenames[file].first;
				path filename_path = dataset_path / path(filename);

				// Read and load image
				//if (!GetBinaryImage(filename_path, Labeling::img_)) {
				//	ob.Cwarning("Unable to open '" + filename + "'");
				//	continue;
				//}
				ImageMetadata metadata;
				if (!LabelingMapSingleton::GetLabeling(
					mode_cfg_.ccl_average_algorithms[0])->GetInput()
				    ->ReadBinary(filename_path.string(), metadata)) {
				    
					ob.Cwarning("Unable to open '" + filename + "'");
					continue;
				}

				shuffle(begin(shuffled_ccl_average_algorithms), end(shuffled_ccl_average_algorithms), g);

				for (const auto& algo_name : shuffled_ccl_average_algorithms) {
					Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
					unsigned i = static_cast<unsigned>(algo_pos[algo_name]);

					// Ensure that pixel count is set
					algorithm->UpdatePixelCount();
					
					try {
						// Perform current algorithm on current image and save result.
					    algorithm->Alloc();
					    algorithm->perf_.start();
					    algorithm->PerformLabeling();
					    algorithm->perf_.stop();
					    algorithm->Dealloc();
					}
					catch (const std::exception& e) {
						algorithm->FreeLabelingData();
						ob.Cerror("Something wrong with " + algo_name + ": " + e.what()); // You should check your algorithms' implementation before performing YACCLAB tests  
					}

					
					// Save time results
					current_res(file, i) = algorithm->perf_.last();
					if (algorithm->perf_.last() < min_res(file, i)) {
						min_res(file, i) = algorithm->perf_.last();
						min_res_norm(file, i) = algorithm->perf_.last()
						    / algorithm->pixel_count;
					}

					// If 'at_colorLabels' is enabled only the first time (test == 0) the output is saved
					if (glob_cfg_.density_color_labels && test == 0) {
					    // Remove gnuplot escape character from output filename
					    YacclabTensorOutput *img_out = algorithm->GetOutput();
					    img_out->PrepareForCheck();
					    img_out->NormalizeLabels();
					    std::string colored_image = (output_colored_images_path / path(filename + "_" + algo_name + ".png")).string();
					    img_out->WriteColored(colored_image);
					}
					algorithm->FreeLabelingData();
				} // END ALGORITHMS FOR
			} // END FILES FOR.
			ob.StopRepeatedBox(false);

			// Save middle results if necessary (flag 'density_save_middle_tests' enabled)
			if (mode_cfg_.density_save_middle_tests) {
				std::string output_middle_results_file =
				    (output_middle_results_path / path(dataset_name + middle_results_suffix + "_" + std::to_string(test) + ".txt")).string();
				SaveBroadOutputResultsMat(current_res, output_middle_results_file,
							  labels, filenames, mode_cfg_.ccl_average_algorithms);
			}
		} // END TEST FOR

		// To write in a file min results
		SaveBroadOutputResultsMat(min_res, output_broad_path.string(), labels, filenames,
					  mode_cfg_.ccl_average_algorithms);
		
		SaveBroadOutputResultsMat(min_res_norm, output_broad_path_norm.string(), labels,
					  filenames, mode_cfg_.ccl_average_algorithms);
		
		// To sum min results, in the correct manner, before make average
		for (unsigned files = 0; files < filenames.size(); ++files) {
			// Note that files correspond to min_res rows
			for (int c = 0; c < min_res.cols; ++c) {
				// Add current time to "supp_density" and "supp_size" in the correct position
				if (isdigit(filenames[files].first[0]) && isdigit(filenames[files].first[1]) && isdigit(filenames[files].first[2]) && filenames[files].second) {
					// superfluous test?
					if (min_res(files, c) != std::numeric_limits<double>::max()) {
						// For density graph
						supp_density[c][ctoi(filenames[files].first[1])].first += min_res(files, c);
						supp_density[c][ctoi(filenames[files].first[1])].second++;

						// For dimension graph
						supp_size[c][ctoi(filenames[files].first[0])].first += min_res(files, c);
						supp_size[c][ctoi(filenames[files].first[0])].second++;
					}
				}
				// Add current time to "supp_density" and "supp_size" in the correct position
			}
		}

		// To calculate average times
		std::vector<std::vector<long double>> density_average(mode_cfg_.ccl_average_algorithms.size(), std::vector<long double>(density));
		std::vector<std::vector<long double>> size_average(mode_cfg_.ccl_average_algorithms.size(), std::vector<long double>(size));

		for (unsigned i = 0; i < mode_cfg_.ccl_average_algorithms.size(); ++i) {
			// For all algorithms
			for (unsigned j = 0; j < density_average[i].size(); ++j) {
				// For all density and normalized density
				if (supp_density[i][j].second != 0) {
					density_average[i][j] = supp_density[i][j].first / supp_density[i][j].second;
				}
				else {
					// If there is no element with this density characteristic the average value is set to zero
					density_average[i][j] = 0.0;
				}
			}
			for (unsigned j = 0; j < size_average[i].size(); ++j) {
				// For all size
				if (supp_size[i][j].second != 0)
					size_average[i][j] = supp_size[i][j].first / supp_size[i][j].second;
				else
					size_average[i][j] = 0.0;  // If there is no element with this size characteristic the average value is set to zero
			}
		}

		// For DENSITY RESULT
		std::ofstream density_os(density_os_path.string());
		if (!density_os.is_open()) {
			ob.Cwarning("Unable to open '" + density_os_path.string() + "'");
		}

		// For SIZE RESULT
		std::ofstream size_os(size_os_path.string());
		if (!size_os.is_open()) {
			ob.Cwarning("Unable to open '" + size_os_path.string() + "'");
		}

		density_os << "# density";
		size_os << "# size";
		for (const auto& algo_name : mode_cfg_.ccl_average_algorithms) {
			density_os << '\t' << algo_name;
			size_os << '\t' << algo_name;
		}
		density_os << '\n';
		size_os << '\n';

		// To write density result on specified file
		for (unsigned i = 0; i < density; ++i) {
			// For every density
			if (density_average[0][i] == 0.0) { // Check it only for the first algorithm (it is the same for the others)
				density_os << "#"; // It means that there is no element with this density characteristic
			}
			density_os << ((float)(i + 1) / 10) << '\t'; //Density value
			for (unsigned j = 0; j < density_average.size(); ++j) {
				// For every algorithm
				density_os << density_average[j][i] << '\t';
			}
			density_os << '\n'; // End of current line (current density)
		}
		// To set sizes's label
		std::vector<std::pair<unsigned, double>> supp_size_labels(size, std::make_pair(0, 0));

		// To write size result on specified file
		for (unsigned i = 0; i < size; ++i) {
			// For every size
			if (size_average[0][i] == 0.0) // Check it only for the first algorithm (it is the same for the others)
				size_os << "#"; // It means that there is no element with this size characteristic
			supp_size_labels[i].first = (int)(pow(2, i + (9 - 2 * dims)));		// dim = 2 -> 5 --- dim = 3 -> 3
			supp_size_labels[i].second = size_average[0][i];
			size_os << (int)pow(supp_size_labels[i].first, dims) << '\t'; //Size value
			for (unsigned j = 0; j < size_average.size(); ++j) {
				// For every algorithms
				size_os << size_average[j][i] << '\t';
			}
			size_os << '\n'; // End of current line (current size)
		}

		// GNUPLOT SCRIPT
		{
			std::string compiler_name(SystemInfo::compiler_name());
			std::string compiler_version(SystemInfo::compiler_version());
			//replace the . with _ for compiler std::strings
			std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');

			path script_os_path = current_output_path / path(dataset_name + glob_cfg_.gnuplot_script_extension);

			std::ofstream script_os(script_os_path.string());
			if (!script_os.is_open()) {
				ob.Cwarning("Unable to create '" + script_os_path.string() + "'");
			}

			script_os << "# This is a gnuplot (http://www.gnuplot.info/) script!" << '\n';
			script_os << "# comment fifth line, open gnuplot's terminal, move to script's path and launch 'load ";
			script_os << dataset_name + glob_cfg_.gnuplot_script_extension << "' if you want to run it" << '\n' << '\n';

			script_os << "reset" << '\n';
			script_os << "# cd '" << current_output_path.string() << "\'" << '\n';
			script_os << "set grid" << '\n' << '\n';

			// DENSITY
			script_os << "# DENSITY GRAPH (COLORS)" << '\n' << '\n';

            script_os << "set output '" + (current_output_path / path(output_density_graph)).string() + "'" << '\n';
			script_os << "set title " << LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])->GetTitle() << '\n' << '\n';

			script_os << "# " << kTerminal << " colors" << '\n';
			script_os << "set terminal " << kTerminal << " enhanced color font ',15'" << '\n' << '\n';

			script_os << "# Axes labels" << '\n';
			script_os << "set xlabel \"Density\"" << '\n';
			script_os << "set ylabel \"Execution Time [ms]\"" << '\n' << '\n';

			script_os << "# Get stats to set labels" << '\n';
			script_os << "stats[1:" << mode_cfg_.ccl_average_algorithms.size() << "] '" 
                + (current_output_path / path(output_density_results)).string() + "' matrix name 'density' noout" << '\n';
			script_os << "stats[1:" << mode_cfg_.ccl_average_algorithms.size() << "] '" 
                + (current_output_path / path(output_size_results)).string() + "' matrix name 'size' noout" << '\n';
			script_os << "ymax = density_max + (density_max / 100) * 10" << '\n';
			script_os << "ymin = density_min - (density_min / 100) * 10" << '\n';

			script_os << "# Axes range" << '\n';
			script_os << "set xrange [0:1]" << '\n';
			script_os << "set yrange [ymin:ymax]" << '\n';
			script_os << "set logscale y" << '\n' << '\n';

			script_os << "# Legend" << '\n';
			script_os << "set key inside left nobox spacing 2 font ', 8'" << '\n' << '\n';

			script_os << "# Plot" << '\n';
			script_os << "plot \\" << '\n';
			std::vector<std::string>::iterator it; // I need it after the cycle
			unsigned i = 2;
			for (it = mode_cfg_.ccl_average_algorithms.begin(); it != (mode_cfg_.ccl_average_algorithms.end() - 1); ++it, ++i) {
				script_os << "'" + (current_output_path / path(output_density_results)).string() + "' using 1:" << i << " with linespoints title \"" + DoubleEscapeUnderscore(std::string(*it)) + "\" , \\" << '\n';
			}
			script_os << "'" + (current_output_path / path(output_density_results)).string() + "' using 1:" << i << " with linespoints title \"" + DoubleEscapeUnderscore(std::string(*it)) + "\"" << '\n' << '\n';

			script_os << "# Replot in latex folder" << '\n';
			script_os << "set title \"\"" << '\n' << '\n';

			script_os << "set output \'" << (current_latex_path / path(compiler_name + compiler_version + output_density_graph)).string() << "\'" << '\n';
			script_os << "replot" << '\n' << '\n';

			script_os << "# DENSITY GRAPH (BLACK AND WHITE)" << '\n' << '\n';

			script_os << "set output '" + (current_output_path / path(output_density_bw_graph)).string() + "'" << '\n';
			script_os << "set title " << LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])->GetTitle() << '\n' << '\n';

			script_os << "# " << kTerminal << " black and white" << '\n';
			script_os << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << '\n' << '\n';

			script_os << "replot" << '\n' << '\n';

			// SIZE
			script_os << "# SIZE GRAPH (COLORS)" << '\n' << '\n';

			script_os << "set output '" + (current_output_path / path(output_size_graph)).string() + "'" << '\n';
			script_os << "set title " << LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])->GetTitle() << '\n' << '\n';

			script_os << "# " << kTerminal << " colors" << '\n';
			script_os << "set terminal " << kTerminal << " enhanced color font ',15'" << '\n' << '\n';

			script_os << "# Axes labels" << '\n';
			script_os << "set xlabel \"Pixels\"" << '\n';
			script_os << "set ylabel \"Execution Time [ms]\"" << '\n' << '\n';

			script_os << "# Get stats to set labels" << '\n';
			script_os << "ymax = size_max + (size_max/100)*30" << '\n';
			script_os << "ymin = size_min - (size_min/100)*30" << '\n';

			script_os << "# Axes range" << '\n';
			script_os << "set format x \"10^{%L}\"" << '\n';
			script_os << "set xrange [100:100000000]" << '\n';
			script_os << "set yrange [*:*]" << '\n';
			script_os << "set logscale xy 10" << '\n' << '\n';

			script_os << "# Legend" << '\n';
			script_os << "set key inside left top nobox spacing 2 font ', 8'" << '\n';

			script_os << "# Plot" << '\n';
			script_os << "plot \\" << '\n';
			i = 2;
			for (it = mode_cfg_.ccl_average_algorithms.begin(); it != (mode_cfg_.ccl_average_algorithms.end() - 1); ++it, ++i) {
				script_os << "'" + (current_output_path / path(output_size_results)).string() + "' using 1:" << i << " with linespoints title \"" + DoubleEscapeUnderscore(std::string(*it)) + "\" , \\" << '\n';
			}
			script_os << "'" + (current_output_path / path(output_size_results)).string() + "' using 1:" << i << " with linespoints title \"" + DoubleEscapeUnderscore(std::string(*it)) + "\"" << '\n' << '\n';

			script_os << "# Replot in latex folder" << '\n';
			script_os << "set title \"\"" << '\n';
			script_os << "set output \'" << (current_latex_path / path(compiler_name + compiler_version + output_size_graph)).string() << "\'" << '\n';
			script_os << "replot" << '\n' << '\n';

			script_os << "# SIZE (BLACK AND WHITE)" << '\n' << '\n';

			script_os << "set output '" + (current_output_path / path(output_size_bw_graph)).string() + "'" << '\n';
			script_os << "set title " << LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])->GetTitle() << '\n' << '\n';

			script_os << "# " << kTerminal << " black and white" << '\n';
			script_os << "set terminal " << kTerminal << " enhanced monochrome dashed font ',15'" << '\n' << '\n';

			script_os << "replot" << '\n' << '\n';

			script_os << "exit gnuplot" << '\n';

			density_os.close();
			size_os.close();
			script_os.close();
			// GNUPLOT SCRIPT
		}

		//if (0 != std::system(("gnuplot \"" + (current_output_path / path(dataset_name + glob_cfg_.gnuplot_script_extension)).string() + "\" 2> gnuplot_errors.txt").c_str())) {
		//	ob.Cwarning("Unable to run gnuplot script");
		//}
		ob.CloseBox();
	} // END DATASET FOR
	// LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])->ReleaseInput();
}

void YacclabTests::GranularityTest() {


    OutputBox ob(mode_cfg_.mode + " Granularity Test");

    std::string complete_results_suffix = "_results.txt",
	middle_results_suffix = "_run",
	granularity_results_suffix = "_granularity";

    std::random_device rd;
    std::mt19937 g(rd());
    
    
    tie_to_cpu();
    
    for (unsigned d = 0; d < mode_cfg_.granularity_datasets.size(); ++d) { // For every dataset in the granularity list

	std::string dataset_name(mode_cfg_.granularity_datasets[d]),
	    output_granularity_results = dataset_name + granularity_results_suffix,
	    output_granularity_graph = dataset_name + "_granularity" + kTerminalExtension,
	    output_granularity_bw_graph = dataset_name + "_granularity_bw" + kTerminalExtension;

	path father_dir_path;
	unsigned int dims = (LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0]))->GetInput()->Dims();
	if (dims == 2) {
	    father_dir_path = "random";
	}
	else if (dims == 3) {
	    father_dir_path = "random3D";
	}
	


	path dataset_path(glob_cfg_.input_path / father_dir_path / path(dataset_name)),
	    is_path = dataset_path / path(glob_cfg_.input_txt), // files.txt path
	    current_output_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / path(glob_cfg_.granularity_folder) / path(dataset_name)),
	    current_latex_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / glob_cfg_.latex_path),
	    output_broad_path = current_output_path / path(dataset_name + complete_results_suffix),
	    output_middle_results_path = current_output_path / path(glob_cfg_.middle_folder),
	    granularity_os_path = current_output_path / path(output_granularity_results);

	if (!create_directories(current_output_path)) {
	    ob.Cwarning("Unable to find/create the output path '" + current_output_path.string() + ", skipped", dataset_name);
	    continue;
	}

	// To save list of filename on which CLLAlgorithms must be tested
	std::vector<std::pair<std::string, bool>> filenames;  // first: filename, second: state of filename (find or not)
	if (!LoadFileList(filenames, is_path)) {
	    ob.Cwarning("Unable to open '" + is_path.string() + "', skipped", dataset_name);
	    continue;
	}
	if (!CheckFileList(dataset_path, filenames)) {
	    ob.Cwarning("Missing some files in dataset folder", dataset_name);
	}


	Results results;
	
	// Number of files
	unsigned int filenames_size = static_cast<unsigned>(filenames.size());

	constexpr uint8_t DENSITY = 101; // For granularity tests density ranges from 0 to 100 with step 1
        uint8_t granularity;
        if (dims == 2) {
            granularity = TestsConf::kGranularities2D; // For granularity tests granularity ranges from 1 to 16 with step 1
        }
        if (dims == 3) {
	    granularity = TestsConf::kGranularities3D;
	}	

	results.Init(mode_cfg_.ccl_average_algorithms, filenames_size);
	
	// Initialize results container
	granularity_results_[dataset_name] = cv::Mat(
	    cv::Size(static_cast<unsigned>(mode_cfg_.ccl_average_algorithms.size()), DENSITY),
	    CV_64FC(granularity), cv::Scalar(0)); // To store minimum values 
 	// std::vector<std::vector<double>> real_densities(granularity, std::vector<double>(density));
							 
	cv::Mat1d min_res(filenames_size, static_cast<unsigned>(
			      mode_cfg_.ccl_average_algorithms.size()),
			  std::numeric_limits<double>::max());
	
	cv::Mat1d current_res(filenames_size, static_cast<unsigned>(
				  mode_cfg_.ccl_average_algorithms.size()),
			      std::numeric_limits<double>::max()); // To store current result
	
	cv::Mat1i labels(filenames_size, static_cast<unsigned>(
			     mode_cfg_.ccl_average_algorithms.size()), 0); // To count number of labels for every image and algorithm 

	// Start output message box
	ob.StartRepeatedBox(dataset_name, filenames_size, mode_cfg_.granularity_tests_number);

	if (mode_cfg_.granularity_save_middle_tests) {
	    if (!create_directories(output_middle_results_path)) {
		ob.Cwarning("Unable to find/create the output path '" + output_middle_results_path.string() + "', middle results won't be saved", dataset_name);
	    }
	}

	std::map<std::string, size_t> algo_pos;
	for (size_t i = 0; i < mode_cfg_.ccl_average_algorithms.size(); ++i)
	    algo_pos[mode_cfg_.ccl_average_algorithms[i]] = i;
	auto shuffled_ccl_average_algorithms = mode_cfg_.ccl_average_algorithms;

	shuffle(begin(filenames), end(filenames), g);
	
	// Test is executed n_test times
	for (unsigned test = 0; test < mode_cfg_.granularity_tests_number; ++test) {
	    // For every file in list
	    for (unsigned file = 0; file < filenames.size(); ++file) {
		// Display output message box
		ob.UpdateRepeatedBox(file);

		std::string filename = filenames[file].first;
		path filename_path = dataset_path / path(filename);

		//int cur_granularity = stoi(filename.substr(0, 2));
		//if (cur_granularity < 15) continue;
		//int cur_density = stoi(filename.substr(2, 3));

		// Read and load image
		//if (!GetBinaryImage(filename_path, Labeling::img_)) {
		//	ob.Cwarning("Unable to open '" + filename + "', granularity results/charts will miss some data");
		//	continue;
		//}
		ImageMetadata metadata;
		if (!LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])
		    ->GetInput()->ReadBinary(filename_path.string(), metadata)) {
		    
		    ob.Cwarning("Unable to open '" + filename + "'");
		    continue;
		}

		// int nonzero = countNonZero(Labeling::img_);
		// real_densities[cur_granularity - 1][cur_density] = 100.0 * nonzero / (Labeling::img_.rows*Labeling::img_.cols);
		shuffle(begin(shuffled_ccl_average_algorithms), end(shuffled_ccl_average_algorithms), g);


		std::string image_name = filename_path.stem().string();
		int granularity = stoi(image_name.substr(0, 2));
		int density = stoi(image_name.substr(2, 3));		
		
		for (const auto& algo_name : shuffled_ccl_average_algorithms) {
		    Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);
		    unsigned i = static_cast<unsigned>(algo_pos[algo_name]);

		    
		    size_t img_size = metadata.depth * metadata.height * metadata.width;
		    
		    try {
			// Perform current algorithm on current image and save result.
			/*algorithm->Alloc(); // Do not save allocation time
			algorithm->perf_.start();
			algorithm->samplers.Start();			    
			
			algorithm->PerformLabeling();
			algorithm->perf_.stop();
			algorithm->samplers.Stop();

			algorithm->perf_.store(Step(StepType::ALL_SCANS),
					       algorithm->perf_.last());
			algorithm->samplers.Store(StepType::ALL_SCANS, img_size);
			algorithm->Dealloc();*/
			algorithm->PerformLabelingWithSteps();
		    }
		    catch (const std::exception& e) {
			algorithm->FreeLabelingData();
			ob.Cerror("Something wrong with " + algo_name + ": " + e.what()); // You should check your algorithms' implementation before performing YACCLAB tests  
		    }
		    
		    // Save time results
		    WriteAlgorithmSamplesSteps(algorithm, algo_name, file, results);

		    double total_time = algorithm->perf_.get(Step(StepType::ALL_SCANS));
		    if (total_time < min_res(file, i)) {
			min_res(file, i) = total_time;
		    }
		    
		    algorithm->FreeLabelingData();
		} // END ALGORITHMS FOR
	    } // END FILES FOR.
	    ob.StopRepeatedBox(false);

	    // Save middle results if necessary
	    if (mode_cfg_.granularity_save_middle_tests) {
		std::string output_middle_results_file =
		    (output_middle_results_path / path(dataset_name + middle_results_suffix + "_"
						       + std::to_string(test) + ".txt")).string();
		
		results.SaveBroadAverageSteps(mode_cfg_.ccl_average_algorithms,
					 output_middle_results_path, dataset_name,
					 middle_results_suffix, test, labels, filenames);
	    }
	} // END TEST FOR

	// To write the min results into a file
	SaveBroadOutputResultsMat(min_res, output_broad_path.string(), labels, filenames, mode_cfg_.ccl_average_algorithms);
	for (int r = 0; r < min_res.rows; ++r) {
	    if (filenames[r].second) {

		std::string cur_filename = filenames[r].first;
		int cur_granularity = stoi(cur_filename.substr(0, 2));
		int cur_density = stoi(cur_filename.substr(2, 3));

		for (int c = 0; c < min_res.cols; ++c) {
		    if (dims == 2) {
			granularity_results_[dataset_name].at<cv::Vec<double, TestsConf::kGranularities2D>>(
			    cur_density, c)[cur_granularity - 1] += min_res(r, c);
		    }
		    else if (dims == 3) {
			granularity_results_[dataset_name].at<cv::Vec<double, TestsConf::kGranularities3D>>(
			    cur_density, c)[cur_granularity - 1] += min_res(r, c);
		    }
		}
	    }
	    else {
		throw;
	    }
	}

	GranularityArgs args(granularity, DENSITY, dims);
	results.SaveGranularityResults(mode_cfg_.ccl_average_algorithms,
				       granularity_os_path.string(), dataset_name,
				       middle_results_suffix, labels, filenames, args);
	
	
	// For SCRIPT which runs multiple times the gnuplot script
	std::string main_script = (current_output_path / path("main_script" + glob_cfg_.system_script_extension)).string();
	std::ofstream main_script_os(main_script);
	if (!main_script_os.is_open()) {
	    ob.Cwarning("Unable to open '" + main_script + "'");
	    return;
	}


	// To write granularity results on specified file
	SaveGranularityResults(granularity_os_path.string(), dataset_name, args);
	
	ob.CloseBox();
    } // END DATASET FOR
    // LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_average_algorithms[0])->ReleaseInput();
}


void YacclabTests::MemoryTest() {
	// Initialize output message box
	OutputBox ob(mode_cfg_.mode + " Memory Test");

	path current_output_path(glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / path(glob_cfg_.memory_folder));

	std::string output_file((current_output_path.string() / path(glob_cfg_.memory_file)).string());

	if (!create_directories(current_output_path)) {
		ob.Cwarning("Unable to find/create the output path '" + current_output_path.string() + "', 'memory test' skipped");
		ob.CloseBox();
		return;
	}

	// To write MEMORY results
	std::ofstream os(output_file);
	if (!os.is_open()) {
		ob.Cwarning("Unable to open '" + output_file + "', 'memory test' skipped");
		ob.CloseBox();
		return;
	}
	os << "#Average number of accesses" << '\n';

	std::random_device rd;
	std::mt19937 g(rd());

	for (unsigned d = 0; d < mode_cfg_.memory_datasets.size(); ++d) { // For every dataset in the average list
		std::string dataset_name(mode_cfg_.memory_datasets[d]);

		path dataset_path(glob_cfg_.input_path / path(dataset_name)),
			is_path = dataset_path / path(glob_cfg_.input_txt); // files.txt path

		// To save list of filename on which CLLAlgorithms must be tested
		std::vector<std::pair<std::string, bool>> filenames;  // first: filename, second: state of filename (find or not)
		if (!LoadFileList(filenames, is_path)) {
			ob.Cwarning("Unable to open '" + is_path.string() + "', skipped", dataset_name);
			continue;
		}

		// Number of files
		unsigned int filenames_size = static_cast<unsigned>(filenames.size());

		unsigned tot_test = 0; // To count the real number of image on which labeling will be applied for every file in list

		// Initialize results container
		// To store average memory accesses (one column for every data_ structure type: col 1 -> BINARY_MAT, col 2 -> LABELED_MAT, col 3 -> EQUIVALENCE_VET, col 0 -> OTHER)
		memory_accesses_[dataset_name] = cv::Mat1d(cv::Size(MD_SIZE, static_cast<unsigned>(mode_cfg_.ccl_mem_algorithms.size())), 0.0);

		// Start output message box
		ob.StartUnitaryBox(dataset_name, filenames_size);

		// For every file in list
		for (unsigned file = 0; file < filenames.size(); ++file) {
			// Display output message box
			ob.UpdateUnitaryBox(file);

			std::string filename = filenames[file].first;
			path filename_path = dataset_path / path(filename);

			// Read and load image
			//if (!GetBinaryImage(filename_path, Labeling::img_)) {
			//	ob.Cwarning("Unable to open '" + filename + "'");
			//	continue;
			//}
			ImageMetadata metadata;
			if (!LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_mem_algorithms[0])
			    ->GetInput()->ReadBinary(filename_path.string(), metadata)) {
				ob.Cwarning("Unable to open '" + filename + "'");
				continue;
			}

			++tot_test;

			// For all the Algorithms in the array
			for (unsigned i = 0; i < static_cast<unsigned>(mode_cfg_.ccl_mem_algorithms.size()); ++i) {
				Labeling *algorithm = LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_mem_algorithms[i]);

				// The following data_ structure is used to get the memory access matrices
				std::vector<uint64_t> accesses; // Rows represents algorithms and columns represent data_ structures

				try {
					algorithm->PerformLabelingMem(accesses);
				}
				catch (const std::exception& e) {
					algorithm->FreeLabelingData();
					ob.Cerror("Something wrong with " + mode_cfg_.ccl_mem_algorithms[i] + ": " + e.what()); // You should check your algorithms' implementation before performing YACCLAB tests  
				}

				// For every data_ structure "returned" by the algorithm
				for (unsigned a = 0; a < static_cast<unsigned>(accesses.size()); ++a) {
					memory_accesses_[dataset_name](i, a) += accesses[a];
				}
				algorithm->FreeLabelingData();
			}
		}
		ob.StopUnitaryBox();

		// To calculate average memory accesses
		for (int r = 0; r < memory_accesses_[dataset_name].rows; ++r) {
			for (int c = 0; c < memory_accesses_[dataset_name].cols; ++c) {
				memory_accesses_[dataset_name](r, c) /= tot_test;
			}
		}

		os << "#" << dataset_name << '\n';
		os << "Algorithm\tBinary Image\tLabel Image\tEquivalence Vector/s\tOther\tTotal Accesses" << '\n';

		for (unsigned a = 0; a < static_cast<unsigned>(mode_cfg_.ccl_mem_algorithms.size()); ++a) {
			double total_accesses{ 0.0 };
			os << mode_cfg_.ccl_mem_algorithms[a] << '\t';
			for (int col = 0; col < memory_accesses_[dataset_name].cols; ++col) {
				os << std::fixed << std::setprecision(0) << memory_accesses_[dataset_name](a, col);
				os << '\t';
				total_accesses += memory_accesses_[dataset_name](a, col);
			}

			os << total_accesses;
			os << '\n';
		}

		os << '\n' << '\n';;
	} // END DATASET FOR
	// LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_mem_algorithms[0])->ReleaseInput();
	os.close();
}

void YacclabTests::LatexGenerator() {
	OutputBox ob("Generation of Latex file/s for mode " + mode_cfg_.mode);
	path latex = glob_cfg_.glob_output_path / mode_cfg_.mode_output_path / glob_cfg_.latex_path / path(glob_cfg_.latex_file);
	std::ofstream os(latex.string());
	if (!os.is_open()) {
		ob.Cwarning("Unable to open/create '" + latex.string() + "', generation skipped");
		return;
	}

	// fixed number of decimal values
	os << std::fixed;
	os << std::setprecision(3);

	// Document begin
	os << "%These file is generated by YACCLAB. Follow our project on GitHub: https://github.com/prittt/YACCLAB" << '\n' << '\n';
	os << "\\documentclass{article}" << '\n' << '\n';

	os << "\\usepackage{siunitx}" << '\n';
	os << "\\usepackage{graphicx}" << '\n';
	os << "\\usepackage{subcaption}" << '\n';
	os << "\\usepackage[top = 1in, bottom = 1in, left = 1in, right = 1in]{geometry}" << '\n' << '\n';

	os << "\\title{{ \\huge\\bfseries YACCLAB TESTS}}" << '\n';
	os << "\\date{" + GetDatetime() + "}" << '\n';
	os << "\\author{}" << '\n' << '\n';
	os << "\\begin{document}" << '\n' << '\n';
	os << "\\maketitle" << '\n' << '\n';

	// Section average results table ------------------------------------------------------------------------------------------
	if (mode_cfg_.perform_average) {
		os << "\\section{Average Table Results}" << '\n' << '\n';

		os << "\\begin{table}[tbh]" << '\n' << '\n';
		os << "\t\\centering" << '\n';
		os << "\t\\caption{Average Results in ms (Lower is better)}" << '\n';
		os << "\t\\label{tab:table1}" << '\n';
		os << "\t\\begin{tabular}{|l|";
		for (unsigned i = 0; i < mode_cfg_.ccl_average_algorithms.size(); ++i)
			os << "S[table-format=2.3]|";
		os << "}" << '\n';
		os << "\t\\hline" << '\n';
		os << '\t';
		for (unsigned i = 0; i < mode_cfg_.ccl_average_algorithms.size(); ++i) {
			//RemoveCharacter(datasets_name, '\\');
			//datasets_name.erase(std::remove(datasets_name.begin(), datasets_name.end(), '\\'), datasets_name.end());
			os << " & {" << EscapeUnderscore(mode_cfg_.ccl_average_algorithms[i]) << "}"; //Header
		}
		os << "\\\\" << '\n';
		os << "\t\\hline" << '\n';
		for (unsigned i = 0; i < mode_cfg_.average_datasets.size(); ++i) {
			os << '\t' << mode_cfg_.average_datasets[i];
			for (int j = 0; j < average_results_.cols; ++j) {
				os << " & ";
				if (average_results_(i, j) != std::numeric_limits<double>::max())
					os << average_results_(i, j); //Data
			}
			os << "\\\\" << '\n';
		}
		os << "\t\\hline" << '\n';
		os << "\t\\end{tabular}" << '\n' << '\n';
		os << "\\end{table}" << '\n';
	}

	{ // CHARTS SECTION ------------------------------------------------------------------------------------------
		std::string info_to_latex = SystemInfo::build() + "_" + SystemInfo::compiler_name() + SystemInfo::compiler_version() + "_" + SystemInfo::os();
		std::replace(info_to_latex.begin(), info_to_latex.end(), ' ', '_');
		info_to_latex = EscapeUnderscore(info_to_latex);

		std::string chart_size{ "0.45" }, chart_width{ "1" };
		// Get information about date and time
		std::string datetime = GetDatetime();

		std::string compiler_name(SystemInfo::compiler_name());
		std::string compiler_version(SystemInfo::compiler_version());
		//replace the . with _ for compiler strings
		std::replace(compiler_version.begin(), compiler_version.end(), '.', '_');

		// SECTION AVERAGE CHARTS  ---------------------------------------------------------------------------
		if (mode_cfg_.perform_average) {
			os << "\\section{Average Charts}" << '\n' << '\n';
			os << "\\begin{figure}[tbh]" << '\n' << '\n';
			// \newcommand{ \machineName }{x86\_MSVC15.0\_Windows\_10\_64\_bit}
			os << "\t\\newcommand{\\machineName}{";
			os << info_to_latex << "}" << '\n';
			// \newcommand{\compilerName}{MSVC15_0}
			os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << '\n';
			os << "\t\\centering" << '\n';

			for (unsigned i = 0; i < mode_cfg_.average_datasets.size(); ++i) {
				os << "\t\\begin{subfigure}[tbh]{" + chart_size + "\\textwidth}" << '\n';
				os << "\t\t\\caption{" << mode_cfg_.average_datasets[i] + "}" << '\n';
				os << "\t\t\\centering" << '\n';
				os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName_" + mode_cfg_.average_datasets[i] + ".pdf}" << '\n';
				os << "\t\\end{subfigure}" << '\n' << '\n';
			}
			os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << '\n' << '\n';
			os << "\\end{figure}" << '\n' << '\n';
		}

		// SECTION AVERAGE WITH STEPS CHARTS  ---------------------------------------------------------------------------
		if (mode_cfg_.perform_average_ws) {
			std::string average_ws_suffix{ "_with_steps_" };

			os << "\\section{Average With Steps Charts}" << '\n' << '\n';
			os << "\\begin{figure}[tbh]" << '\n' << '\n';
			// \newcommand{ \machineName }{x86\_MSVC15.0\_Windows\_10\_64\_bit}
			os << "\t\\newcommand{\\machineName}{";
			os << info_to_latex << "}" << '\n';
			// \newcommand{\compilerName}{MSVC15_0}
			os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << '\n';
			os << "\t\\centering" << '\n';
			for (unsigned i = 0; i < mode_cfg_.average_ws_datasets.size(); ++i) {
				os << "\t\\begin{subfigure}[tbh]{" + chart_size + "\\textwidth}" << '\n';
				os << "\t\t\\caption{" << mode_cfg_.average_ws_datasets[i] + "}" << '\n';
				os << "\t\t\\centering" << '\n';
				os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName_" + average_ws_suffix + mode_cfg_.average_ws_datasets[i] + ".pdf}" << '\n';
				os << "\t\\end{subfigure}" << '\n' << '\n';
			}
			os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << '\n' << '\n';
			os << "\\end{figure}" << '\n' << '\n';
		}

		// SECTION DENSITY CHARTS  ---------------------------------------------------------------------------
		if (mode_cfg_.perform_density) {
			std::vector<std::string> density_datasets{ "density", "size" };

			os << "\\section{Density Charts}" << '\n' << '\n';
			os << "\\begin{figure}[tbh]" << '\n' << '\n';
			// \newcommand{ \machineName }{x86\_MSVC15.0\_Windows\_10\_64\_bit}
			os << "\t\\newcommand{\\machineName}{";
			os << info_to_latex << "}" << '\n';
			// \newcommand{\compilerName}{MSVC15_0}
			os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << '\n';
			os << "\t\\centering" << '\n';

			for (unsigned i = 0; i < density_datasets.size(); ++i) {
				os << "\t\\begin{subfigure}[tbh]{" + chart_size + "\\textwidth}" << '\n';
				os << "\t\t\\caption{" << density_datasets[i] + "}" << '\n';
				os << "\t\t\\centering" << '\n';
				os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName_" + density_datasets[i] + ".pdf}" << '\n';
				os << "\t\\end{subfigure}" << '\n' << '\n';
			}
			os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << '\n' << '\n';
			os << "\\end{figure}" << '\n' << '\n';
		}

		// SECTION GRANULARITY CHARTS  ---------------------------------------------------------------------------
		if (mode_cfg_.perform_granularity) {
			//vector<std::string> density_datasets{ "density", "size" };

			os << "\\section{Granularity Charts}" << '\n' << '\n';
			os << "\\begin{figure}[tbh]" << '\n' << '\n';
			// \newcommand{ \machineName }{x86\_MSVC15.0\_Windows\_10\_64\_bit}
			os << "\t\\newcommand{\\machineName}{";
			os << info_to_latex << "}" << '\n';
			// \newcommand{\compilerName}{MSVC15_0}
			os << "\t\\newcommand{\\compilerName}{" + compiler_name + compiler_version + "}" << '\n';
			os << "\t\\centering" << '\n';

			for (unsigned i = 0; i < mode_cfg_.granularity_datasets.size(); ++i) {
				os << "\t\\begin{subfigure}[tbh]{" + chart_size + "\\textwidth}" << '\n';
				os << "\t\t\\caption{" << mode_cfg_.granularity_datasets[i] + "}" << '\n';
				os << "\t\t\\centering" << '\n';
				os << "\t\t\\includegraphics[width=" + chart_width + "\\textwidth]{\\compilerName_" + mode_cfg_.granularity_datasets[i] + ".pdf}" << '\n';
				os << "\t\\end{subfigure}" << '\n' << '\n';
			}
			os << "\t\\caption{\\machineName \\enspace " + datetime + "}" << '\n' << '\n';
			os << "\\end{figure}" << '\n' << '\n';
		}
	} // END CHARTS SECTION

	// SECTION MEMORY RESULT TABLE ---------------------------------------------------------------------------
	if (mode_cfg_.perform_memory) {
		os << "\\section{Memory Accesses tests}" << '\n' << '\n';
		os << "Analysis of memory accesses required by connected components computation. The numbers are given in millions of accesses." << '\n';

		for (const auto& dataset : memory_accesses_) {
			const auto& dataset_name = dataset.first;
			const auto& accesses = dataset.second;

			os << "\\begin{table}[tbh]" << '\n' << '\n';
			os << "\t\\centering" << '\n';
			os << "\t\\caption{Memory accesses on ``" << dataset_name << "'' dataset " << '\n';
			os << "\t\\label{tab:table_" << dataset_name << "}" << '\n';
			os << "\t\\begin{tabular}{|l|";
			for (int i = 0; i < accesses.cols + 1; ++i)
				os << "S[table-format=2.3]|";
			os << "}" << '\n';
			os << "\t\\hline" << '\n';
			os << '\t';

			// Header
			os << "{Algorithm} & {Binary Image} & {Label Image} & {Equivalence Vector/s}  & {Other} & {Total Accesses}";
			os << "\\\\" << '\n';
			os << "\t\\hline" << '\n';

			for (unsigned i = 0; i < mode_cfg_.ccl_mem_algorithms.size(); ++i) {
				// For every algorithm escape the underscore
				const std::string& alg_name = EscapeUnderscore(mode_cfg_.ccl_mem_algorithms[i]);
				//RemoveCharacter(alg_name, '\\');
				os << "\t{" << alg_name << "}";

				double tot = 0;

				for (int s = 0; s < accesses.cols; ++s) {
					// For every data_ structure

					os << "\t& " << (accesses(i, s) / 1000000);

					tot += (accesses(i, s) / 1000000);
				}
				// Total Accesses
				os << "\t& " << tot;

				// EndLine
				os << "\t\\\\" << '\n';
			}

			// EndTable
			os << "\t\\hline" << '\n';
			os << "\t\\end{tabular}" << '\n' << '\n';
			os << "\\end{table}" << '\n';
		}
	}

	os << "\\end{document}";
	os.close();
}
