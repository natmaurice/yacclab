// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_TESTS_PERFORMER_H_
#define YACCLAB_TESTS_PERFORMER_H_

#include <string>
#include <map>
#include <memory>

#include <opencv2/imgproc.hpp>

#include "file_manager.h"
#include "system_info.h"
#include "utilities.h"
#include "config_data.h"
#include "labeling_algorithms.h"
#include "progress_bar.h"


using namespace filesystem;


struct TestsConf {
    constexpr static uint8_t kGranularities2D = 16;
    constexpr static uint8_t kSamples2D = 10;
	
    constexpr static uint8_t kGranularities3D = 16;
    constexpr static uint8_t kSamples3D = 3;
};

struct GranularityArgs {
    GranularityArgs(double g, double density, int dims) : g(g), density(density), dims(dims) {
    }
	
    double g;
    double density;
    int dims;
};


class YacclabTests {

private:
	OutputBox ob_;
	std::error_code &ec_;
	ModeConfig &mode_cfg_;
	GlobalConfig &glob_cfg_;
	const path output_path;

	cv::Mat1d average_results_;
	cv::Mat1d density_results_;
	std::map<std::string, cv::Mat> granularity_results_;

	std::map<std::string, cv::Mat1d> average_ws_results_; // String for dataset_name, Mat1d for steps results
	std::map<std::string, cv::Mat1d> memory_accesses_; // String for dataset_name, Mat1d for memory accesses
    
public:
    YacclabTests(ModeConfig &mode_cfg, GlobalConfig &glob_cfg, std::error_code &ec) : 
	ec_(ec), mode_cfg_(mode_cfg), glob_cfg_(glob_cfg), output_path(glob_cfg.glob_output_path / mode_cfg.mode_output_path) {}

    void CheckPerformLabeling() {
	std::string title = "Checking Correctness of 'PerformLabeling()'";
	CheckAlgorithms(title, mode_cfg_.ccl_average_algorithms, &Labeling::PerformLabeling);
    }
    void CheckPerformLabelingWithSteps() {
	std::string title = "Checking Correctness of 'PerformLabelingWithSteps()'";
	CheckAlgorithms(title, mode_cfg_.ccl_average_ws_algorithms, &Labeling::PerformLabelingWithSteps);
    }
    void CheckPerformLabelingMem() {
	std::string title = "Checking Correctness of 'PerformLabelingMem()'";
	std::vector<uint64_t> unused;
	CheckAlgorithms(title, mode_cfg_.ccl_mem_algorithms, &Labeling::PerformLabelingMem, unused);
    }
    
    void InitialOperations();

    void AverageTest();
    void AverageTestWithSteps();
    void DensityTest();
    void GranularityTest();
    void MemoryTest();
    void LatexGenerator();

    void ParallelAverageTestWithSteps();
    void ParallelGranularityTest();
    
    ~YacclabTests() {
        LabelingMapSingleton::GetLabeling(mode_cfg_.ccl_existing_algorithms[0])->GetInput()->Release();
    }

private:

    void CheckAlgorithmsExistence();
    void CheckMethodsExistence();
    void CheckDatasets();
    void CreateDirectories();

    bool LoadFileList(std::vector<std::pair<std::string, bool>>& filenames, const path& files_path);
    bool CheckFileList(const path& base_path, std::vector<std::pair<std::string, bool>>& filenames);
    bool SaveBroadOutputResults(std::map<std::string, cv::Mat1d>& results,
				const std::string& o_filename, const cv::Mat1i& labels,
				const std::vector<std::pair<std::string, bool>>& filenames,
				const std::vector<std::string>& ccl_algorithms);
    
    /*bool SaveBroadOutputResults(const cv::Mat1d& results, const std::string& o_filename, const cv::Mat1i& labels,
				const std::vector<std::pair<std::string, bool>>& filenames, const std::vector<std::string>& ccl_algorithms);
    bool SaveBroadAverageWithStepsResults(std::map<std::string, cv::Mat1d>& results,
					  const std::string& o_filename, const cv::Mat1i& labels, 
					  const std::vector<std::pair<std::string, bool>>& filenames,
					  const std::vector<std::string>& ccl_algorithms);*/
    
    void SaveAverageWithStepsResults(const std::string& os_name, const std::string& dataset_name, bool rounded);


    void SaveGranularityResults(const std::string& os_name, const std::string& dataset_name,
				const GranularityArgs& args);
    
    template <typename FnP, typename... Args>
    void CheckAlgorithms(const std::string& title, const std::vector<std::string>& ccl_algorithms, const FnP func, Args&&... args) {

	OutputBox ob(title);

	std::vector<bool> stats(ccl_algorithms.size(), true);  // True if the i-th algorithm is correct, false otherwise
	std::vector<std::string> first_fail(ccl_algorithms.size());  // Name of the file on which algorithm fails the first time
	bool stop = false; // True if all the algorithms are not correct

	std::string correct_algo_name;
	try {
	    correct_algo_name = LabelingMapSingleton::GetLabeling(ccl_algorithms[0])->CheckAlg();
	}
	catch (std::out_of_range) {
	    ob.Cwarning("No correct algorithm is available, correctness test skipped.");
	    return;
	}
	Labeling* correct_algo = LabelingMapSingleton::GetLabeling(correct_algo_name);

	for (unsigned i = 0; i < mode_cfg_.check_datasets.size(); ++i) { // For every dataset in the check_datasets list
	    std::string dataset_name(mode_cfg_.check_datasets[i]);
	    path dataset_path(glob_cfg_.input_path / path(dataset_name));
	    path is_path = dataset_path / path(glob_cfg_.input_txt); // files.txt path

	    // Load list of images on which ccl_algorithms must be tested
	    std::vector<std::pair<std::string, bool>> filenames; // first: filename, second: state of filename (find or not)
	    if (!LoadFileList(filenames, is_path)) {
		ob.Cwarning("Unable to open '" + is_path.string() + "'", dataset_name);
		continue;
	    }

	    // Number of files
	    size_t filenames_size = filenames.size();
	    ob.StartUnitaryBox(dataset_name, filenames_size);

	    for (unsigned file = 0; file < filenames_size && !stop; ++file) { // For each file in list
		ob.UpdateUnitaryBox(file);

		std::string filename = filenames[file].first;
		path filename_path = dataset_path / path(filename);

		// Load image
		ImageMetadata metadata;
		if (!LabelingMapSingleton::GetLabeling(ccl_algorithms[0])->GetInput()
		    ->ReadBinary(filename_path.string(), metadata)) {
		    ob.Cmessage("Unable to open '" + filename + "'");
		    continue;
		}

		// These variables aren't necessary
		// unsigned n_labels_correct, n_labels_to_control;

		correct_algo->PerformLabeling();
		//n_labels_correct = sauf->n_labels_;
                YacclabTensorOutput* correct_algo_out = correct_algo->GetOutput();
                correct_algo_out->PrepareForCheck();

                std::unique_ptr<YacclabTensorOutput> labels_correct = correct_algo_out->Copy();
		Features correct_features = correct_algo->features.Copy();
		
                correct_algo->FreeLabelingData();

		if (correct_algo->IsCCA()) {
		    labels_correct->NormalizeLabelsFeatures(correct_features, correct_algo->IsLabelBackground(), true);
		} else {
		    labels_correct->NormalizeLabels(correct_algo->IsLabelBackground());
		}

		unsigned j = 0;
		for (const auto& algo_name : ccl_algorithms) {
		    Labeling *algorithm = LabelingMapSingleton::GetLabeling(algo_name);

		    // Perform labeling on current algorithm if it has no previously failed
		    if (stats[j]) {
			algorithm->Alloc();
			(algorithm->*func)(std::forward<Args>(args)...);
			algorithm->Dealloc();
			
                        YacclabTensorOutput* labels_to_check = algorithm->GetOutput();
			Features features = algorithm->features.Copy();
			
                        labels_to_check->PrepareForCheck();

			if (correct_algo->IsCCA()) {
			    labels_to_check->NormalizeLabelsFeatures(features, algorithm->IsLabelBackground(), true);
			} else {
			    labels_to_check->NormalizeLabels(algorithm->IsLabelBackground());
			}
                        bool correct = true;


			
			if (algorithm->UseRelabeling()) {
			    correct = labels_correct->Equals(labels_to_check);
			}

			unsigned n_labels = algorithm->n_labels_;
			if (n_labels != correct_algo->n_labels_) {
			    std::cout << "CORRECT = " << correct_algo_name << "\n";
			    std::cout << "Wrong number of labels\n";
			    std::cout << "Labels = " << n_labels << ", expected = "
				      << correct_algo->n_labels_ << " \n";
			    //assert(false);
			    correct = false;
			}

			
			if (correct && algorithm->IsCCA() && correct_algo->IsCCA()) {
			    
			    correct = features.Equals<ConfFeatures3DAll>(correct_features, n_labels);
			}
			
                        // const bool diff = algorithm->Check(correct_algo);
			algorithm->FreeLabelingData();
			if (!correct) {
			    stats[j] = false;
			    first_fail[j] = (path(dataset_name) / path(filename)).string();

			    // Stop check test if all the algorithms fail
			    if (adjacent_find(stats.begin(), stats.end(), std::not_equal_to<int>()) == stats.end()) {
				stop = true;
				break;
			    }
			}
		    }
		    ++j;
		} // For all the Algorithms in the array
                //correct_algo->FreeLabelingData();

	    }// END WHILE (LIST OF IMAGES)
	    ob.StopUnitaryBox();
	}// END FOR (LIST OF DATASETS)

	// LabelingMapSingleton::GetLabeling(ccl_algorithms[0])->ReleaseInput();

	// To display report of correctness test
	std::vector<std::string> messages(static_cast<unsigned int>(ccl_algorithms.size()));
	unsigned longest_name = static_cast<unsigned>(max_element(ccl_algorithms.begin(), ccl_algorithms.end(), CompareLengthCvString)->length());

	unsigned j = 0;
	for (const auto& algo_name : ccl_algorithms) {
	    messages[j] = "'" + algo_name + "'" + std::string(longest_name - algo_name.size(), '-');
	    if (stats[j]) {
		messages[j] += "-> correct!";
	    }
	    else {
		messages[j] += "-> NOT correct, it first fails on '" + first_fail[j] + "'";
	    }
	    ++j;
	}
	ob.DisplayReport("Report", messages);
    }
};

//using TestsPerfPtr = std::unique_ptr<YacclabTests>;
//
//TestsPerfPtr YacclabTestsFactory(ModeConfig mode_cfg, GlobalConfig glob_cfg, std::error_code& ec) {
//	TestsPerfPtr ptr;
//	if (mode_cfg.mode == "2D_CPU") {
//		ptr = std::make_unique<YacclabTests>(mode_cfg, glob_cfg, ec);
//	}
//	else if (mode_cfg.mode == "3D_CPU") {
//		ptr = std::make_unique<YacclabTests>(mode_cfg, glob_cfg, ec);
//	}
//#if defined YACCLAB_WITH_CUDA
//	else if (mode_cfg.mode == "2D_GPU") {
//		ptr = std::make_unique<YacclabTests>(mode_cfg, glob_cfg, ec);
//	}
//	else if (mode_cfg.mode == "3D_GPU") {
//		ptr = std::make_unique<YacclabTests>(mode_cfg, glob_cfg, ec);
//	}
//#endif
//	else ptr = nullptr;
//	return ptr;
//}

#endif // !YACCLAB_TESTS_PERFORMER_H_
