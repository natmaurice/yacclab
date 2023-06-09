// Copyright (c) 2020, the YACCLAB contributors, as 
// shown by the AUTHORS file. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef YACCLAB_UTILITIES_H_
#define YACCLAB_UTILITIES_H_

#include <string>
#include <opencv2/core.hpp>

#include "file_manager.h"
#include "stream_demultiplexer.h"

extern const std::string kTerminal;
extern const std::string kTerminalExtension;

// extern struct ConfigData cfg;

namespace dmux {
    extern class StreamDemultiplexer cout;
}

// To compare lengths of OpenCV String
bool CompareLengthCvString(cv::String const& lhs, cv::String const& rhs);

// This function is useful to delete eventual carriage return from a string
// and is especially designed for windows file newline format
//void DeleteCarriageReturn(std::string& s);
void RemoveCharacter(std::string& s, const char c);

// This function take a char as input and return the corresponding int value (not ASCII one)
unsigned ctoi(const char& c);

// This function help us to manage '\' escape character
//void EraseDoubleEscape(std::string& str);

/*@brief Get information about date and time

@param[in] bool if true substitute both ' ' and ':' chars with '_' and '.'

@return string value with datetime stringyfied
*/
std::string GetDatetime();
std::string GetDatetimeWithoutSpecialChars();

// Create a bunch of pseudo random colors from labels indexes and create a
// color representation for the labels
//void ColorLabels(const cv::Mat1i& img_labels, cv::Mat3b& img_out);
//void ColorLabels(const cv::Mat& img_labels, cv::Mat& img_out);
//
//// This function may be useful to compare the output of different labeling procedures
//// which may assign different labels to the same object. Use this to force a row major
//// ordering of labels.
//void NormalizeLabels(cv::Mat1i& img_labels);
//void NormalizeLabels(cv::Mat& img_labels);
//
//// Get binary image given a image's filename;
//bool GetBinaryImage(const std::string& filename, cv::Mat& binary_mat);
//bool GetBinaryImage(const filesystem::path& p, cv::Mat& binary_mat);

// Compare two int matrices element by element
//bool CompareMat(const cv::Mat& mat_a, const cv::Mat& mat_b);

//void Divide(cv::Mat& mat);

bool CheckLabeledImage(const cv::Mat1b& img, const cv::Mat1i& labels, cv::Mat1i& errors);

bool CheckLabeledVolume(const cv::Mat& img, const cv::Mat& labels, cv::Mat& errors);

/*@brief Read bool from YAML configuration file

@param[in] node_list FileNode that contain bool data_
@return bool value of field in node_list
*/
//bool ReadBool(const cv::FileNode& node_list);

// Hide blinking cursor from console
void HideConsoleCursor();

void ShowConsoleCursor();

int RedirectCvError(int status, const char* func_name, const char* err_msg, const char* file_name, int line, void*);

/*
@brief Return the string title to insert in gnuplot charts
@return string which represents the title
*/
//std::string GetGnuplotTitle(const SystemInfo& s_info);
//
//#if defined YACCLAB_WITH_CUDA
//std::string GetGnuplotTitleGpu(const SystemInfo& s_info);
//#endif
std::string GetGnuplotTitle();
#if defined YACCLAB_WITH_CUDA
std::string GetGnuplotTitleGpu();
#endif

std::string EscapeUnderscore(const std::string& s);
std::string DoubleEscapeUnderscore(const std::string& s);

#endif // !YACCLAB_UTILITIES_H_
