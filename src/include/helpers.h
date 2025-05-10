#pragma once

#include <carfac/carfac.h>
#include <carfac/pitchogram_pipeline.h>
#include <carfac/image.h>
#include <iostream>
#include <filesystem>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <htm/algorithms/SpatialPooler.hpp>

namespace fs = std::filesystem;

// Views a 4-channel RGBA image as a 3-channel RGB image.
inline Image<const uint8_t> RgbaToRgbImage(const Image<const uint8_t>& image_rgba) {
  return Image<const uint8_t>(image_rgba.data(), image_rgba.width(),
                              image_rgba.x_stride(), image_rgba.height(),
                              image_rgba.y_stride(), 3, image_rgba.c_stride());
}

// Returns true if string `s` starts with `prefix`.
inline bool StartsWith(const char* s, const char* prefix) {
  while (*prefix) {
    if (*s++ != *prefix++) { return false; }
  }
  return true;
}

struct TestParams {
  bool light_color_theme;

  std::string name() const {
    return std::string("pitchogram_pipeline_test-") +
          (light_color_theme ? "light" : "dark") + "_theme";
  }
};

inline std::string PrintToString(const TestParams& params) { return params.name(); }

inline std::vector<fs::path> list_wav_files(const fs::path& directory) {
    std::vector<fs::path> wav_files;

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return wav_files; // empty if invalid directory
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file())
            continue;

        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (ext == ".wav") {
            wav_files.push_back(entry.path());
        }
    }

    return wav_files;
}

// for string delimiter
inline std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

inline std::string replaced(std::string const& input, const std::string &search, const std::string &replace) {
    auto s = input;
    for( size_t pos = 0; ; pos += replace.length() ) {
        // Locate the substring to replace
        pos = s.find( search, pos );
        if( pos == std::string::npos ) break;
        // Replace by erasing and inserting
        s.erase( pos, search.length() );
        s.insert( pos, replace );
    }
    return s;
}

inline void exit_on_ctrl_c(int a)
{
    std::exit(1);
}

inline cv::Mat sdr3DToColorMap(const htm::SDR &sdr) {
  auto d = sdr.dimensions;             // {H, W, C}
  int H = d[0], W = d[1], C = d[2];
  int plane = H*W;

  // accumulator for sum of slice indices and count per pixel
  cv::Mat sumIdx(H, W, CV_32FC1, cv::Scalar(0));
  cv::Mat count(H,  W, CV_32FC1, cv::Scalar(0));

  for (auto idx : sdr.getSparse()) {
    int sl = idx / plane;
    int rem = idx % plane;
    int r = rem / W, c = rem % W;
    sumIdx.at<float>(r,c) += float(sl);
    count.at<float>(r,c)  += 1.0f;
  }

  // build normalized slice image in 0–255
  cv::Mat norm(H, W, CV_8UC1);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float cnt = count.at<float>(y,x);
      if (cnt > 0.0f) {
        float avg = sumIdx.at<float>(y,x) / cnt;
        // scale 0…C-1 to 0…255
        norm.at<uint8_t>(y,x) = uint8_t((avg / float(C-1)) * 255.0f);
      } else {
        norm.at<uint8_t>(y,x) = 0;
      }
    }
  }

  // apply a colorful colormap (e.g. COLORMAP_JET)
  cv::Mat color;
  cv::applyColorMap(norm, color, cv::COLORMAP_JET);
  return color;
}

inline std::vector<char> mat_to_vector(cv::Mat img)
{
  // allocate vector large enough to hold all bytes
  size_t nBytes = img.total() * img.elemSize();  
  std::vector<char> buf(nBytes);
  // copy raw data
  std::memcpy(buf.data(), img.data, nBytes);
  return buf;
}

inline std::string read_text_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Unable to open " + path);

  // find length
  in.seekg(0, std::ios::end);
  std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);

  // read it all
  std::string buf(size, '\0');
  if (!in.read(&buf[0], size))
      throw std::runtime_error("Error reading " + path);

  return buf;
}
