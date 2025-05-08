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

// Reads an int16 in little endian byte order from `f`.
inline int16_t ReadInt16Le(std::FILE* f) {
  uint16_t sample_u16 = static_cast<uint16_t>(std::getc(f));
  sample_u16 |= static_cast<uint16_t>(std::getc(f)) << 8;
  int16_t sample_i16;
  std::memcpy(&sample_i16, &sample_u16, sizeof(sample_i16));
  return sample_i16;
}

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

/// Convert a 3-D SDR (H×W×C) into an RGB image by grouping the C slices
/// into 3 equal bins (R, G, B).  Any active bit in a bin lights that color.
inline cv::Mat sdr3DToRGB(const htm::SDR &sdr) {
  auto dims = sdr.dimensions;
  if (dims.size() != 3) {
    throw std::runtime_error("sdr3DToRGB only supports 3-D SDRs");
  }
  int H = dims[0], W = dims[1], C = dims[2];
  int planeSize = H * W;

  // three single-channel mats, initially zero
  cv::Mat R(H, W, CV_8UC1, cv::Scalar(0));
  cv::Mat G(H, W, CV_8UC1, cv::Scalar(0));
  cv::Mat B(H, W, CV_8UC1, cv::Scalar(0));

  // for each active bit
  for (auto idx : sdr.getSparse()) {
    int slice = idx / planeSize;           // which depth slice
    int rem   = idx % planeSize;
    int r     = rem / W, c = rem % W;

    // assign slice → color bin
    int bin = (slice * 3) / C;             // 0,1,2
    switch(bin) {
      case 0: R.at<uint8_t>(r,c) = 255; break;
      case 1: G.at<uint8_t>(r,c) = 255; break;
      case 2: B.at<uint8_t>(r,c) = 255; break;
    }
  }

  // merge into one 3-channel image
  std::vector<cv::Mat> chans = { B, G, R };
  cv::Mat rgb;
  cv::merge(chans, rgb);
  return rgb;
}

/// Convert a 2-D SDR into a CV_8UC1 image (white where bits are 1).
inline cv::Mat sdrToMat(const htm::SDR &sdr) {
  // expect exactly 2 dimensions
  auto dims = sdr.dimensions;
  if (dims.size() != 2) {
    throw std::runtime_error("sdrToMat only supports 2-D SDRs");
  }
  int H = dims[0], W = dims[1];

  // create a black image
  cv::Mat img(H, W, CV_8UC1, cv::Scalar(0));

  // get the list of active (flattened) indices
  auto active = sdr.getSparse();
  for (auto idx : active) {
    int r = idx / W;
    int c = idx % W;
    // safety check
    if (r >= 0 && r < H && c >= 0 && c < W) {
      img.at<uint8_t>(r, c) = 255;
    }
  }
  return img;
}

inline cv::Mat sdr3DToMat(const htm::SDR &sdr) {
  auto d = sdr.dimensions; // {H,W,C}
  int H=d[0], W=d[1], C=d[2];
  cv::Mat img(H, W, CV_8UC1, cv::Scalar(0));
  for (auto idx : sdr.getSparse()) {
    int slice = idx / (H*W);
    int rem   = idx % (H*W);
    int r = rem / W, c = rem % W;
    img.at<uint8_t>(r,c) = 255/C;  // or += 255/C if you want intensity
  }
  return img;
}

/// Convert a 3-D SDR (H×W×C) into an RGB image by grouping the C slices
/// into 3 bins (R,G,B), summing activations per pixel per bin, then
/// normalizing each bin’s count to the 0–255 range.
inline cv::Mat sdr3DToRGBSummed(const htm::SDR &sdr) {
  auto dims = sdr.dimensions;
  if (dims.size() != 3) {
    throw std::runtime_error("sdr3DToRGBSummed only supports 3-D SDRs");
  }
  int H = dims[0], W = dims[1], C = dims[2];
  int planeSize = H * W;
  // how many slices map into each bin? (last bin may get the remainder)
  int baseBinSize = C / 3;
  int rem = C % 3;

  // accumulator mats (32-bit ints so we don’t overflow)
  cv::Mat accR(H, W, CV_32SC1, cv::Scalar(0));
  cv::Mat accG(H, W, CV_32SC1, cv::Scalar(0));
  cv::Mat accB(H, W, CV_32SC1, cv::Scalar(0));

  // accumulate
  for (auto idx : sdr.getSparse()) {
    int slice = idx / planeSize;           // which depth slice
    int remIdx = idx % planeSize;
    int r = remIdx / W, c = remIdx % W;

    // determine which of the 3 bins this slice belongs to
    int threshold1 = baseBinSize;
    int threshold2 = baseBinSize * 2 + rem;  // give extra slices to last bin

    if (slice < threshold1) {
      accR.at<int>(r, c)++;
    }
    else if (slice < threshold2) {
      accG.at<int>(r, c)++;
    }
    else {
      accB.at<int>(r, c)++;
    }
  }

  // now normalize each accumulator to [0,255]
  // compute the max possible count per bin:
  int maxR = baseBinSize;
  int maxG = baseBinSize;
  int maxB = baseBinSize + rem;

  cv::Mat R(H, W, CV_8UC1), G(H, W, CV_8UC1), B(H, W, CV_8UC1);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      // clamp and scale each
      int vR = accR.at<int>(y, x);
      int vG = accG.at<int>(y, x);
      int vB = accB.at<int>(y, x);
      R.at<uint8_t>(y,x) = static_cast<uint8_t>(std::min(255, (vR * 255) / std::max(1, maxR)));
      G.at<uint8_t>(y,x) = static_cast<uint8_t>(std::min(255, (vG * 255) / std::max(1, maxG)));
      B.at<uint8_t>(y,x) = static_cast<uint8_t>(std::min(255, (vB * 255) / std::max(1, maxB)));
    }
  }

  // merge into BGR image
  std::vector<cv::Mat> chans = { B, G, R };
  cv::Mat rgb;
  cv::merge(chans, rgb);
  return rgb;
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
