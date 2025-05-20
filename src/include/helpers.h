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


inline std::vector<fs::path> list_audio_files(const fs::path& directory) {
    std::vector<fs::path> audio_files;

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return audio_files; // return empty if invalid directory
    }

    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (!entry.is_regular_file())
            continue;

        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (ext == ".wav" || ext == ".midi" || ext == ".mid") {
            audio_files.push_back(entry.path());
        }
    }

    return audio_files;
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

inline cv::Mat vector_to_mat(const std::vector<uint8_t>& vec, int rows, int cols) {
    cv::Mat mat(rows, cols, CV_8U);
    mat = cv::Scalar(0);
    for (int i = 0; i < std::min((int)vec.size(), rows * cols); ++i) {
        int r = i / cols, c = i % cols;
        mat.at<uchar>(r, c) = vec[i];
    }
    return mat;
}

inline htm::SDR reshape_sdr(const htm::SDR& flat, std::vector<uint> dims) {
    htm::SDR reshaped(dims);
    reshaped.setSparse(flat.getSparse()); // or use .setDense() if needed
    return reshaped;
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

inline cv::Mat sdr1DToColorMap(const htm::SDR &sdr)
{
  int N = 1;
  for (auto d : sdr.dimensions)
    N *= d;

  const int rows = static_cast<int>(std::floor(std::sqrt(N)));
  const int cols = static_cast<int>(std::ceil(static_cast<double>(N) / rows));

  cv::Mat mat(rows, cols, CV_8UC1, cv::Scalar(0));

  for (int bit : sdr.getSparse()) {
    int r = bit / cols;
    int c = bit % cols;
    if (r < rows && c < cols)
      mat.at<uint8_t>(r, c) = 255;
  }

  // Optional: colorize for nicer display
  cv::Mat color;
  cv::applyColorMap(mat, color, cv::COLORMAP_JET);
  return color;
}

inline cv::Mat sdr1DToColorMapBySlice(const htm::SDR &sdr, int cellsPerColumn = 32)
{
    int N = 1;
    for (auto d : sdr.dimensions)
      N *= d;

    const int rows = static_cast<int>(std::floor(std::sqrt(N)));
    const int cols = static_cast<int>(std::ceil(static_cast<double>(N) / rows));

    cv::Mat sumIdx(rows, cols, CV_32FC1, cv::Scalar(0));
    cv::Mat count (rows, cols, CV_32FC1, cv::Scalar(0));

    for (int bit : sdr.getSparse()) {
        int row = bit / cols;
        int col = bit % cols;

        int slice = bit % cellsPerColumn;  // Which cell in the column
        sumIdx.at<float>(row, col) += float(slice);
        count.at<float>(row, col) += 1.0f;
    }

    cv::Mat norm(rows, cols, CV_8UC1);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float cnt = count.at<float>(r, c);
            if (cnt > 0.0f) {
                float avg = sumIdx.at<float>(r, c) / cnt;
                norm.at<uint8_t>(r, c) = uint8_t((avg / float(cellsPerColumn - 1)) * 255.0f);
            } else {
                norm.at<uint8_t>(r, c) = 0;
            }
        }
    }

    cv::Mat color;
    cv::applyColorMap(norm, color, cv::COLORMAP_JET);
    return color;
}

inline cv::Mat draw_sp_output(const htm::SDR& columns, int H = 32, int W = 32, int note_size = 256) {
    int image_size = H * W;

    cv::Mat sai_img(H, W, CV_8U, cv::Scalar(0));
    cv::Mat note_img(note_size, 1, CV_8U, cv::Scalar(0));

    for (auto idx : columns.getSparse()) {
        if (idx < image_size) {
            int r = idx / W;
            int c = idx % W;
            sai_img.at<uchar>(r, c) = 255;
        } else if (idx < image_size + note_size) {
            int i = idx - image_size;
            note_img.at<uchar>(i, 0) = 255;
        }
    }

    // Resize note SDR to match image height visually
    cv::Mat note_resized;
    cv::resize(note_img, note_resized, cv::Size(8, H), 0, 0, cv::INTER_NEAREST);

    // Concatenate horizontally: [SAI | Note SDR]
    cv::Mat combined;
    cv::hconcat(sai_img, note_resized, combined);

    // Optionally add color map
    cv::Mat color;
    cv::applyColorMap(combined, color, cv::COLORMAP_JET);
    return color;
}

inline std::vector<uint8_t> mat_to_vector(cv::Mat img)
{
  // allocate vector large enough to hold all bytes
  size_t nBytes = img.total() * img.elemSize();  
  std::vector<uint8_t> buf(nBytes);
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

inline void write_text_to_file(std::string const& path, std::string const& content)
{
    std::ofstream file;
    file.open(path);
    file << content;
    file.close();
}

// Return indices of top-n largest elements in a vector
inline std::vector<size_t> topNIndices(const std::vector<double>& values, size_t n) {
    // Min-heap to store the top n elements (value, index)
    using Pair = std::pair<float, size_t>;
    auto cmp = [](const Pair& a, const Pair& b) { return a.first > b.first; };
    std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)> minHeap(cmp);

    for (size_t i = 0; i < values.size(); ++i) {
        if (minHeap.size() < n) {
            minHeap.emplace(values[i], i);
        } else if (values[i] > minHeap.top().first) {
            minHeap.pop();
            minHeap.emplace(values[i], i);
        }
    }

    // Extract indices
    std::vector<size_t> result;
    while (!minHeap.empty()) {
        result.push_back(minHeap.top().second);
        minHeap.pop();
    }

    // Optional: sort indices by descending value
    std::sort(result.begin(), result.end(), [&values](size_t a, size_t b) {
        return values[a] > values[b];
    });

    return result;
}


inline int random_midi_note(int min = 21, int max = 108) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(min, max); // MIDI: 21 (A0) to 108 (C8)
    return dist(gen);
}

inline void show(std::string title, cv::Mat image)
{
    cv::namedWindow(title, 2);
    cv::imshow(title, image);
}

template<typename T>
void concat(std::vector<T>* dst, const std::vector<T>& src) {
    if (!dst) return;
    dst->insert(dst->end(), src.begin(), src.end());
}

inline std::pair<int, int> square_ish_sdr(int size)
{
    int rows = static_cast<int>(std::floor(std::sqrt(size)));
    int cols = static_cast<int>(std::ceil(static_cast<double>(size) / rows));
    return {rows, cols};
}

template <typename T>
std::vector<T> remove_zero(std::vector<T> input)
{
    input.erase(std::remove(input.begin(), input.end(), 0), input.end());
    return input;
}

template <typename Model>
bool load_model_with_check(Model& model, std::string const& file_path){
    if(!std::filesystem::exists(file_path)){
      std::cerr << "loading model: " << file_path << " failed! File doesn't exist";
      return false;
    }
    std::ifstream in(file_path, std::ios_base::in | std::ios_base::binary);
    cereal::BinaryInputArchive iarchive(in);
    model.load_ar(iarchive);
    in.close();
    return true;
}
