#pragma once

#include <opencv2/core.hpp>

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <random>

inline std::vector<cv::Rect> splitImageWithOverlap(int imgWidth, int imgHeight, int numRegions, int overlapPx) {
    std::vector<cv::Rect> regions;

    // Determine grid layout (rows × cols) close to square
    int rows = std::sqrt(numRegions);
    int cols = (numRegions + rows - 1) / rows;

    // Base region size (without overlap)
    int baseW = imgWidth / cols;
    int baseH = imgHeight / rows;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (regions.size() >= numRegions) break;

            int x0 = c * baseW - overlapPx;
            int y0 = r * baseH - overlapPx;
            int x1 = (c + 1) * baseW + overlapPx;
            int y1 = (r + 1) * baseH + overlapPx;

            // Clamp to image boundaries
            x0 = std::max(0, x0);
            y0 = std::max(0, y0);
            x1 = std::min(imgWidth, x1);
            y1 = std::min(imgHeight, y1);

            int w = x1 - x0;
            int h = y1 - y0;

            regions.emplace_back(cv::Rect(x0, y0, w, h));
        }
    }

    return regions;
}


inline cv::Mat tileImages(const std::vector<cv::Mat>& images, int tileCols, int padding = 2, cv::Scalar padColor = {0, 0, 0}) {
    if (images.empty()) return cv::Mat();

    // Reference size and type from first image
    int imgW = images[0].cols;
    int imgH = images[0].rows;
    int targetType = CV_8UC3;
    int targetChannels = 3;

    std::vector<cv::Mat> processed;
    for (const auto& img : images) {
        cv::Mat tmp;

        // Convert type if needed
        if (img.type() != targetType) {
            if (targetChannels == 3 && img.channels() == 1)
                cv::cvtColor(img, tmp, cv::COLOR_GRAY2BGR);
            else if (targetChannels == 1 && img.channels() == 3)
                cv::cvtColor(img, tmp, cv::COLOR_BGR2GRAY);
            else
                img.convertTo(tmp, targetType);  // fallback

        } else {
            tmp = img;
        }

        // Resize if needed
        if (tmp.cols != imgW || tmp.rows != imgH)
            cv::resize(tmp, tmp, cv::Size(imgW, imgH));

        processed.push_back(tmp);
    }

    // Determine tile layout
    int tileRows = (processed.size() + tileCols - 1) / tileCols;
    int canvasW = tileCols * imgW + (tileCols - 1) * padding;
    int canvasH = tileRows * imgH + (tileRows - 1) * padding;

    cv::Mat canvas(canvasH, canvasW, targetType, padColor);

    for (size_t i = 0; i < processed.size(); ++i) {
        int row = i / tileCols;
        int col = i % tileCols;

        int x = col * (imgW + padding);
        int y = row * (imgH + padding);

        cv::Rect roi(x, y, imgW, imgH);
        processed[i].copyTo(canvas(roi));
    }

    return canvas;
}

inline std::vector<cv::Rect> generateFoveatedRegions(cv::Size imgSize, int numPetals = 8, int foveaSize = 200, int overlap = 10) {
    std::vector<cv::Rect> regions;

    // 1. Central fovea (centered square or rectangle)
    int centerX = imgSize.width / 2;
    int centerY = imgSize.height / 2;
    int halfFovea = foveaSize / 2;

    cv::Rect fovea(
        std::max(0, centerX - halfFovea),
        std::max(0, centerY - halfFovea),
        std::min(foveaSize, imgSize.width),
        std::min(foveaSize, imgSize.height)
    );
    regions.push_back(fovea);

    // 2. Petal regions around the fovea
    int petalW = foveaSize;
    int petalH = foveaSize;

    // Coordinates for 8 petals (top, bottom, left, right, and corners)
    std::vector<cv::Point> offsets = {
        {0, -1}, {0, 1}, {-1, 0}, {1, 0},  // top, bottom, left, right
        {-1, -1}, {1, -1}, {-1, 1}, {1, 1} // corners
    };

    for (int i = 0; i < std::min(numPetals, 8); ++i) {
        int dx = offsets[i].x;
        int dy = offsets[i].y;

        int x = centerX + dx * petalW - halfFovea - overlap;
        int y = centerY + dy * petalH - halfFovea - overlap;
        int w = foveaSize + 2 * overlap;
        int h = foveaSize + 2 * overlap;

        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, imgSize.width - x);
        h = std::min(h, imgSize.height - y);

        regions.emplace_back(cv::Rect(x, y, w, h));
    }

    return regions;
}


inline std::vector<cv::Rect> generateSpiralRegions(cv::Size imgSize, int numRegions, int patchSize = 64) {
    std::vector<cv::Rect> regions;

    float cx = imgSize.width / 2.0f;
    float cy = imgSize.height / 2.0f;

    float maxRadius = std::min(cx, cy) - patchSize / 2.0f;
    float maxTheta = 4 * M_PI;  // controls number of turns in the spiral

    for (int i = 0; i < numRegions; ++i) {
        float t = float(i) / std::max(1, numRegions - 1);  // normalized [0, 1]
        float theta = t * maxTheta;
        float r = t * maxRadius;

        float x = cx + r * std::cos(theta);
        float y = cy + r * std::sin(theta);

        int x0 = std::round(x - patchSize / 2.);
        int y0 = std::round(y - patchSize / 2.);

        x0 = std::max(0, std::min(imgSize.width - patchSize, x0));
        y0 = std::max(0, std::min(imgSize.height - patchSize, y0));

        regions.emplace_back(cv::Rect(x0, y0, patchSize, patchSize));
    }

    return regions;
}

inline std::vector<cv::Rect> reg_setup_1()
{
  std::vector<cv::Rect> regions;
  regions.push_back(cv::Rect(100, 0, 600, 200));

  regions.push_back(cv::Rect(0, 100, 200, 200));
  regions.push_back(cv::Rect(200, 200, 200, 200));
  regions.push_back(cv::Rect(400, 300, 200, 200));
  regions.push_back(cv::Rect(600, 250, 200, 200));

  regions.push_back(cv::Rect(100, 250, 300, 100));
  regions.push_back(cv::Rect(300, 200, 300, 100));

  regions.push_back(cv::Rect(400, 100, 100, 300));
  regions.push_back(cv::Rect(100, 200, 100, 300));

  regions.push_back(cv::Rect(100, 150, 150, 150));
  regions.push_back(cv::Rect(300, 300, 150, 150));
  regions.push_back(cv::Rect(500, 400, 150, 150));
  regions.push_back(cv::Rect(600, 300, 150, 150));

  regions.push_back(cv::Rect(100, 400, 600, 200));
  return regions;
}

inline std::vector<cv::Rect> reg_setup_2()
{
  std::vector<cv::Rect> regions;
  regions.push_back(cv::Rect(100, 0, 600, 200));

  regions.push_back(cv::Rect(100, 250, 300, 100));
  regions.push_back(cv::Rect(300, 200, 300, 100));

  regions.push_back(cv::Rect(150, 100, 300, 100));
  regions.push_back(cv::Rect(350, 350, 300, 100));

  regions.push_back(cv::Rect(400, 100, 100, 300));
  regions.push_back(cv::Rect(100, 200, 100, 300));

  regions.push_back(cv::Rect(500, 150, 100, 300));
  regions.push_back(cv::Rect(150, 250, 100, 300));
  regions.push_back(cv::Rect(200, 200, 100, 300));
  regions.push_back(cv::Rect(650, 100, 100, 300));

  regions.push_back(cv::Rect(100, 400, 600, 200));
  return regions;
}

inline std::vector<cv::Rect> generateMidFocusedBands(cv::Size imgSize, int bandHeight = 96, int overlap = 16, int numTimeSlices = 1, int patchWidth = 64) {
    std::vector<cv::Rect> regions;

    int H = imgSize.height;
    int W = imgSize.width;

    // Define band centers (based on middle emphasis)
    std::vector<int> bandCenters = {
        H / 2 - bandHeight - overlap, // lower-middle
        H / 2,                        // center
        H / 2 + bandHeight + overlap  // upper-middle
    };

    // Optionally add low and high extreme bands
    int lowBand = H / 2 - 2 * bandHeight;
    int highBand = H / 2 + 2 * bandHeight;
    if (lowBand > 0) bandCenters.insert(bandCenters.begin(), lowBand);
    if (highBand + bandHeight < H) bandCenters.push_back(highBand);

    // For each band and time slice, create a region
    int timeStep = (W - patchWidth) / std::max(1, numTimeSlices - 1);

    for (int t = 0; t < numTimeSlices; ++t) {
        int x = t * timeStep;

        for (int yc : bandCenters) {
            int y0 = std::max(0, yc - bandHeight / 2);
            int h = std::min(bandHeight, H - y0);
            regions.emplace_back(cv::Rect(x, y0, patchWidth, h));
        }
    }

    return regions;
}

inline std::vector<cv::Rect> generateOffsetMidBands(
    cv::Size imgSize,
    int baseBandHeight = 96,
    int basePatchWidth = 64,
    int numTimeSlices = 4,
    int overlap = 16,
    int bandJitter = 12,
    int sizeJitter = 8,
    unsigned int seed = 42
) {
    std::vector<cv::Rect> regions;

    int H = imgSize.height;
    int W = imgSize.width;

    // Setup reproducible RNG
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> jitterDist(-bandJitter, bandJitter);
    std::uniform_int_distribution<int> sizeDist(-sizeJitter, sizeJitter);

    // Define center frequency bands
    std::vector<int> baseCenters = {
        H / 2 - baseBandHeight - overlap, // lower-mid
        H / 2,                            // mid
        H / 2 + baseBandHeight + overlap  // upper-mid
    };

    // Optionally add low and high extremes
    int low = H / 2 - 2 * baseBandHeight;
    int high = H / 2 + 2 * baseBandHeight;
    if (low > 0) baseCenters.insert(baseCenters.begin(), low);
    if (high + baseBandHeight < H) baseCenters.push_back(high);

    // Time step for slicing
    int timeStep = (W - basePatchWidth) / std::max(1, numTimeSlices - 1);

    for (int t = 0; t < numTimeSlices; ++t) {
        int x = t * timeStep;

        for (size_t i = 0; i < baseCenters.size(); ++i) {
            int jitterY = jitterDist(rng);
            int jitterH = sizeDist(rng);
            int jitterW = sizeDist(rng);

            int centerY = baseCenters[i] + jitterY;
            int patchH = std::clamp(baseBandHeight + jitterH, 32, H);
            int patchW = std::clamp(basePatchWidth + jitterW, 32, W / 2);

            int y0 = std::clamp(centerY - patchH / 2, 0, H - patchH);
            int x0 = std::clamp(x, 0, W - patchW);

            regions.emplace_back(cv::Rect(x0, y0, patchW, patchH));
        }
    }

    return regions;
}

inline std::vector<cv::Rect> createLogFrequencyBands(cv::Size imgSize, int numBands, int patchTimeSize = 32, int overlap = 4) {
    std::vector<cv::Rect> regions;

    int timeSteps = imgSize.width;
    int freqSteps = imgSize.height;

    float minF = 1.0f;
    float maxF = std::log2(freqSteps + 1.0f);

    float step = (maxF - minF) / numBands;

    for (int i = 0; i < numBands; ++i) {
        float f0 = std::pow(2.0f, minF + i * step) - 1.0f;
        float f1 = std::pow(2.0f, minF + (i + 1) * step) - 1.0f;

        int y0 = std::max(0, int(f0) - overlap);
        int y1 = std::min(freqSteps, int(f1) + overlap);
        int h = y1 - y0;

        cv::Rect r(0, y0, patchTimeSize, h);  // patchTimeSize can be moved across time
        regions.push_back(r);
    }

    return regions;
}

inline std::vector<cv::Rect> basic_regions()
{
    std::vector<cv::Rect> result;
    result.push_back(cv::Rect(96, 41, 147, 526));
    result.push_back(cv::Rect(314, 234, 375, 174));
    result.push_back(cv::Rect(512, 249, 200, 253));
    result.push_back(cv::Rect(503, 69, 100, 412));
    result.push_back(cv::Rect(114, 140, 341, 138));
    return result;
}

inline std::vector<cv::Rect> more_regions()
{
    std::vector<cv::Rect> result;
    result.push_back(cv::Rect(410, 142, 318, 159));
    result.push_back(cv::Rect(300, 31, 233, 492));
    result.push_back(cv::Rect(132, 298, 563, 253));
    result.push_back(cv::Rect(136, 47, 526, 203));
    return result;
}

inline std::vector<cv::Rect> top_bottom_regions()
{
    std::vector<cv::Rect> result;
    result.push_back(cv::Rect(96, 74, 147, 526));
    result.push_back(cv::Rect(314, 234, 375, 174));
    result.push_back(cv::Rect(512, 249, 200, 253));
    result.push_back(cv::Rect(503, 69, 100, 412));
    result.push_back(cv::Rect(114, 140, 341, 138));

    result.push_back(cv::Rect(410, 142, 318, 159));
    result.push_back(cv::Rect(300, 31, 233, 492));
    result.push_back(cv::Rect(132, 347, 563, 253));
    result.push_back(cv::Rect(136, 47, 526, 203));
    return result;
}

inline std::vector<cv::Rect> generate_diagonal_regions(
    int num_regions,
    const cv::Size& image_size,
    const cv::Size& default_size = cv::Size(200, 100),
    float center_bias = 1.0f,               // 1 = full diagonal, 0 = only center
    bool bottom_left_to_top_right = true,
    int offset_jitter = 15,
    int size_jitter = 10,
    int seed = 42 
) {
    std::vector<cv::Rect> regions;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> jitter_offset(-offset_jitter, offset_jitter);
    std::uniform_int_distribution<int> jitter_size(-size_jitter, size_jitter);

    // Clamp bias
    center_bias = std::clamp(center_bias, 0.0f, 1.0f);

    // Diagonal start/end
    cv::Point2f start, end;
    if (bottom_left_to_top_right) {
        start = cv::Point2f(0.0f, static_cast<float>(image_size.height - default_size.height));
        end   = cv::Point2f(static_cast<float>(image_size.width - default_size.width), 0.0f);
    } else {
        start = cv::Point2f(0.0f, 0.0f);
        end   = cv::Point2f(static_cast<float>(image_size.width - default_size.width),
                           static_cast<float>(image_size.height - default_size.height));
    }

    // Active range of t around center
    float t_min = 0.5f - 0.5f * center_bias;
    float t_max = 0.5f + 0.5f * center_bias;

    for (int i = 0; i < num_regions; ++i) {
        float t = static_cast<float>(i) / std::max(1, num_regions - 1);
        float warped_t = t_min + t * (t_max - t_min);  // shrink t around center

        cv::Point2f base_pos = start + (end - start) * warped_t;

        int x = static_cast<int>(std::round(base_pos.x)) + jitter_offset(rng);
        int y = static_cast<int>(std::round(base_pos.y)) + jitter_offset(rng);

        int w = default_size.width + jitter_size(rng);
        int h = default_size.height + jitter_size(rng);

        x = std::clamp(x, 0, image_size.width - 1);
        y = std::clamp(y, 0, image_size.height - 1);
        w = std::min(w, image_size.width - x);
        h = std::min(h, image_size.height - y);

        regions.emplace_back(cv::Rect(x, y, w, h));
    }

    return regions;
}
