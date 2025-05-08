#include <iostream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <sndfile.h>
#include <filesystem>

// Reads an int16 in little endian byte order from `f`.
int16_t ReadInt16Le(std::FILE* f) {
  uint16_t sample_u16 = static_cast<uint16_t>(std::getc(f));
  sample_u16 |= static_cast<uint16_t>(std::getc(f)) << 8;
  int16_t sample_i16;
  std::memcpy(&sample_i16, &sample_u16, sizeof(sample_i16));
  return sample_i16;
}

void computeSpectrogram(const std::vector<double>& signal, int windowSize, int hopSize, int fftSize) {
    int numSegments = (signal.size() - windowSize) / hopSize + 1;
    std::vector<std::vector<double>> spectrogram(numSegments, std::vector<double>(fftSize / 2 + 1));

    // Allocate FFTW input/output
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (fftSize / 2 + 1));
    double* in = (double*)fftw_malloc(sizeof(double) * fftSize);
    fftw_plan plan = fftw_plan_dft_r2c_1d(fftSize, in, out, FFTW_MEASURE);

    // Hanning window
    std::vector<double> window(windowSize);
    for (int i = 0; i < windowSize; ++i) {
        window[i] = 0.5 * (1 - cos(2 * M_PI * i / (windowSize - 1)));
    }

    // Process each segment
    for (int seg = 0; seg < numSegments; ++seg) {
        int start = seg * hopSize;

        // Apply window and zero-padding
        for (int i = 0; i < windowSize; ++i) {
            in[i] = signal[start + i] * window[i];
        }
        for (int i = windowSize; i < fftSize; ++i) {
            in[i] = 0.0;
        }

        // Execute FFT
        fftw_execute(plan);

        // Compute magnitude and store in spectrogram
        for (int k = 0; k < fftSize / 2 + 1; ++k) {
            spectrogram[seg][k] = sqrt(out[k][0] * out[k][0] + out[k][1] * out[k][1]);
        }
    }

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    // Convert spectrogram to OpenCV Mat
    cv::Mat spectrogramMat(numSegments, fftSize / 2 + 1, CV_64F);
    for (int i = 0; i < numSegments; ++i) {
        for (int j = 0; j < fftSize / 2 + 1; ++j) {
            spectrogramMat.at<double>(i, j) = spectrogram[i][j];
        }
    }

    // Normalize and visualize
    // cv::normalize(spectrogramMat, spectrogramMat, 0, 1, cv::NORM_MINMAX);
    spectrogramMat.convertTo(spectrogramMat, CV_8U, 255);
    cv::applyColorMap(spectrogramMat, spectrogramMat, cv::COLORMAP_JET);
    cv::rotate(spectrogramMat, spectrogramMat, cv::ROTATE_90_COUNTERCLOCKWISE);

    cv::imshow("Spectrogram", spectrogramMat);
    cv::waitKey(0);
}

int main() {
    // Example signal (sine wave + noise)
    // int sampleRate = 44100;
    // double duration = 2.0;
    // int totalSamples = sampleRate * duration;
    // std::vector<double> signal(totalSamples);

    // double freq = 440.0; // A4 tone
    // for (int i = 0; i < totalSamples; ++i) {
    //     signal[i] = sin(2 * M_PI * freq * i / sampleRate) + 0.5 * ((rand() / (double)RAND_MAX) - 0.5);
    // }

    // std::string input_file = "/Users/scrimix/goji.wav";
    std::string input_file = "../../sound_data/new_song.wav";
    std::FILE* wav = std::fopen(input_file.c_str(), "rb");
    assert(wav != nullptr);
    // Seek past 44-byte header and one second into the recording.
    assert(std::fseek(wav, 44 + int(kSampleRateHz)*2, SEEK_CUR) == 0);

    constexpr float kSampleRateHz = 44100.0f;
    constexpr int kChunkSize = 512;  // 11 ms.
    // constexpr int kNumChunks = 2048;
    int kNumChunks = kSampleRateHz / kChunkSize;
    
    auto file_size = std::filesystem::file_size(input_file);
    auto seconds = (file_size - 44) / kSampleRateHz / sizeof(int);
    std::cout << "file size? " << file_size << " " << seconds << std::endl;
    kNumChunks *= seconds;

    std::vector<double> signal;

    for (int i = 0; i < kNumChunks; ++i) {
        // Read a chunk, converting int16 -> float and mixing stereo down to mono.
        float input[kChunkSize];
        for (int j = 0; j < kChunkSize; ++j) {
            input[j] = (static_cast<float>(ReadInt16Le(wav)) +
                        static_cast<float>(ReadInt16Le(wav))) / 65536.0f;
        }
        for(auto j = 0; j < kChunkSize; j++){
            signal.push_back(input[j]);
        }
    }

    std::cout << signal.size() << std::endl;


    // SF_INFO sfInfo;
    // SNDFILE* sndFile = sf_open(input_file.c_str(), SFM_READ, &sfInfo);

    // if (!sndFile) {
    //     std::cerr << "Error opening WAV file: " << sf_strerror(sndFile) << std::endl;
    //     return -1;
    // }

    // // Check if the file is multi-channel
    // std::vector<double> signal(sfInfo.frames);
    // if (sfInfo.channels == 1) {
    //     // Mono
    //     sf_read_double(sndFile, signal.data(), sfInfo.frames);
    // } else {
    //     // Stereo or multi-channel: Convert to mono
    //     std::vector<double> multiChannelSignal(sfInfo.frames * sfInfo.channels);
    //     sf_read_double(sndFile, multiChannelSignal.data(), sfInfo.frames * sfInfo.channels);
    //     for (int i = 0; i < sfInfo.frames; ++i) {
    //         double sum = 0.0;
    //         for (int ch = 0; ch < sfInfo.channels; ++ch) {
    //             sum += multiChannelSignal[i * sfInfo.channels + ch];
    //         }
    //         signal[i] = sum / sfInfo.channels; // Average the channels
    //     }
    // }
    // sf_close(sndFile);

    // signal.resize(44100 * 8);

    std::cout << "siglans?? " << signal.size() << std::endl;

    int windowSize = 4096;
    int hopSize = 128;
    int fftSize = 4096;

    computeSpectrogram(signal, windowSize, hopSize, fftSize);

    return 0;
}
