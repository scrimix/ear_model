#include "helpers.h"

inline void pipeline_test(std::string input_file, std::string output_file, bool light_color_theme) {
  constexpr float kSampleRateHz = 44100.0f;
  constexpr int kChunkSize = 512;  // 11 ms.
  // constexpr int kNumChunks = 2048;
  int kNumChunks = kSampleRateHz / kChunkSize;

  auto file_size = std::filesystem::file_size(input_file);
  auto seconds = (file_size - 44) / kSampleRateHz / sizeof(int);
  std::cout << "file size? " << file_size << " " << seconds << std::endl;
  kNumChunks *= seconds;

  PitchogramPipelineParams params;
  params.num_frames = kNumChunks + 1;
  params.num_samples_per_segment = kChunkSize;
  params.pitchogram_params.light_color_theme = light_color_theme;
  PitchogramPipeline pipeline(kSampleRateHz, params);

  std::FILE* wav = std::fopen(input_file.c_str(), "rb");
  assert(wav != nullptr);
  // Seek past 44-byte header and one second into the recording.
  assert(std::fseek(wav, 44 + int(kSampleRateHz)*2, SEEK_CUR) == 0);

  cv::Mat nap_image;
  std::vector<cv::Mat> frames;

  for (int i = 0; i < kNumChunks; ++i) {
    // Read a chunk, converting int16 -> float and mixing stereo down to mono.
    float input[kChunkSize];
    for (int j = 0; j < kChunkSize; ++j) {
      input[j] = (static_cast<float>(ReadInt16Le(wav)) +
                  static_cast<float>(ReadInt16Le(wav))) / 65536.0f;
      input[j] /= 16;
    }
    std::cout << i << std::endl;

    pipeline.ProcessSamples(input, kChunkSize);

    // auto& nap = pipeline.carfac_output().nap()[0].transpose().eval();
    auto& nap = pipeline.sai_output().transpose().eval();
    cv::Mat mat(nap.rows(), nap.cols(), CV_32F, (void*)nap.data());
    cv::Mat rot_mat;
    cv::rotate(mat, rot_mat, cv::ROTATE_90_CLOCKWISE);
    cv::resize(rot_mat, rot_mat, cv::Size(1500, 1000));
    // rot_mat = mat;

    // frames.push_back(rot_mat.clone());

    // std::cout << frames.size() << std::endl;
    // if(frames.size() == 5){
    //   cv::Mat stack;
    //   stack = cv::Mat(frames[0].rows, frames[0].cols * frames.size(), CV_32F);
    //   stack = cv::Scalar(0);
    //   auto pos = 0;
    //   for(auto frame : frames){
    //     frame.copyTo(stack(cv::Rect(pos, 0, frame.cols, frame.rows)));
    //     pos += frame.cols * 0.2;
    //   }

    //   // cv::hconcat(frames, stack);
    //   cv::namedWindow("stack", 2);
    //   cv::imshow("stack", stack);
    //   cv::waitKey();
    //   frames.clear();
    // }

    if(i % 1 == 0){
      cv::namedWindow("nap", 2);
      cv::imshow("nap", rot_mat);
      cv::waitKey(11);
    }
  }

  std::fclose(wav);
  WritePnm(output_file, pipeline.image());
}

int main()
{
    std::string sound_dir = "../../sound_data";
    auto input_file = sound_dir + "/rnd.wav";
    // input_file = "/Users/scrimix/Music/lofi.wav";
    // auto input_file = sound_dir + "/test_output.wav";
    // auto input_file = "/Users/scrimix/goji.wav";
    // auto output_file = "../hello.pnm";
    auto output_file = "../test_output.pnm";
    bool light_theme = false;

    pipeline_test(input_file, output_file, light_theme);
}