#pragma once

#include <cstdint> //uint8_t
#include <iostream>
#include <fstream>      // std::ofstream
#include <vector>

#include <htm/algorithms/SpatialPooler.hpp>
#include <htm/algorithms/SDRClassifier.hpp>
#include <htm/algorithms/TemporalMemory.hpp>
#include <htm/utils/SdrMetrics.hpp>
#include <htm/os/Timer.hpp>

#include "carfac_reader.h"
#include "helpers.h"

using namespace std;
using namespace htm;

struct note_model_params_t {
  // path to models
  std::string models_path;

  // htm
  uint dim = 32;
  uint column_count = 8;
  int pot_radius = 8;
  int tm_cell_per_column = 6;
  int binary_thresh = 40;
  float train_noise = 0.1;
  
  // carfac
  float loudness_coef = 40;
  int sample_rate = 44100;
  int buffer_size = 1024;
};

class note_model_t {
public:
  SpatialPooler sp;
  TemporalMemory tm;
  SDR input;
  SDR columns;
  SDR outTM;
  Classifier clsr;

  carfac_reader_t carfac_reader;
  AudioData audio;

  note_model_params_t params;

  void setup(note_model_params_t model_params) {
    params = model_params;
    input.initialize({params.dim, params.dim, 1});
    columns.initialize({params.dim, params.dim, params.column_count}); //1D vs 2D no big difference, 2D seems more natural for the problem. Speed-----, Results+++++++++; #columns HIGHEST impact. 
    sp.initialize(
      /* inputDimensions */             input.dimensions,
      /* columnDimensions */            columns.dimensions,
      /* potentialRadius */             params.pot_radius, // with 2D, 7 results in 15x15 area, which is cca 25% for the input area. Slightly improves than 99999 aka "no topology, all to all connections"
      /* potentialPct */                0.1f, //we have only 10 classes, and << #columns. So we want to force each col to specialize. Cca 0.3 w "7" above, or very small (0.1) for "no topology". Cannot be too small due to internal checks. Speed++
      /* globalInhibition */            true, //Speed+++++++; SDR quality-- (global does have active nearby cols, which we want to avoid (local)); Results+-0
      /* localAreaDensity */            0.02f,  // % active bits
      /* numActiveColumnsPerInhArea */  0,
      /* stimulusThreshold */           6u,
      /* synPermInactiveDec */          0.002f, //FIXME inactive decay permanence plays NO role, investigate! (slightly better w/o it)
      /* synPermActiveInc */            0.14f, //takes upto 5x steps to get dis/connected
      /* synPermConnected */            0.5f, //no difference, let's leave at 0.5 in the middle
      /* minPctOverlapDutyCycles */     0.2f, //speed of re-learning?
      /* dutyCyclePeriod */             1402,
      /* boostStrength */               1.0f, // Boosting does help, but entropy is high, on MNIST it does not matter, for learning with TM prefer boosting off (=0.0), or "neutral"=1.0
      /* seed */                        4u,
      /* spVerbosity */                 1u,
      /* wrapAround */                  true); // does not matter (helps slightly)

    tm.initialize(columns.dimensions, params.tm_cell_per_column, 13, 0.21, 0.5, 10, 20, 0.1, 0.1, 0, 42, 50, 50);
    clsr.initialize( /* alpha */ 0.001f);
  }

  void load_audio_file(std::string file_path)
  {
    carfac_reader.reset();
    carfac_reader.init(file_path);
    carfac_reader.set(params.sample_rate, params.buffer_size, params.loudness_coef);
    audio.buffer = readWavFile(file_path);
  }

  void load_audio(std::vector<float> const& wav)
  {
    carfac_reader.reset();
    carfac_reader.init(wav);
    carfac_reader.set(params.sample_rate, params.buffer_size, params.loudness_coef);
    audio.buffer = wav;
  }

  void save(std::string model_name)
  {
    // Save the model
    ofstream dump(model_name+"_sp.model", ofstream::binary | ofstream::trunc | ofstream::out);
    cereal::BinaryOutputArchive oarchive(dump);
    sp.save_ar(oarchive);
    dump.close();

    ofstream dump2(model_name+"_clsr.model", ofstream::binary | ofstream::trunc | ofstream::out);
    cereal::BinaryOutputArchive oarchive2(dump2);
    clsr.save_ar(oarchive2);
    dump2.close();

    ofstream dump3(model_name+"_tm.model", ofstream::binary | ofstream::trunc | ofstream::out);
    cereal::BinaryOutputArchive oarchive3(dump3);
    tm.save_ar(oarchive3);
    dump3.close();
  }

  void load(std::string model_name)
  {
    std::ifstream in(model_name+"_sp.model", std::ios_base::in | std::ios_base::binary);
    cereal::BinaryInputArchive iarchive(in);
    sp.load_ar(iarchive);
    in.close();

    std::ifstream in2(model_name+"_clsr.model", std::ios_base::in | std::ios_base::binary);
    cereal::BinaryInputArchive iarchive2(in2);
    clsr.load_ar(iarchive2);
    in2.close();

    std::ifstream in3(model_name+"_tm.model", std::ios_base::in | std::ios_base::binary);
    cereal::BinaryInputArchive iarchive3(in3);
    tm.load_ar(iarchive3);
    in3.close();
  }

  void load() { load(params.models_path); }
  void save() { save(params.models_path); }

  cv::Mat preproc_input(cv::Mat original_sai)
  {
    cv::Mat img;
    cv::resize(original_sai, img, cv::Size(params.dim, params.dim), 0, 0, cv::INTER_LANCZOS4);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::threshold(img, img, params.binary_thresh, 255, cv::THRESH_BINARY);
    return img;
  }

  void feedforward(cv::Mat const& img, bool train)
  {
    auto vec_img = mat_to_vector(img);
    input.setDense(vec_img);
    if(train)
      input.addNoise(params.train_noise);
    sp.compute(input, train, columns);
    tm.compute(columns, train);
    tm.activateDendrites();
    outTM = tm.cellsToColumns(tm.getPredictiveCells());
  }

  void visualize(note_image_t note_image, cv::Mat preproc_input, int pred_midi)
  {
    auto columns_mat = sdr3DToColorMap(columns);
    cv::namedWindow("columns", 2);
    cv::imshow("columns", columns_mat);

    auto tm_mat = sdr3DToColorMap(outTM);
    cv::namedWindow("tm", 2);
    cv::imshow("tm", tm_mat);

    cv::namedWindow("input", 2);
    cv::imshow("input", preproc_input);
    
    draw_notes(note_image);
    if(pred_midi > 0){
      note_image.midi.clear();
      note_image.midi.push_back(str_note_event_t::from_int(pred_midi));
      draw_notes(note_image, note_image.mat.rows - 30);
    }
    cv::namedWindow("nap", 2);
    cv::imshow("nap", note_image.mat);

    cv::waitKey(1);
  }
};
