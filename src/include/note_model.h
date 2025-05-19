#pragma once

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
#include "note_location.h"
#include "crow.h"

using namespace std;
using namespace htm;

struct note_model_params_t {
  // path to models
  std::string models_path;

  // htm
  uint height = 32;
  uint width = 32;
  uint column_count = 8;
  int pot_radius = 8;
  int binary_thresh = 40;
  float train_noise = 0.1;
  bool with_note_location = false;
  std::string note_map_path = "../../dataset/note_map.txt";
  
  bool with_tm = false;
  int tm_memory = 50;
  uint tm_cell_per_column = 6;

  // tbt
  cv::Rect region;
  
  // carfac
  float loudness_coef = 0.1;
  int sample_rate = 44100;
  int buffer_size = 1024;
};

class note_model_t {
public:
  cv::Mat input_image;
  SpatialPooler sp;
  TemporalMemory tm;
  note_location_t note_sdr;
  SDR input;
  SDR columns;
  SDR outTM;
  Classifier clsr;

  carfac_reader_t carfac_reader;
  AudioData audio;

  note_model_params_t params;
  note_map_t note_map;

  void setup_note_map(std::string file_path)
  {
    if(fs::exists(file_path)){
      note_map = read_note_map_from_file(file_path);
    }
    else{
      note_map = create_note_map();
      write_note_map_to_file(note_map, file_path);
    }
  }

  void setup(note_model_params_t model_params) {
    params = model_params;

    input.initialize({params.height, params.width, 1});
    columns.initialize({params.height, params.width, params.column_count}); //1D vs 2D no big difference, 2D seems more natural for the problem. Speed-----, Results+++++++++; #columns HIGHEST impact. 
    
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

    if(params.with_tm)
      tm.initialize(columns.dimensions, params.tm_cell_per_column, 13, 0.21, 0.5, 10, 20, 0.1, 0.1, 0, 42, params.tm_memory, params.tm_memory);
    clsr.initialize( /* alpha */ 0.001f);
  }

  void load_audio_file_and_notes(std::string file_path)
  {
    carfac_reader.reset();
    carfac_reader.clear_all_notes();
    carfac_reader.init(file_path);
    carfac_reader.set(params.sample_rate, params.buffer_size, params.loudness_coef);
    audio.buffer = readWavFile(file_path);
  }

  void load_audio(std::vector<float> const& wav)
  {
    carfac_reader.reset();
    carfac_reader.clear_all_notes();
    carfac_reader.init(wav);
    carfac_reader.set(params.sample_rate, params.buffer_size, params.loudness_coef);
    audio.buffer = wav;
  }

  double audio_progress() const
  {
    return carfac_reader.get_render_pos() / float(audio.total_bytes()) * 100;
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

  bool load(std::string model_name)
  {
    if(!std::filesystem::exists(model_name+"_sp.model")){
      std::cerr << "note_model_t | loading model: " << model_name << " failed! File doesn't exist";
      return false;
    }

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
    return true;
  }

  void load() { load(params.models_path); }
  void save() { save(params.models_path); }

  cv::Mat preproc_input(cv::Mat original_sai)
  {
    cv::Mat img;
    cv::resize(original_sai, img, cv::Size(params.width, params.width), 0, 0, cv::INTER_LANCZOS4);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::threshold(img, img, params.binary_thresh, 255, cv::THRESH_BINARY);
    return img;
  }

  void feedforward(cv::Mat const& sai, std::vector<uint> const& labels, bool train)
  {
    input_image = preproc_input(sai);
    auto vec_img = mat_to_vector(input_image);

    if(params.with_note_location){
      auto orig = vec_img.size();
      note_sdr = {};
      for(auto& label : labels)
        note_sdr |= note_map.at(label);
      vec_img = concat(vec_img, note_sdr);
      input_image = vectorToMat(vec_img, params.height, params.width);
    }

    input.setDense(vec_img);
    if(train)
      input.addNoise(params.train_noise);
    sp.compute(input, train, columns);
    if(params.with_tm){
      tm.compute(columns, train);
      tm.activateDendrites();
      outTM = tm.cellsToColumns(tm.getPredictiveCells());
    }
  }

  static std::vector<int> get_labels(vector<double> const& pdf, double thresh = 0.5)
  {
    std::vector<int> result;
    auto best_preds = topNIndices(pdf, 10);
      for(auto idx : best_preds)
        if(pdf.at(idx) > thresh)
          result.push_back(idx);
    return result;
  }

  void draw_notes(note_image_t& note_image, std::vector<int> pred_midi)
  {
    draw_notes_as_keys(note_image);
    if(!pred_midi.empty()){
      note_image.midi.clear();
      for(auto note : pred_midi)
        note_image.midi.push_back(str_note_event_t::from_int(note));
      // draw_notes(note_image, note_image.mat.rows - 30);
      draw_notes_as_keys(note_image, note_image.mat.rows - 30);
    }
  }

  std::vector<cv::Mat> get_visualizations()
  {
    std::vector<cv::Mat> result;
    result.push_back(input_image.clone());
    result.push_back(sdr3DToColorMap(columns));
    if(params.with_tm)
      result.push_back(sdr3DToColorMap(outTM));
    return result;
  }

  void visualize(note_image_t note_image, std::vector<int> pred_midi)
  {
    // cv::Mat columns_mat;
    // if(params.with_note_location)
    //   columns_mat = draw_sp_output(columns, params.dim, params.dim, note_location_resolution);
    // else
    auto columns_mat = sdr3DToColorMap(columns);
    cv::namedWindow("columns", 2);
    cv::imshow("columns", columns_mat);

    // cv::Mat tm_mat;
    // if(params.with_note_location)
    //   tm_mat = draw_sp_output(outTM, params.dim, params.dim, note_location_resolution);
    // else
    auto tm_mat = sdr3DToColorMap(outTM);
    cv::namedWindow("tm", 2);
    cv::imshow("tm", tm_mat);

    cv::namedWindow("input", 2);
    cv::imshow("input", input_image);
    
    // draw_notes(note_image);
    draw_notes_as_keys(note_image);
    if(!pred_midi.empty()){
      note_image.midi.clear();
      for(auto note : pred_midi)
        note_image.midi.push_back(str_note_event_t::from_int(note));
      // draw_notes(note_image, note_image.mat.rows - 30);
      draw_notes_as_keys(note_image, note_image.mat.rows - 30);
    }
    cv::namedWindow("nap", 2);
    cv::imshow("nap", note_image.mat);

    cv::waitKey(1);
  }
};


inline crow::json::wvalue params_to_json(const note_model_params_t& params) {
  crow::json::wvalue j;

  j["models_path"] = params.models_path;
  j["height"] = params.height;
  j["width"] = params.width;
  j["column_count"] = params.column_count;
  j["pot_radius"] = params.pot_radius;
  j["tm_cell_per_column"] = params.tm_cell_per_column;
  j["binary_thresh"] = params.binary_thresh;
  j["train_noise"] = params.train_noise;
  j["with_note_location"] = params.with_note_location;
  j["with_tm"] = params.with_tm;
  j["note_map_path"] = params.note_map_path;
  j["tm_memory"] = params.tm_memory;
  j["region"] = {
      {"x", params.region.x},
      {"y", params.region.y},
      {"width", params.region.width},
      {"height", params.region.height}
  };
  j["loudness_coef"] = params.loudness_coef;
  j["sample_rate"] = params.sample_rate;
  j["buffer_size"] = params.buffer_size;

  return j;
}

inline note_model_params_t params_from_json(const crow::json::rvalue& j) {
  note_model_params_t params;

  params.models_path = j["models_path"].s();
  params.height = j["height"].u();
  params.width = j["width"].u();
  params.column_count = j["column_count"].u();
  params.pot_radius = j["pot_radius"].i();
  params.tm_cell_per_column = j["tm_cell_per_column"].u();
  params.binary_thresh = j["binary_thresh"].i();
  params.train_noise = static_cast<float>(j["train_noise"].d());
  params.with_note_location = j["with_note_location"].b();
  params.with_tm = j["with_tm"].b();
  params.note_map_path = j["note_map_path"].s();
  params.tm_memory = j["tm_memory"].i();
  params.region = cv::Rect(
      j["region"]["x"].i(),
      j["region"]["y"].i(),
      j["region"]["width"].i(),
      j["region"]["height"].i()
  );
  params.loudness_coef = static_cast<float>(j["loudness_coef"].d());
  params.sample_rate = j["sample_rate"].i();
  params.buffer_size = j["buffer_size"].i();

  return params;
}
