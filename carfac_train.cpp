
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

// static const int dim = 32;
// static const int pot_radius = 8;
// static const int column_count = 12;
// static const int tm_cell_per_column = 8;

static const int dim = 32;
static const int pot_radius = 8;
static const int column_count = 8;
static const int tm_cell_per_column = 6;

class NoteDataset {
public:
  SpatialPooler sp;
  TemporalMemory tm;
  SDR input;
  SDR columns;
  SDR outTM;
  Classifier clsr;

  carfac_reader_t carfac_reader;
  AudioData audio;

  void setup() {

    input.initialize({dim, dim, 1});
    columns.initialize({dim, dim, column_count}); //1D vs 2D no big difference, 2D seems more natural for the problem. Speed-----, Results+++++++++; #columns HIGHEST impact. 
    sp.initialize(
      /* inputDimensions */             input.dimensions,
      /* columnDimensions */            columns.dimensions,
      /* potentialRadius */             pot_radius, // with 2D, 7 results in 15x15 area, which is cca 25% for the input area. Slightly improves than 99999 aka "no topology, all to all connections"
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

    tm.initialize(columns.dimensions, tm_cell_per_column, 13, 0.21, 0.5, 10, 20, 0.1, 0.1, 0, 42, 50, 50);
    clsr.initialize( /* alpha */ 0.001f);
  }

  void load_audio_file(std::string file_path)
  {
    carfac_reader.reset();
    auto sample_rate = 44100;
    auto buffer_size = 1024;
    auto loudness_coef = 40;
    carfac_reader.init(file_path);
    carfac_reader.set(sample_rate, buffer_size, loudness_coef);
    audio.buffer = read_wav(file_path, sample_rate);
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

  cv::Mat preproc_input(cv::Mat original_sai)
  {
    cv::Mat img;
    cv::resize(original_sai, img, cv::Size(dim,dim), 0, 0, cv::INTER_LANCZOS4);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::threshold(img, img, 40, 255, cv::THRESH_BINARY);
    return img;
  }

  void feedforward(cv::Mat const& img, bool train)
  {
    auto vec_img = mat_to_vector(img);
    input.setDense(vec_img);
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

  void train_notes()
  {
    for(auto file : list_wav_files("../../dataset/rnd_train")){
      std::cout << "loading file: " << file << std::endl;
      load_audio_file(file);
      tm.reset();
      while(carfac_reader.get_render_pos() < audio.total_bytes()){
        auto note_image = carfac_reader.next();
        
        auto img = preproc_input(note_image.mat);
        auto labels = midi_to_labels(note_image.midi);
        auto label = labels.empty() ? 0 : labels.at(0);

        feedforward(img, true);

        // clsr.learn(columns, label);
        clsr.learn(outTM, label);

        visualize(note_image, img, 0);

        std::cout << "\rstep... " << carfac_reader.get_render_pos() / float(audio.total_bytes()) * 100 << "%";
        std::cout.flush();
      }
      // break;
    }
    save("carfac_notes");
  }

  void test_notes()
  {
    // load("../../models/carfac_large");
    load("../../models/carfac_notes");

    for(auto file : list_wav_files("../../dataset/rnd_train")){
      std::cout << "loading file: " << file << std::endl;
      load_audio_file(file);
      tm.reset();
      while(carfac_reader.get_render_pos() < audio.total_bytes()){
        auto note_image = carfac_reader.next();
        
        auto img = preproc_input(note_image.mat);
        auto labels = midi_to_labels(note_image.midi);

        feedforward(img, false);

        auto pdf = clsr.infer(outTM);
        auto pred_midi = argmax(pdf);

        for(auto note : labels)
          std::cout << note << " ";
        std::cout << " | ";
        if(pred_midi)
          std::cout << pred_midi;
        std::cout << std::endl;

        visualize(note_image, img, pred_midi);
      }
    }
  }

};  // End class NoteDataset

int main(int argc, char **argv) {
  NoteDataset m;
  m.setup();
  // m.train_notes();
  m.test_notes();

  return 0;
}
