#include "note_model.h"
#include "midi_to_wav.h"
#include <fstream>

static const std::string model_name = "poly";

void train_notes(note_model_t& model)
{
  auto root = "../dataset/"s;

  // std::vector<std::string> dirs = {
  //   "warmup", "warmup", "train/train2", "train/rnd_train"
  // };

  // std::vector<std::string> dirs = { 
  //   root+"warmup", root+"recs", root+"recs", root+"recs"
  // };

  // auto dirs = { "../../../maestro-v3.0.0" };

  std::vector<std::string> dirs = { "rnd_train" };

  for(auto& dir : dirs)
    dir = root + dir;

  // model.load(model_name); // continue training

  for(auto dir : dirs){
    for(auto file : list_audio_files(dir)){
      std::cout << "loading file: " << file << std::endl;

      if(check_and_gen_if_midi(file))
        file = "midi_train.wav";

      model.load_audio_file_and_notes(file);
      model.tm.reset();
      while(model.carfac_reader.get_render_pos() < model.audio.total_bytes()){
        auto note_image = model.carfac_reader.next();
        
        auto labels_int = midi_to_labels(note_image.midi);
        std::vector<uint32_t> labels(labels_int.begin(), labels_int.end());
        auto label = labels.empty() ? 0 : labels.at(0);
        std::sort(labels.begin(), labels.end());
        if(labels.empty())
          labels.push_back(0);

        model.feedforward(note_image.mat, labels, true);

        if(model.carfac_reader.total_note_count() != 0){
          if(model.params.with_tm)
            model.clsr.learn(model.outTM, labels);
          else
            model.clsr.learn(model.columns, labels);
        }

        static int64_t skip_some = 0;
        if(++skip_some % 10 == 0)
          model.visualize(note_image, {});

        std::cout << "\rstep... " << model.carfac_reader.get_render_pos() / float(model.audio.total_bytes()) * 100 << "%";
        std::cout.flush();
      }
      model.save(model_name);
    }
  }
}

std::vector<double> loc_softmax(const std::vector<double>& logits, float temperature = 1.0) {
    std::vector<double> result(logits.size());
    float maxLogit = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp((logits[i] - maxLogit) / temperature);
        sum += result[i];
    }

    for (double &val : result) {
        val /= sum;
    }

    return result;
}

void test_notes(note_model_t& model)
{
  // if(!model.load("../../stable_models/"+model_name+"/"+model_name))
  if(!model.load(model_name))
    return;

  auto test_dir = "../dataset/rnd_train";
  // auto test_dir = "../../../maestro-v3.0.0";

  for(auto file : list_audio_files(test_dir)){
    std::cout << "loading file: " << file << std::endl;
    if(check_and_gen_if_midi(file))
        file = "midi_train.wav";
    model.load_audio_file_and_notes(file);
    model.tm.reset();
    while(model.carfac_reader.get_render_pos() < model.audio.total_bytes()){
      auto note_image = model.carfac_reader.next();
      auto labels = midi_to_labels(note_image.midi);

      model.feedforward(note_image.mat, {0}, false);

      PDF pdf;
      if(model.params.with_tm)
        pdf = model.clsr.infer(model.outTM);
      else
        pdf = model.clsr.infer(model.columns);
      auto pred_midi = note_model_t::get_labels(pdf, 0.3);

      model.visualize(note_image, pred_midi);
    }
  }
}

int main()
{
  note_model_t model;
  note_model_params_t params;
  params.with_note_location = true;

  model.setup(params);
  train_notes(model);
  test_notes(model);
}