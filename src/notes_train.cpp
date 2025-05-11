#include "note_model.h"
#include "midi_to_wav.h"
#include <fstream>

static const std::string model_name = "fenrir";

bool check_and_gen_if_midi(std::string file)
{
  std::filesystem::path p(file);
  std::string ext = p.extension();
  for(auto& c : ext)
    c = tolower(c);
  if(ext == ".mid" || ext == ".midi"){
    smf::MidiFile midi_file;
    std::ifstream stream(file);
    midi_file.read(stream);
    midi_file.doTimeAnalysis();
    midi_file.linkNotePairs();
    create_wav_and_labels(midi_file, ".",  "midi_train", 2);
    return true;
  }
  return false;
}

void train_notes(note_model_t& model)
{
  auto root = "../../dataset/"s;

  std::vector<std::string> dirs = {
    "warmup", "warmup", "train/train2", "train/rnd_train"
  };
  for(auto& dir : dirs)
    dir = root + dir;

  // std::vector<std::string> dirs = { 
  //   root+"warmup", root+"recs", root+"recs", root+"recs"
  // };

  // auto dirs = { "../../../maestro-v3.0.0" };

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
        
        auto img = model.preproc_input(note_image.mat);
        auto labels = midi_to_labels(note_image.midi);
        auto label = labels.empty() ? 0 : labels.at(0);
        std::sort(labels.begin(), labels.end());
        if(labels.empty())
          labels.push_back(0);

        model.feedforward(img, true);

        // model.clsr.learn(model.columns, label);
        if(model.carfac_reader.total_note_count() != 0)
          model.clsr.learn(model.outTM, labels);

        static int64_t skip_some = 0;
        if(++skip_some % 10 == 0)
          model.visualize(note_image, img, {});

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

  auto test_dir = "../../dataset/train/rnd_multi";
  // auto test_dir = "../../../maestro-v3.0.0";

  for(auto file : list_audio_files(test_dir)){
    std::cout << "loading file: " << file << std::endl;
    if(check_and_gen_if_midi(file))
        file = "midi_train.wav";
    model.load_audio_file_and_notes(file);
    model.tm.reset();
    while(model.carfac_reader.get_render_pos() < model.audio.total_bytes()){
      auto note_image = model.carfac_reader.next();
      
      auto img = model.preproc_input(note_image.mat);
      auto labels = midi_to_labels(note_image.midi);

      model.feedforward(img, false);

      auto pdf = model.clsr.infer(model.outTM);
      auto pred_midi = note_model_t::get_labels(pdf, 0.3);

      model.visualize(note_image, img, pred_midi);
    }
  }
}

int main()
{
  note_model_t model;
  model.setup({});
  train_notes(model);
  test_notes(model);
}