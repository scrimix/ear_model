#include "tbt_model.h"
#include "accuracy_score.h"
#include "named_models.h"

static tbt_params_t params = bandits;

void accuracy_test(tbt_model_t& tbt, bool with_voting = false)
{
  std::string test_dir = "../../dataset/test";
  auto files = list_audio_files(test_dir);
  std::sort(files.begin(), files.end());
  for(auto file : files){
    std::cout << "loading file: " << file << std::endl;
    
    if(check_and_gen_if_midi(file))
      file = "midi_train.wav";
    tbt.core.load_audio_file_and_notes(file);
    tbt.reset_tms();

    AccuracyStats stats;
    AccuracyStats voting_stats;
    while(tbt.core.carfac_reader.get_render_pos() < tbt.core.audio.total_bytes()){
        auto note_image = tbt.core.carfac_reader.next();

        static int64_t skip_some = 0;
        skip_some++;
        
        auto true_labels = midi_to_labels(note_image.midi);
        auto predictions = tbt.infer(note_image);
        if(skip_some % 10 == 0)
          tbt.visualize(note_image, predictions);
        if(with_voting){
          auto voting_preds = tbt.infer_voting(note_image);
          voting_stats.update(true_labels, voting_preds);
        }
        stats.update(true_labels, predictions);

      // std::cout << "sp hist: " << midi_array_to_string(predictions);
      // std::cout << ", voting: " << midi_array_to_string(voting_preds);
      // std::cout << ", gt: " << midi_array_to_string(true_labels) << std::endl;

        std::cout << "\rstep... " << std::fixed << std::setprecision(2) << tbt.core.audio_progress() << "%";
        std::cout << "    sp[" << stats << "]    vt[" << voting_stats << "]  ";
        std::cout.flush();
    }
  }
}

void train_tbt_regions()
{
  tbt_model_t tbt;
  
  if(!fs::exists(params.core.models_path)){
    tbt.setup(params, true);
  }
  else {
    tbt.params.core.models_path = params.core.models_path;
    tbt.loadv2();
  }

  auto root = "../../dataset/"s;
  std::vector<std::string> dirs = params.train_dirs;
  for(auto& dir : dirs)
    dir = root + dir;

  for(auto dir : dirs){
    for(auto file : list_audio_files(dir)){
      std::cout << "loading file: " << file << std::endl;

      if(check_and_gen_if_midi(file))
        file = "midi_train.wav";

      tbt.core.load_audio_file_and_notes(file);
      tbt.reset_tms();
      while(tbt.core.carfac_reader.get_render_pos() < tbt.core.audio.total_bytes()){
        auto note_image = tbt.core.carfac_reader.next();

        static int64_t skip_some = 0;
        skip_some++;

        tbt.train(note_image);
        
        if(skip_some % 9 == 0)
          tbt.visualize(note_image);

        std::cout << "\rstep... " << tbt.core.audio_progress() << "%";
        std::cout.flush();
      }
      std::cout << "\n";

      tbt.save();
      // break;
    }
    accuracy_test(tbt);
    // break;
  }
}


void train_tbt_voting()
{
  tbt_model_t tbt;
  
  if(!fs::exists(params.core.models_path)){
    tbt.setup(params, true);
  }
  else {
    tbt.params.core.models_path = params.core.models_path;
    tbt.loadv2();
  }

  auto root = "../../dataset/"s;
  std::vector<std::string> dirs = params.voting_dirs;
  for(auto& dir : dirs)
    dir = root + dir;

  for(auto dir : dirs){
    for(auto file : list_audio_files(dir)){
      std::cout << "loading file: " << file << std::endl;

      if(check_and_gen_if_midi(file))
        file = "midi_train.wav";

      tbt.core.load_audio_file_and_notes(file);
      tbt.reset_tms();
      auto reset_ts = 0;
      while(tbt.core.carfac_reader.get_render_pos() < tbt.core.audio.total_bytes()){
        auto note_image = tbt.core.carfac_reader.next();

        static int64_t skip_some = 0;
        skip_some++;

        tbt.train_voting(note_image);
        
        if(skip_some % 9 == 0)
          tbt.visualize(note_image);

        std::cout << "\rstep... " << tbt.core.audio_progress() << "%";
        std::cout.flush();

        auto real_ts = note_image.midi_ts / 1000.f;
        if(real_ts - reset_ts > 1.f){
          tbt.reset_tms();
          reset_ts = real_ts;
        }
      }
      std::cout << "\n";

      tbt.save_voting();
      // break;
    }
    accuracy_test(tbt, true);
    // break;
  }
}

void test_tbt()
{
  tbt_model_t tbt;
  tbt.params.core.models_path = params.core.models_path;
  tbt.loadv2();
  accuracy_test(tbt, params.use_voting_tm);
}

int main()
{
  train_tbt_regions();
  if(params.use_voting_tm)
    train_tbt_voting();
  test_tbt();
}