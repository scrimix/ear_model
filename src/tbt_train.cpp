#include "tbt_model.h"
#include "accuracy_score.h"
#include "named_models.h"

static tbt_params_t params = deep_eye;

void accuracy_test(tbt_model_t& tbt)
{
  std::string test_dir = "../../dataset/test";
  for(auto file : list_audio_files(test_dir)){
    std::cout << "loading file: " << file << std::endl;
    
    if(check_and_gen_if_midi(file))
      file = "midi_train.wav";
    tbt.core.load_audio_file_and_notes(file);
    tbt.reset_tms();

    AccuracyStats stats;
    while(tbt.core.carfac_reader.get_render_pos() < tbt.core.audio.total_bytes()){
        auto note_image = tbt.core.carfac_reader.next();

        static int64_t skip_some = 0;
        skip_some++;
        
        auto true_labels = midi_to_labels(note_image.midi);
        auto predictions = tbt.infer(note_image);
        if(skip_some % 10 == 0)
          tbt.visualize(note_image, predictions);
        stats.update(true_labels, predictions);

        std::cout << "\rstep... " << tbt.core.audio_progress() << "%" << " | f1: "
          << std::fixed << std::setprecision(5) 
          << stats.f1() << ", recall: " << stats.recall() << ", precision:" << stats.precision();
        std::cout.flush();
    }
  }
}

void train_tbt()
{
  tbt_model_t tbt;
  
  if(!fs::exists(params.core.models_path)){
    tbt.setup(params, true);
  }
  else {
    tbt.setup(params, false);
    tbt.load();
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

        // if(skip_some % 3 == 0)
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

void test_tbt()
{
  tbt_model_t tbt;
  tbt.setup(params, false);
  tbt.load();
  accuracy_test(tbt);
}

int main()
{
  // train_tbt();
  test_tbt();
}