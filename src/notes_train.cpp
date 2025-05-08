#include "note_model.h"

void train_notes(note_model_t& model)
{
  std::vector<std::string> dirs = {
    "../../dataset/warmup", "../../dataset/warmup", "../../dataset/warmup",
    "../../dataset/train2", "../../dataset/rnd_train"
  };

  for(auto dir : dirs){
    for(auto file : list_wav_files(dir)){
      std::cout << "loading file: " << file << std::endl;
      model.load_audio_file(file);
      model.tm.reset();
      while(model.carfac_reader.get_render_pos() < model.audio.total_bytes()){
        auto note_image = model.carfac_reader.next();
        
        auto img = model.preproc_input(note_image.mat);
        auto labels = midi_to_labels(note_image.midi);
        auto label = labels.empty() ? 0 : labels.at(0);

        model.feedforward(img, true);

        // model.clsr.learn(model.columns, label);
        if(model.carfac_reader.total_note_count() != 0)
          model.clsr.learn(model.outTM, label);

        static int64_t skip_some = 0;
        if(++skip_some % 10 == 0)
          model.visualize(note_image, img, 0);

        std::cout << "\rstep... " << model.carfac_reader.get_render_pos() / float(model.audio.total_bytes()) * 100 << "%";
        std::cout.flush();
      }
      // break;
      model.save("carfac_test");
    }
  }
}

void test_notes(note_model_t& model)
{
  model.load("carfac_test");
  for(auto file : list_wav_files("../../dataset/train2")){
    std::cout << "loading file: " << file << std::endl;
    model.load_audio_file(file);
    model.tm.reset();
    while(model.carfac_reader.get_render_pos() < model.audio.total_bytes()){
      auto note_image = model.carfac_reader.next();
      
      auto img = model.preproc_input(note_image.mat);
      auto labels = midi_to_labels(note_image.midi);

      model.feedforward(img, false);

      auto pdf = model.clsr.infer(model.outTM);
      auto pred_midi = argmax(pdf);

      for(auto note : labels)
        std::cout << note << " ";
      std::cout << " | ";
      if(pred_midi)
        std::cout << pred_midi;
      std::cout << std::endl;

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