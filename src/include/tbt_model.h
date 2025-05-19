#pragma once
#include "note_model.h"
#include "midi_to_wav.h"
#include "region_split.h"
#include "crow.h"
#include "voting.h"

template <typename T>
using ptr = std::shared_ptr<T>;

struct tbt_params_t 
{
  note_model_params_t core;
  std::vector<cv::Rect> regions;
  std::vector<std::string> train_dirs;

  bool use_voting_tm = false;
  voting_params_t voting_params;
  int vote_repeats = 0;
  float pred_thresh = 0.1;
};

inline crow::json::wvalue regions_to_json(const std::vector<cv::Rect>& regions);
inline std::vector<cv::Rect> regions_from_json(const crow::json::rvalue& arr);
inline crow::json::wvalue tbt_params_to_json(const tbt_params_t& params);
inline tbt_params_t tbt_params_from_json(const crow::json::rvalue& j);

struct tbt_model_t 
{
  note_model_t core;
  tbt_params_t params;
  std::vector<ptr<note_model_t>> models;
  voting_t voting;

  void setup_models()
  {
    std::vector<std::future<void>> tasks;
    for(auto& model : models)
      tasks.push_back(std::async(std::launch::async, [model]{ model->setup(model->params); }));
    for(auto& task : tasks)
      task.get();
  }

  void setup(tbt_params_t in_params, bool create_models = true) {
    params = in_params;
    if(params.core.with_note_location && !params.use_voting_tm)
      core.setup_note_map(params.core.note_map_path);

    if(create_models){
      models.clear();
      for(auto i = 0; i < params.regions.size(); i++){
        auto model = std::make_shared<note_model_t>();
        auto loc_param = params.core;
        auto model_name = "model_"+std::to_string(i);
        loc_param.models_path = params.core.models_path+"/"+model_name+"/"+model_name;
        loc_param.region = params.regions.at(i);
        if(params.core.with_note_location && !params.use_voting_tm)
          model->note_map = core.note_map;
        else
          model->note_map = create_note_map();
        model->params = loc_param;
        models.push_back(model);
      }
      setup_models();
    }

    if(params.use_voting_tm && create_models){
      params.voting_params.region_count = params.regions.size();
      voting.setup(params.voting_params);
    }
  }

  void reset_tms()
  {
    core.tm.reset();
    for(auto& model : models)
      model->tm.reset();
  }

  void train(note_image_t& note_image)
  {
    auto labels_int = midi_to_labels(note_image.midi);
    std::vector<uint32_t> labels(labels_int.begin(), labels_int.end());
    auto label = labels.empty() ? 0 : labels.at(0);
    std::sort(labels.begin(), labels.end());
    if(labels.empty())
      labels.push_back(0);

    auto train_step = [&](auto i){
      auto model = models.at(i);
      auto img = note_image.mat(model->params.region);
      model->feedforward(img, labels, true);
      if(core.carfac_reader.total_note_count() != 0){
        if(model->params.with_tm)
          model->clsr.learn(model->outTM, labels);
        else
          model->clsr.learn(model->columns, labels);
      }
    };

    std::vector<std::future<void>> tasks;
    for(auto i = 0; i < models.size(); i++)
      tasks.push_back(std::async(std::launch::async, [&, i]{ train_step(i); }));
    for (auto& task : tasks)
      task.get();

    if(params.use_voting_tm){
      std::vector<note_location_t> notes_per_region;
      for(auto& model : models){
        auto pred = midi_pred_to_location(model->note_map, infer_step(model, note_image));
        notes_per_region.push_back(pred);
      }
      voting.train(labels, notes_per_region);
    }
  }

  std::vector<int> infer_step(ptr<note_model_t> model, note_image_t const& note_image) {
    auto img = note_image.mat(model->params.region);
    model->feedforward(img, {0}, false);
    PDF pdf;
    if(model->params.with_tm)
      pdf = model->clsr.infer(model->outTM);
    else
      pdf = model->clsr.infer(model->columns);
    return note_model_t::get_labels(pdf, params.pred_thresh);
  }

  std::vector<int> infer(note_image_t& note_image, std::vector<int>* voting_preds = nullptr){
    auto labels = midi_to_labels(note_image.midi);
    auto label = labels.empty() ? 0 : labels.at(0);
    std::sort(labels.begin(), labels.end());
    if(labels.empty())
      labels.push_back(0);

    std::vector<std::future<std::vector<int>>> tasks;
    for(auto model : models)
      tasks.push_back(std::async(std::launch::async, std::bind(&tbt_model_t::infer_step, this, model, note_image)));
    
    std::vector<std::vector<int>> midi_preds;
    for (auto& task : tasks)
      midi_preds.push_back(task.get());

    std::map<int, int> midi_hist;
    for(auto& model_result : midi_preds){
      for(auto& label_idx : model_result)
        midi_hist[label_idx]++;
    }

    std::vector<int> result;
    for(auto& [label_idx, repeats] : midi_hist)
      if(repeats > params.vote_repeats)
        result.push_back(label_idx);
    
    if(params.use_voting_tm){
      std::vector<note_location_t> notes_per_region;
      for(auto& model : models){
        auto pred = midi_pred_to_location(model->note_map, infer_step(model, note_image));
        notes_per_region.push_back(pred);
      }
      auto voting_result = voting.infer(notes_per_region);
      if(voting_preds)
        *voting_preds = voting_result;
    }

    return result;
  }

  void visualize(note_image_t& note_image, std::vector<int> pred_midi = {})
  {
    core.draw_notes(note_image, pred_midi);
    
    std::vector<cv::Mat> nn_vis;
    auto images_per_nn = 3;
    for(auto& model : models){
      auto model_vis = model->get_visualizations();
      concat(&nn_vis, model_vis);
      images_per_nn = model_vis.size();
      cv::rectangle(note_image.mat, model->params.region, cv::Scalar(80,80,80), 2);
    }
    auto nn_mat = tileImages(nn_vis, sqrt(nn_vis.size()), 15);

    show("SAI", note_image.mat);
    show("activations", nn_mat);
    if(params.use_voting_tm)
      voting.visualize();
    cv::waitKey(1);
  }

  void save(){
    std::ofstream file;
    fs::create_directories(params.core.models_path);
    
    file.open(params.core.models_path+"/main_params.json");
    file << tbt_params_to_json(params).dump();
    file.close();

    if(params.core.with_note_location && !params.use_voting_tm)
      write_note_map_to_file(core.note_map, params.core.models_path+"/note_map.txt");

    for(auto& model : models){
      model->save();

      fs::path p(model->params.models_path);
      write_text_to_file(p.parent_path().string()+"/params.json", params_to_json(model->params).dump());

      if(!params.core.with_note_location && params.use_voting_tm)
        write_note_map_to_file(model->note_map, p.parent_path().string()+"/note_map.txt");
    }

    if(params.use_voting_tm){
      voting.save(params.core.models_path+"/voting/voting");
      write_text_to_file(params.core.models_path+"/voting/params.json", voting_params_to_json(voting.params).dump());
    }
  }

  void load(){
    params = tbt_params_from_json(crow::json::load(read_text_file(params.core.models_path+"/main_params.json")));
    if(params.core.with_note_location && !params.use_voting_tm)
      core.note_map = read_note_map_from_file(params.core.models_path+"/note_map.txt");
    models.clear();

    std::vector<std::future<void>> tasks;

    std::vector<std::string> model_dirs;
    for (const auto& entry : fs::directory_iterator(params.core.models_path)) {
      if(entry.is_directory() && entry.path().stem().string().starts_with("model")){
        model_dirs.push_back(entry.path());
      }
    }
    std::sort(model_dirs.begin(), model_dirs.end(), [](auto& d1, auto& d2) -> bool {
      auto idx1 = std::stoi(split(fs::path(d1).stem(), "_").at(1));
      auto idx2 = std::stoi(split(fs::path(d2).stem(), "_").at(1));
      return idx1 < idx2;
    });

    for (auto& model_dir : model_dirs) {
      auto p = fs::path(model_dir);
      auto model = std::make_shared<note_model_t>();
      auto params_js = crow::json::load(read_text_file(p.string()+"/params.json"));
      auto model_params = params_from_json(params_js);
      if(params.core.with_note_location && !params.use_voting_tm)
        model->note_map = core.note_map;
      else if(params.use_voting_tm)
        model->note_map = read_note_map_from_file(p.string()+"/note_map.txt");
      model->params = model_params;
      models.push_back(model);
    }

    for(auto model : models){
      tasks.push_back(std::async(std::launch::async, [model]{ 
        model->setup(model->params);
        model->load();
      }));
    }
    for(auto& task : tasks)
      task.get();

    if(params.use_voting_tm){
      auto params_txt = read_text_file(params.core.models_path+"/voting/params.json");
      params.voting_params = voting_params_from_json(crow::json::load(params_txt));
      voting.setup(params.voting_params);
      voting.load(params.core.models_path+"/voting/voting");
    }
  }

};

inline crow::json::wvalue regions_to_json(const std::vector<cv::Rect>& regions) {
  crow::json::wvalue arr;
  for (size_t i = 0; i < regions.size(); ++i) {
    const cv::Rect& r = regions[i];
    arr[i]["x"] = r.x;
    arr[i]["y"] = r.y;
    arr[i]["width"] = r.width;
    arr[i]["height"] = r.height;
  }
  return arr;
}

inline std::vector<cv::Rect> regions_from_json(const crow::json::rvalue& arr) {
  std::vector<cv::Rect> regions;
  for (size_t i = 0; i < arr.size(); ++i) {
    int x = arr[i]["x"].i();
    int y = arr[i]["y"].i();
    int w = arr[i]["width"].i();
    int h = arr[i]["height"].i();
    regions.emplace_back(x, y, w, h);
  }
  return regions;
}

inline crow::json::wvalue tbt_params_to_json(const tbt_params_t& params) {
  crow::json::wvalue result;
  result["core"] = params_to_json(params.core);
  result["regions"] = regions_to_json(params.regions);
  for (size_t i = 0; i < params.train_dirs.size(); ++i)
    result["train_dirs"][i] = params.train_dirs[i];
  result["use_voting_tm"] = params.use_voting_tm;
  result["voting_params"] = voting_params_to_json(params.voting_params);
  result["vote_repeats"] = params.vote_repeats;
  result["pred_thresh"] = params.pred_thresh;
  return result;
}

inline tbt_params_t tbt_params_from_json(const crow::json::rvalue& j) {
  tbt_params_t result;
  result.core = params_from_json(j["core"]);
  result.regions = regions_from_json(j["regions"]);
  for (size_t i = 0; i < j["train_dirs"].size(); ++i)
    result.train_dirs.push_back(j["train_dirs"][i].s());
  result.use_voting_tm = j["use_voting_tm"].b();
  result.voting_params = voting_params_from_json(j["voting_params"]);
  result.vote_repeats = j["vote_repeats"].i();
  result.pred_thresh = j["pred_thresh"].d();
  return result;
}
