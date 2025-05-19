#pragma once
#include "note_model.h"
#include "midi_to_wav.h"
#include "region_split.h"
#include "crow.h"
#include <future>

template <typename T>
using ptr = std::shared_ptr<T>;

struct tbt_params_t 
{
  note_model_params_t core;
  std::vector<cv::Rect> regions;
  std::vector<std::string> train_dirs;
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

  void setup(tbt_params_t in_params, bool create_models = true) {
    params = in_params;
    if(params.core.with_note_location)
      core.setup_note_map(params.core.note_map_path);

    if(create_models){
      models.clear();
      for(auto i = 0; i < params.regions.size(); i++){
        auto model = std::make_shared<note_model_t>();
        auto loc_param = params.core;
        auto model_name = "model_"+std::to_string(i);
        loc_param.models_path = params.core.models_path+"/"+model_name+"/"+model_name;
        loc_param.region = params.regions.at(i);
        model->note_map = core.note_map;
        model->setup(loc_param);
        models.push_back(model);
      }
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
  }

  std::vector<int> infer(note_image_t& note_image){
    auto labels = midi_to_labels(note_image.midi);
    auto label = labels.empty() ? 0 : labels.at(0);
    std::sort(labels.begin(), labels.end());
    if(labels.empty())
      labels.push_back(0);

    auto infer_step = [&note_image, thresh = params.pred_thresh](ptr<note_model_t> model) -> std::vector<int> {
      auto img = note_image.mat(model->params.region);
      model->feedforward(img, {0}, false);
      PDF pdf;
      if(model->params.with_tm)
        pdf = model->clsr.infer(model->outTM);
      else
        pdf = model->clsr.infer(model->columns);
      return note_model_t::get_labels(pdf, thresh);
    };

    std::vector<std::future<std::vector<int>>> tasks;
    for(auto model : models)
      tasks.push_back(std::async(std::launch::async, std::bind(infer_step, model)));
    
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
    
    cv::waitKey(1);
  }

  void save(){
    std::ofstream file;
    fs::create_directories(params.core.models_path);
    
    file.open(params.core.models_path+"/main_params.json");
    file << tbt_params_to_json(params).dump();
    file.close();

    write_note_map_to_file(core.note_map, params.core.models_path+"/note_map.txt");

    for(auto& model : models){
      fs::path p(model->params.models_path);
      auto dir = p.parent_path();
      fs::create_directory(dir);
      model->save();
      file.open(dir.string()+"/params.json");
      file << params_to_json(model->params).dump();
      file.close();
    }
  }

  void load(){
    params = tbt_params_from_json(crow::json::load(read_text_file(params.core.models_path+"/main_params.json")));
    core.note_map = read_note_map_from_file(params.core.models_path+"/note_map.txt");
    models.clear();

    std::vector<std::future<void>> tasks;

    for (const auto& entry : fs::directory_iterator(params.core.models_path)) {
      if(entry.is_directory()){
        auto p = fs::path(entry);
        auto model = std::make_shared<note_model_t>();
        auto params_js = crow::json::load(read_text_file(p.string()+"/params.json"));
        model->note_map = core.note_map;
        model->setup(params_from_json(params_js));
        models.push_back(model);
      }
    }

    for(auto model : models)
      tasks.push_back(std::async(std::launch::async, [model]{ model->load(); }));
    for(auto& task : tasks)
      task.get();
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
  result.vote_repeats = j["vote_repeats"].i();
  result.pred_thresh = j["pred_thresh"].d();
  return result;
}
