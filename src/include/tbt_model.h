#pragma once
#include "note_model.h"
#include "midi_to_wav.h"
#include "region_split.h"
#include "crow.h"
#include "voting.h"
#include <semaphore>

template <typename T>
using ptr = std::shared_ptr<T>;

static constexpr int LOADING_THREADS = 8;

struct tbt_params_t 
{
  note_model_params_t core;
  std::vector<cv::Rect> regions;
  std::vector<std::string> train_dirs;
  std::vector<std::string> voting_dirs;

  bool use_voting_tm = false;
  bool limit_region_notes = false;
  voting_params_t voting_params;
  int vote_repeats = 0;
  float pred_thresh = 0.1;

  bool operator==(tbt_params_t const& other) const;
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
    std::counting_semaphore<LOADING_THREADS> thread_limit(LOADING_THREADS);
    std::vector<std::future<void>> tasks;
    for(auto& model : models){
      thread_limit.acquire();
      tasks.push_back(std::async(std::launch::async, [&, model]{
        model->setup(model->params);
        thread_limit.release();
      }));
    }
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
    voting.tm.reset();
  }

  void train(note_image_t& note_image)
  {
    auto labels = get_labels(note_image);

    auto train_step = [&](auto i){
      auto model = models.at(i);
      auto img = note_image.mat(model->params.region);

      model->feedforward(img, labels, true);
      if(core.carfac_reader.total_note_count() != 0){
        auto local_labels = labels;
        if(params.limit_region_notes)
          local_labels = labels_to_region_specific(labels, model->params.region, note_image.mat.size());
        if(model->params.with_tm)
          model->clsr.learn(model->outTM, local_labels);
        else
          model->clsr.learn(model->columns, local_labels);
      }
    };

    std::vector<std::future<void>> tasks;
    for(auto i = 0; i < models.size(); i++)
      tasks.push_back(std::async(std::launch::async, [&, i]{ train_step(i); }));
    for (auto& task : tasks)
      task.get();
  }

  std::vector<uint32_t> get_labels(note_image_t const& note_image)
  {
    auto labels_int = midi_to_labels(note_image.midi);
    std::vector<uint32_t> labels(labels_int.begin(), labels_int.end());
    auto label = labels.empty() ? 0 : labels.at(0);
    std::sort(labels.begin(), labels.end());
    if(labels.empty())
      labels.push_back(0);
    return labels;
  }

  std::vector<note_location_t> get_votes(note_image_t& note_image)
  {
    auto region_preds = infer_many(note_image);
    return voting.region_preds_to_location(region_preds);
  }

  void train_voting(note_image_t& note_image)
  {
    voting.train(get_labels(note_image), get_votes(note_image));
  }

  std::vector<int> infer_step(ptr<note_model_t> model, note_image_t const& note_image) {
    auto img = note_image.mat(model->params.region);
    model->feedforward(img, {0}, false);
    PDF pdf;
    if(model->params.with_tm)
      pdf = model->clsr.infer(model->outTM);
    else
      pdf = model->clsr.infer(model->columns);
    auto labels = note_model_t::get_labels(pdf, params.pred_thresh);
    if(params.limit_region_notes)
      labels = labels_from_region_to_global(labels, model->params.region, note_image.mat.size());
    return remove_zero(labels);
  }

  std::vector<std::vector<int>> infer_many(note_image_t const& note_image)
  {
    std::counting_semaphore<8> thread_limit(8);
    std::vector<std::future<std::vector<int>>> tasks;
    for(auto model : models){
      thread_limit.acquire();
      tasks.push_back(std::async(std::launch::async, [&, model]{
        auto result = this->infer_step(model, note_image);
        thread_limit.release();
        return result;
      }));
    }
    std::vector<std::vector<int>> region_preds;
    for (auto& task : tasks)
      region_preds.push_back(task.get());
    return region_preds;
  }

  std::vector<int> hist_voting(std::vector<std::vector<int>> const& region_preds)
  {
    std::map<int, int> midi_hist;
    for(auto& model_result : region_preds){
      for(auto& label_idx : model_result)
        midi_hist[label_idx]++;
    }

    std::vector<int> result;
    for(auto& [label_idx, repeats] : midi_hist)
      if(repeats > params.vote_repeats)
        result.push_back(label_idx);

    return remove_zero(result);
  }

  std::vector<int> infer(note_image_t const& note_image){
    auto labels = get_labels(note_image);
    auto region_preds = infer_many(note_image);
    return hist_voting(region_preds);
  }

  std::vector<int> infer_voting(note_image_t& note_image)
  {
    auto votes = get_votes(note_image);
    auto result = remove_zero(voting.infer(votes));
    return result;
  }

  void draw_regions(note_image_t& note_image)
  {
    auto is_active = [&](auto model){
      if(!params.limit_region_notes)
        return false;
      auto region = model->params.region;
      auto image_size = note_image.mat.size();
      auto [midi_low, midi_high] = get_midi_range_for_region(region.y, region.height, image_size.height);
      for(auto& note : note_image.midi)
        if(note.to_midi_int() >= midi_low && note.to_midi_int() <= midi_high)
          return true;
      return false;
    };

    auto get_color = [&](auto model){
      if(is_active(model))
        return cv::Scalar(50,50,150);
      else
        return cv::Scalar(30,30,30);
    };

    for(auto& model : models)
      cv::rectangle(note_image.mat, model->params.region, get_color(model), 2);
  }

  cv::Mat get_activations_image()
  {
    std::vector<cv::Mat> nn_vis;
    auto images_per_nn = 3;
    for(auto& model : models){
      auto model_vis = model->get_visualizations();
      concat(&nn_vis, model_vis);
      images_per_nn = model_vis.size();
    }
    auto nn_mat = tileImages(nn_vis, sqrt(nn_vis.size()), 15);
    return nn_mat;
  }

  void visualize(note_image_t& note_image, std::vector<int> pred_midi = {})
  {
    core.draw_notes(note_image, pred_midi);
    draw_regions(note_image);
    auto nn_mat = get_activations_image();

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

  void save_voting()
  {
    voting.save(params.core.models_path+"/voting/voting");
    write_text_to_file(params.core.models_path+"/voting/params.json", voting_params_to_json(voting.params).dump());
  }

  void loadv2()
  {
    auto full_path = params.core.models_path;
    params = tbt_params_from_json(crow::json::load(read_text_file(params.core.models_path+"/main_params.json")));
    params.core.models_path = full_path;
    if(params.core.with_note_location && !params.use_voting_tm)
      core.note_map = read_note_map_from_file(params.core.models_path+"/note_map.txt");
    setup(params, true);

    std::counting_semaphore<LOADING_THREADS> thread_limit(LOADING_THREADS);
    std::vector<std::future<void>> tasks;
    for(auto model : models){
      thread_limit.acquire();
      tasks.push_back(std::async(std::launch::async, [&thread_limit, model]{ 
        model->load();
        thread_limit.release();
      }));
    }
    for(auto& task : tasks)
      task.get();

    if(params.use_voting_tm){
      if(fs::exists(params.core.models_path+"/voting")){
        auto params_txt = read_text_file(params.core.models_path+"/voting/params.json");
        params.voting_params = voting_params_from_json(crow::json::load(params_txt));
        voting.setup(params.voting_params);
        voting.load(params.core.models_path+"/voting/voting");
      }
      else{
        params.voting_params.region_count = params.regions.size();
        voting.setup(params.voting_params);
      }
    }
  }

  // segfaults, something is not right
  // something to do with setup + load sequence??
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
      if(fs::exists(params.core.models_path+"/voting")){
        auto params_txt = read_text_file(params.core.models_path+"/voting/params.json");
        params.voting_params = voting_params_from_json(crow::json::load(params_txt));
        voting.setup(params.voting_params);
        voting.load(params.core.models_path+"/voting/voting");
      }
      else{
        params.voting_params.region_count = params.regions.size();
        voting.setup(params.voting_params);
      }
    }
  }

  void reset()
  {
    reset_tms();
    core.carfac_reader.reset();
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
  for (size_t i = 0; i < params.voting_dirs.size(); ++i)
    result["voting_dirs"][i] = params.voting_dirs[i];
  result["use_voting_tm"] = params.use_voting_tm;
  result["voting_params"] = voting_params_to_json(params.voting_params);
  result["vote_repeats"] = params.vote_repeats;
  result["pred_thresh"] = params.pred_thresh;
  result["limit_region_notes"] = params.limit_region_notes;
  return result;
}

inline tbt_params_t tbt_params_from_json(const crow::json::rvalue& j) {
  tbt_params_t result;
  result.core = params_from_json(j["core"]);
  result.regions = regions_from_json(j["regions"]);
  for (size_t i = 0; i < j["train_dirs"].size(); ++i)
    result.train_dirs.push_back(j["train_dirs"][i].s());
  for (size_t i = 0; i < j["voting_dirs"].size(); ++i)
    result.voting_dirs.push_back(j["voting_dirs"][i].s());
  result.use_voting_tm = j["use_voting_tm"].b();
  result.voting_params = voting_params_from_json(j["voting_params"]);
  result.vote_repeats = j["vote_repeats"].i();
  result.pred_thresh = j["pred_thresh"].d();
  result.limit_region_notes = j["limit_region_notes"].b();
  return result;
}

inline bool tbt_params_t::operator==(tbt_params_t const& other) const
{
  return 
    core == other.core && 
    regions == other.regions && 
    train_dirs == other.train_dirs && 
    voting_dirs == other.voting_dirs && 
    use_voting_tm == other.use_voting_tm && 
    voting_params == other.voting_params && 
    vote_repeats == other.vote_repeats && 
    pred_thresh == other.pred_thresh;
}