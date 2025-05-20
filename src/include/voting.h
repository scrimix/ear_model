#pragma once
#include "note_location.h"
#include "note_model.h"

struct voting_params_t
{
  int region_count = 0;
  int cell_count = 16;
  int context_length = 50;
  float pred_thresh = 0.1;

  bool operator==(voting_params_t const& other) const;
};

inline crow::json::wvalue voting_params_to_json(voting_params_t const& params)
{
  crow::json::wvalue result;
  result["region_count"] = params.region_count;
  result["cell_count"] = params.cell_count;
  result["context_length"] = params.context_length;
  result["pred_thresh"] = params.pred_thresh;
  return result;
}

inline voting_params_t voting_params_from_json(crow::json::rvalue const& j)
{
  voting_params_t result;
  result.region_count = j["region_count"].i();
  result.cell_count = j["cell_count"].i();
  result.context_length = j["context_length"].i();
  result.pred_thresh = j["pred_thresh"].d();
  return result;
}

inline bool voting_params_t::operator==(voting_params_t const& other) const {
  return
    region_count == other.region_count && 
    cell_count == other.cell_count && 
    context_length == other.context_length && 
    pred_thresh == other.pred_thresh;
}

struct voting_t
{
  voting_params_t params;
  std::vector<uint8_t> note_sdr;
  SDR input;
  TemporalMemory tm;
  SDR tm_out;
  Classifier clsr;

  void setup(voting_params_t in_params)
  {
    params = in_params;
    auto dimensions = std::vector<uint32_t>{note_location_resolution, 1};
    input.initialize(dimensions);
    tm.initialize(dimensions, params.cell_count, 13, 0.21, 0.5, 10, 20, 0.1, 0.1, 0, 42, params.context_length, params.context_length);
    clsr.initialize( /* alpha */ 0.001f);
  }

  void train(std::vector<uint32_t> const& labels, std::vector<note_location_t> const& notes_per_region)
  {
    // note_sdr.clear();
    // for(auto& note_loc : notes_per_region){
    //   auto notes = note_location_to_vec(note_loc);
    //   concat(&note_sdr, notes);
    // }

    note_location_t note_sdr_bits;
    for(auto& region : notes_per_region)
      note_sdr_bits |= region;
    note_sdr = note_location_to_vec(note_sdr_bits);

    auto sparse_note_sdr = to_sparse_indices(note_sdr);
    input.setSparse(sparse_note_sdr);

    // input.setDense(note_sdr);

    input.addNoise(0.05);
    // tm.compute(input, true);
    // tm.activateDendrites();
    // tm_out = tm.cellsToColumns(tm.getPredictiveCells());
    // clsr.learn(tm_out, labels);

    clsr.learn(input, labels);
  }

  std::vector<int> infer(std::vector<note_location_t> const& notes_per_region)
  {
    // note_sdr.clear();
    // for(auto& note_loc : notes_per_region){
    //   auto notes = note_location_to_vec(note_loc);
    //   concat(&note_sdr, notes);
    // }

    note_location_t note_sdr_bits;
    for(auto& region : notes_per_region)
      note_sdr_bits |= region;
    note_sdr = note_location_to_vec(note_sdr_bits);


    auto sparse_note_sdr = to_sparse_indices(note_sdr);
    input.setSparse(sparse_note_sdr);
    // input.setDense(note_sdr);

    // tm.compute(input, false);
    // tm.activateDendrites();
    // tm_out = tm.cellsToColumns(tm.getPredictiveCells());
    // auto pdf = clsr.infer(tm_out);

    auto pdf = clsr.infer(input);

    return note_model_t::get_labels(pdf, params.pred_thresh);
  }

  void visualize()
  {
    if(note_sdr.empty())
      return;
    auto [rows, cols] = square_ish_sdr(note_sdr.size());
    auto note_mat = vector_to_mat(note_sdr, rows, cols);
    show("note_sdr", note_mat);
    // show("voting_tm", sdr1DToColorMapBySlice(tm_out));
  }

  void save(std::string models_path)
  {
    fs::create_directory(fs::path(models_path).parent_path());

    ofstream dump2(models_path+"_clsr.model", ofstream::binary | ofstream::trunc | ofstream::out);
    cereal::BinaryOutputArchive oarchive2(dump2);
    clsr.save_ar(oarchive2);
    dump2.close();

    ofstream dump3(models_path+"_tm.model", ofstream::binary | ofstream::trunc | ofstream::out);
    cereal::BinaryOutputArchive oarchive3(dump3);
    tm.save_ar(oarchive3);
    dump3.close();
  }

  bool load(std::string models_path)
  {
    if(!std::filesystem::exists(models_path+"_tm.model")){
      std::cerr << "voting_t | loading model: " << models_path << " failed! File doesn't exist";
      return false;
    }

    std::ifstream in2(models_path+"_clsr.model", std::ios_base::in | std::ios_base::binary);
    cereal::BinaryInputArchive iarchive2(in2);
    clsr.load_ar(iarchive2);
    in2.close();

    std::ifstream in3(models_path+"_tm.model", std::ios_base::in | std::ios_base::binary);
    cereal::BinaryInputArchive iarchive3(in3);
    tm.load_ar(iarchive3);
    in3.close();
    return true;
  }
};