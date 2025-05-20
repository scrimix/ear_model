#pragma once
#include "tbt_model.h"
#include "region_split.h"

static tbt_params_t fenrir = []() -> tbt_params_t {
    tbt_params_t result;
    result.core.models_path = "fenrir";
    result.core.with_note_location = false;
    result.core.with_tm = true;
    result.regions = {cv::Rect(0,0,800,600)};
    result.train_dirs = { "train/rnd_train",  "train/rnd_train" };
    return result;
}();

static tbt_params_t brainiac = []() -> tbt_params_t {
  tbt_params_t result;
  result.core.models_path = "brainiac";
  result.core.with_tm = true;
  result.core.with_note_location = true;
  result.core.height = result.core.height + int(round(sqrt(note_location_resolution) / 2));
  result.core.tm_memory = 30;
  result.core.column_count = 4;
  result.core.tm_cell_per_column = 4;
  result.regions = basic_regions();
  result.train_dirs = { "train/midi_train", "train/rnd_train", "train/rnd_multi"  };
  return result;
}();

static tbt_params_t deep_eye = []() -> tbt_params_t {
    tbt_params_t result;
    result.core.models_path = "deep_eye";
    result.core.with_note_location = false;
    result.core.with_tm = false;
    result.core.height = 48;
    result.core.width = 48;
    result.core.column_count = 16;
    result.regions = {cv::Rect(0,0,800,600)};
    result.train_dirs = { "train/rnd_train",  "train/rnd_train" };
    return result;
}();

static tbt_params_t deep_eye2 = []() -> tbt_params_t {
    tbt_params_t result;
    result.core.models_path = "deep_eye2";
    result.core.with_note_location = false;
    result.core.with_tm = false;
    result.core.height = 48;
    result.core.width = 48;
    result.core.column_count = 22;
    result.regions = {cv::Rect(0,0,800,600)};
    result.train_dirs = { "train/rnd_train", "train/rnd_multi", "train/midi_train"  };
    return result;
}();

static tbt_params_t many_eyes = []() -> tbt_params_t {
    tbt_params_t result;
    result.core.models_path = "many_eyes";
    result.core.with_note_location = false;
    result.core.with_tm = false;
    result.core.height = 32;
    result.core.width = 32;
    result.core.column_count = 16;
    result.vote_repeats = 2;
    result.use_voting_tm = true;
    result.pred_thresh = 0.099;
    result.voting_params.cell_count = 8;
    result.regions = basic_regions();
    concat(&result.regions, more_regions());
    concat(&result.regions, generateFoveatedRegions(cv::Size(800,600), 8));
    result.train_dirs = { "train/rnd_train", "train/rnd_multi", "train/midi_train"  };
    result.voting_dirs = { "train/rnd_train", "train/rnd_multi", "train/midi_train"  };
    return result;
}();