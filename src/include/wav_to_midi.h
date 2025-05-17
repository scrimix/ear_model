#pragma once

#include "note_model.h"
#include "vector_buf.h"
#include <midifile/MidiFile.h>

struct midi_tick_t
{
    int64_t index = 0;
    int64_t ts = 0;
};

struct midi_event_t
{
    int midi_value = 0;
    std::vector<midi_tick_t> ticks;
};

inline std::ostream& operator<<(std::ostream& os, midi_event_t const& midi_event)
{
    auto str_event = str_note_event_t::from_int(midi_event.midi_value);
    os << "note(" << str_event.note_name << ", " << midi_event.midi_value;
    os << ", " << "from_ts: " << midi_event.ticks.front().ts << " to_ts: " << midi_event.ticks.back().ts;
    os << " [" << midi_event.ticks.front().index << ":" << midi_event.ticks.back().index << "]" << ")";
    return os;
}

class midi_labeler_t
{
public:
    int64_t current_index = 0;
    std::vector<midi_event_t> all_midi_events;

    void add_new(uint label, int64_t midi_ts)
    {
        all_midi_events.push_back({int(label), {{current_index, midi_ts}}});
        current_index++;
    }

    void skip()
    {
        current_index++;
    }

    void dump(std::string file_path)
    {
        std::fstream out(file_path, ofstream::trunc | ofstream::out);
        for(auto event : all_midi_events){
            if(event.ticks.size() != 1)
                throw std::runtime_error("oops, midi_event expecting only one tick!");
            auto tick = event.ticks.at(0);
            out << event.midi_value << "," << tick.index << "," << tick.ts << "\n";
        }
        out.close();
    }

    void read(std::string file_path)
    {
        std::ifstream file(file_path);
        all_midi_events.clear();
        std::string line;
        while(std::getline(file, line)){
            auto parts = split(line, ",");
            midi_event_t event;
            event.midi_value = std::stoll(parts[0]);
            auto index = std::stoll(parts[1]);
            auto ts = std::stoll(parts[2]);
            event.ticks.push_back({index, ts});
            all_midi_events.push_back(event);
        }
        current_index = all_midi_events.back().ticks.back().index;
    }

    std::vector<midi_event_t> get_stable_notes()
    {
        std::vector<midi_event_t> resulting_notes;
        std::vector<midi_event_t> active_notes;

        auto prune_active = [&](int64_t current_index)
        {
            auto pruning_thresh = 3;
            for(auto it = active_notes.begin(); it != active_notes.end();){
                if(current_index - it->ticks.back().index > pruning_thresh){
                    auto total_life_span = (it->ticks.back().index - it->ticks.front().index);
                    auto stability = total_life_span / float(it->ticks.size());
                    if(stability > 0.3 && total_life_span > 3){
                        resulting_notes.push_back(*it);
                    }
                    it = active_notes.erase(it);
                }
                else{
                    it++;
                }
            }
        };

        for(auto event : all_midi_events){
            prune_active(event.ticks.back().index);
            auto same_note = [&](auto active_note) { return active_note.midi_value == event.midi_value; };
            auto it = std::find_if(active_notes.begin(), active_notes.end(), same_note);
            if(it != active_notes.end())
                it->ticks.push_back({event.ticks.back().index, event.ticks.back().ts});
            else
                active_notes.push_back(event);
        }
        prune_active(9999*9999);

        return resulting_notes;
    }

    void reset()
    {
        current_index = 0;
        all_midi_events.clear();
    }
};

inline std::vector<midi_event_t> detect_notes(note_model_t& model, midi_labeler_t& labeler, std::vector<float> const& wav)
{
    model.load_audio(wav);
    model.tm.reset();
    while(model.carfac_reader.get_render_pos() < model.audio.total_bytes()){
      auto note_image = model.carfac_reader.next();
      
      auto img = model.preproc_input(note_image.mat);
      auto labels = midi_to_labels(note_image.midi);
      model.feedforward(img, {0}, false);
      auto pdf = model.clsr.infer(model.outTM);
      auto pred_midi = argmax(pdf);
      if(pred_midi > 0)
        labeler.add_new(pred_midi, note_image.midi_ts);
      else
        labeler.skip();

    //   model.visualize(note_image, img, pred_midi);
    //   cv::waitKey(100);
    }

    return labeler.get_stable_notes();
}

inline std::vector<uchar> note_events_to_midi_file(std::vector<midi_event_t> const& notes)
{
    smf::MidiFile midifile;
    int track = 0;
    int channel = 0;
    int instr = 0;
    int velocity = 100;
    midifile.addTimbre(track, 0, channel, instr);
    int tpq = midifile.getTPQ();
    float offset = 4.2; // dunno, might be something to do with TPQ and difference from fluidsynth ts
    for(auto event : notes){
        midifile.addNoteOn(track, event.ticks.front().ts / offset, channel, event.midi_value, velocity);
        midifile.addNoteOff(track, event.ticks.back().ts / offset, channel, event.midi_value);
    }
    midifile.sortTracks();

    std::vector<uchar> result;
    VectorStreamBuf vec_buf(result);
    std::ostream vec_writer(&vec_buf);
    midifile.write(vec_writer);

    return result;
}

inline std::vector<uchar> convert_wav_to_midi(std::vector<float> const& wav)
{
    midi_labeler_t labeler;
    note_model_params_t params;
    params.models_path = "../../models/carfac_latest";

    note_model_t model;
    model.setup(params);
    model.load();

    auto notes = detect_notes(model, labeler, wav);
    auto midi_file = note_events_to_midi_file(notes);

    return midi_file;
}