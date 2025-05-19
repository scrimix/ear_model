#pragma once

#include "note_model.h"
#include "vector_buf.h"
#include <midifile/MidiFile.h>
#include <fluidsynth.h>
#include <vector>
#include <string>
#include <fstream>

struct note_event_t {
    int midi_value = 0;
    int64_t time_point = 0;
    enum {UP, DOWN} pos = UP;
};

inline void save_notes(std::vector<note_event_t> note_events, std::string const& file_path)
{
    std::string notes[]={"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    std::ofstream file(file_path);
    for(auto& note : note_events){
        auto octave = note.midi_value / 12 - 1;
        auto note_name = notes[note.midi_value % 12];
        file << note_name << octave << ",";
        file << note.time_point << ",";
        file << (note.pos == note_event_t::UP ? "UP" : "DOWN");
        file << "\n";
    }
    file.close();
}

struct fluid_synth_ctx_t
{
    fluid_synth_t* synth;
    fluid_file_renderer_t* renderer;
    fluid_sequencer_t* sequencer;
    short synth_destination;
};

/* schedule a note on message */
inline void schedule_noteon(fluid_synth_ctx_t ctx, int chan, short key, unsigned int ticks, unsigned int velocity = 127)
{
    fluid_event_t *ev = new_fluid_event();
    fluid_event_set_source(ev, -1);
    fluid_event_set_dest(ev, ctx.synth_destination);
    fluid_event_noteon(ev, chan, key, velocity);
    fluid_sequencer_send_at(ctx.sequencer, ev, ticks, 1);
    delete_fluid_event(ev);
}

/* schedule a note off message */
inline void schedule_noteoff(fluid_synth_ctx_t ctx, int chan, short key, unsigned int ticks)
{
    fluid_event_t *ev = new_fluid_event();
    fluid_event_set_source(ev, -1);
    fluid_event_set_dest(ev, ctx.synth_destination);
    fluid_event_noteoff(ev, chan, key);
    fluid_sequencer_send_at(ctx.sequencer, ev, ticks, 1);
    delete_fluid_event(ev);
}

inline void read_midi_from_buffer(smf::MidiFile& midi_file, std::string const& bin_data)
{
    std::istringstream stream(bin_data);
    midi_file.read(stream);
    midi_file.doTimeAnalysis();
    midi_file.linkNotePairs();
}

inline void create_wav_and_labels(smf::MidiFile& midi_file, std::string dir, std::string file_name, float gain)
{
    std::string sound_font_path = "../../sound_fonts/Steinway-Chateau-Plus-Instruments-v1.7.sf2";

    int n = 0;
    auto settings = new_fluid_settings();
    fluid_settings_setstr(settings, "audio.file.name", (dir + "/" + file_name + ".wav").c_str());
    // use number of samples processed as timing source, rather than the system timer
    fluid_settings_setstr(settings, "player.timing-source", "sample");
    // since this is a non-realtime scenario, there is no need to pin the sample data
    fluid_settings_setint(settings, "synth.lock-memory", 0);
    fluid_settings_setnum(settings, "synth.gain", gain);

    fluid_synth_ctx_t ctx;
   
    /* create the synth, driver and sequencer instances */
    ctx.synth = new_fluid_synth(settings);
    n = fluid_synth_sfload(ctx.synth, sound_font_path.c_str(), 1); // load sound font
    if(n == -1){
        std::cerr << "Failed to load audio font! " << sound_font_path << std::endl;
        delete_fluid_synth(ctx.synth);
        delete_fluid_settings(settings);
        return;
    }

    ctx.renderer = new_fluid_file_renderer(ctx.synth);
    ctx.sequencer = new_fluid_sequencer2(0);
    ctx.synth_destination = fluid_sequencer_register_fluidsynth(ctx.sequencer, ctx.synth);

    // iterate over midi events, schedule and save to file
    std::vector<note_event_t> notes;
    int tracks = midi_file.getTrackCount();
    auto channel = 6;
    int64_t last_tick = 0;
    for (int track = 0; track < tracks; track++) {
        for (int event_idx=0; event_idx < midi_file[track].size(); event_idx++) {
            auto& event = midi_file[track][event_idx];
            if(event.isNote()){
                auto event_type = (event.isNoteOn() ? note_event_t::UP : note_event_t::DOWN);
                int64_t ts = (event.tick / double(midi_file.getTicksPerQuarterNote())) * 1000 / 1.6667;
                notes.push_back({event.getKeyNumber(), ts, event_type});
                if(event.isNoteOn())
                    schedule_noteon(ctx, channel, event.getKeyNumber(), ts, event.getVelocity());
                else
                    schedule_noteoff(ctx, channel, event.getKeyNumber(), ts);
            }
        }
    }
    std::sort(notes.begin(), notes.end(), [](auto e1, auto e2){ return e1.time_point < e2.time_point; });
    save_notes(notes, dir + "/" + file_name + ".csv");
    if(notes.empty())
        std::cerr << "empty notes!! "  << std::endl;
    else
        last_tick = notes.back().time_point;

    while (fluid_file_renderer_process_block(ctx.renderer) == FLUID_OK && fluid_sequencer_get_tick(ctx.sequencer) < last_tick){
        // std::cout << "time tick " << fluid_sequencer_get_tick(ctx.sequencer) << " / " << last_tick  << std::endl;
        continue;
    }

    /* clean and exit */
    delete_fluid_sequencer(ctx.sequencer);
    delete_fluid_file_renderer(ctx.renderer);
    delete_fluid_synth(ctx.synth);
    delete_fluid_settings(settings);
}

inline bool check_and_gen_if_midi(std::string file)
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