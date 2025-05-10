/* FluidSynth Arpeggio - Sequencer API example
 *
 * This code is in the public domain.
 *
 * To compile:
 *   gcc -o fluidsynth_arpeggio -lfluidsynth fluidsynth_arpeggio.c
 *
 * To run:
 *   fluidsynth_arpeggio soundfont [steps [duration]]
 *
 * [Pedro Lopez-Cabanillas <plcl@users.sf.net>]
 */
 
#include <stdlib.h>
#include <stdio.h>
#include <fluidsynth.h>
#include <thread>
#include <iostream>
#include <random>
#include <fstream>
#include "midi_to_wav.h"

using namespace std::literals;
using pc_clock = std::chrono::high_resolution_clock;
 
fluid_synth_t *synth;
fluid_audio_driver_t *audiodriver;
fluid_sequencer_t *sequencer;
fluid_file_renderer_t* renderer;
short synth_destination, client_destination;
unsigned int time_marker;
/* duration of the pattern in ticks. */
unsigned int duration = 1440;
/* notes of the arpeggio */
// unsigned int notes[] = { 60, 64, 67, 72, 76, 79, 84, 79, 76, 72, 67, 64 };

// unsigned int notes[] = {60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70 };
// unsigned int notes[] = {40, 52, 64, 76, 88, 100}; // E2-E7

unsigned int notes[] = {84, 79, 76, 72, 67, 64, 60}; // c6-c4

/* number of notes in one pattern */
unsigned int pattern_size;
/* prototype */
void
sequencer_callback(unsigned int time, fluid_event_t *event,
                   fluid_sequencer_t *seq, void *data);
 
/* schedule a note on message */
void
schedule_noteon(int chan, short key, unsigned int ticks)
{
    fluid_event_t *ev = new_fluid_event();
    fluid_event_set_source(ev, -1);
    fluid_event_set_dest(ev, synth_destination);
    fluid_event_noteon(ev, chan, key, 127);
    fluid_sequencer_send_at(sequencer, ev, ticks, 1);
    delete_fluid_event(ev);
}
 
/* schedule a note off message */
void
schedule_noteoff(int chan, short key, unsigned int ticks)
{
    fluid_event_t *ev = new_fluid_event();
    fluid_event_set_source(ev, -1);
    fluid_event_set_dest(ev, synth_destination);
    fluid_event_noteoff(ev, chan, key);
    fluid_sequencer_send_at(sequencer, ev, ticks, 1);
    delete_fluid_event(ev);
}
 
/* schedule a timer event (shall trigger the callback) */
void
schedule_timer_event(void)
{
    fluid_event_t *ev = new_fluid_event();
    fluid_event_set_source(ev, -1);
    fluid_event_set_dest(ev, client_destination);
    fluid_event_timer(ev, NULL);
    fluid_sequencer_send_at(sequencer, ev, time_marker, 1);
    delete_fluid_event(ev);
}

/* schedule the arpeggio's notes */
std::vector<note_event_t> schedule_pattern(int64_t time_marker)
{
    std::vector<note_event_t> events;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> midi_range(22, 100); // define the range
    std::uniform_int_distribution<> note_duration_range(250, 1000);
    auto note_count = 100;
    auto channel = 6;
 
    for(auto i = 0; i < note_count; ++i)
    {
        auto note = midi_range(gen);
        schedule_noteon(channel, note, time_marker);
        events.push_back({note, time_marker, note_event_t::UP});
        time_marker += note_duration_range(gen);
        schedule_noteoff(channel, note, time_marker);
        events.push_back({note, time_marker, note_event_t::DOWN});
        time_marker += note_duration_range(gen) * 2; // pause
    }

    return events;
}

std::vector<note_event_t> schedule_main_notes(int64_t time_marker, int duration)
{
    std::vector<note_event_t> events;

    auto channel = 6;
    for(auto i = 0; i < std::size(notes); ++i)
    {
        int note = notes[i];
        schedule_noteon(channel, note, time_marker);
        events.push_back({note, time_marker, note_event_t::UP});
        time_marker += duration;
        schedule_noteoff(channel, note, time_marker);
        events.push_back({note, time_marker, note_event_t::DOWN});
        time_marker += duration; // pause
    }

    return events;
}

void
sequencer_callback(unsigned int time, fluid_event_t *event,
                   fluid_sequencer_t *seq, void *data)
{
    schedule_timer_event();
    // schedule_pattern();
}
 
void
usage(char *prog_name)
{
    printf("Usage: %s soundfont.sf2 [steps [duration]]\n", prog_name);
    printf("\t(optional) steps: number of pattern notes, from 2 to %d\n",
           pattern_size);
    printf("\t(optional) duration: of the pattern in ticks, default %d\n",
           duration);
}

void generate_midi(std::string file_name = "rnd")
{
    std::string sound_font_path = "../../sound_fonts/Steinway-Chateau-Plus-Instruments-v1.7.sf2";
    std::string dir = "../../dataset/rnd_train";

    int n;
    fluid_settings_t *settings;
    settings = new_fluid_settings();
    

    // specify the file to store the audio to
    // make sure you compiled fluidsynth with libsndfile to get a real wave file
    // otherwise this file will only contain raw s16 stereo PCM
    fluid_settings_setstr(settings, "audio.file.name", (dir + "/" + file_name + ".wav").c_str());
    
    // use number of samples processed as timing source, rather than the system timer
    fluid_settings_setstr(settings, "player.timing-source", "sample");
    
    // since this is a non-realtime scenario, there is no need to pin the sample data
    fluid_settings_setint(settings, "synth.lock-memory", 0);

    fluid_settings_setnum(settings, "synth.gain", 0.6);


    pattern_size = sizeof(notes) / sizeof(int);
 
   
    /* create the synth, driver and sequencer instances */
    synth = new_fluid_synth(settings);
    /* load a SoundFont */
    n = fluid_synth_sfload(synth, sound_font_path.c_str(), 1);

    renderer = new_fluid_file_renderer(synth);


    if(n != -1)
    {
        sequencer = new_fluid_sequencer2(0);
        /* register the synth with the sequencer */
        synth_destination = fluid_sequencer_register_fluidsynth(sequencer,
                            synth);

        
        // /* register the client name and callback */
        // client_destination = fluid_sequencer_register_client(sequencer,
        //                     "arpeggio", sequencer_callback, NULL);

        // audiodriver = new_fluid_audio_driver(settings, synth);

        /* get the current time in ticks */
        time_marker = fluid_sequencer_get_tick(sequencer);

        /* schedule patterns */
        auto start = time_marker;
        std::vector<note_event_t> note_events;
        for(auto i = 0; i < 1; i++){
            // auto events = schedule_pattern(start);
            auto events = schedule_main_notes(start, 300);
            for(auto& note : events)
                note_events.push_back(note);
        }
        std::sort(note_events.begin(), note_events.end(), [](auto e1, auto e2){ return e1.time_point < e2.time_point; });
        time_marker = note_events.back().time_point;
        save_notes(note_events, dir + "/" + file_name + ".csv");

        std::cout << "time marker?? " << time_marker << std::endl;

        while (fluid_file_renderer_process_block(renderer) == FLUID_OK && fluid_sequencer_get_tick(sequencer) <= time_marker){
            // std::cout << "time tick " << fluid_sequencer_get_tick(sequencer) << std::endl;
            continue;
        }
    }

    /* clean and exit */
    delete_fluid_audio_driver(audiodriver);
    delete_fluid_sequencer(sequencer);
    delete_fluid_file_renderer(renderer);
    delete_fluid_synth(synth);
 
    delete_fluid_settings(settings);
}
 
int
main(int argc, char *argv[])
{
    generate_midi("arpegio_test_c6_c4");
    return 0;
}