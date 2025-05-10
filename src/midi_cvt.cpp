#include <midifile/MidiFile.h>
#include <midifile/Options.h>
#include <iostream>
#include <fluidsynth.h>
#include "midi_to_wav.h"

int main(int argc, char** argv) {
    smf::Options options;
    options.process(argc, argv);

    smf::MidiFile midi_file;
    if (options.getArgCount() == 0)
        midi_file.read(std::cin);
    else
        midi_file.read(options.getArg(1));

    midi_file.doTimeAnalysis();
    midi_file.linkNotePairs();

    create_wav_and_labels(midi_file, ".", "midi_test");

    return 0;
}