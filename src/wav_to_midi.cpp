#include "wav_to_midi.h"
#include "vector_buf.h"
#include "crow.h"

void generate_labels(midi_labeler_t& labeler)
{
    note_model_params_t params;
    params.models_path = "../../models/carfac_latest";

    note_model_t model;
    model.setup(params);
    model.load();

    // model.load_audio(read_wav("/Users/scrimix/Music/my_piano_test.wav"));
    // model.load_audio(read_wav("../../dataset/rnd_multi/rnd_single_1.wav"));
    // auto wav = read_wav("../../dataset/tests/arpegio_test_c6_c4.wav");
    auto wav = read_wav("../../dataset/warmup/random_speech.wav");
    auto stable_notes = detect_notes(model, labeler, wav);
    for(auto event : stable_notes)
        std::cout << event << std::endl;
}

std::vector<uchar> convert_wav_to_midi(std::vector<float> const& wav)
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

int main()
{
    // midi_labeler_t labeler;
    // generate_labels(labeler);
    // labeler.dump("speech.csv");
    // return 0;

    // labeler.read("my_piano_labels.csv");

    // auto stable_notes = labeler.get_stable_notes();
    // for(auto& midi_event : stable_notes)
    //     std::cout << midi_event << std::endl;

}