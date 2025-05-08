#include "wav_to_midi.h"
#include "crow.h"

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

void run_web_app() {
    crow::SimpleApp app;
    CROW_ROUTE(app, "/ping")([](){ return "pong"; });

    CROW_ROUTE(app, "/convert")
        .methods("POST"_method)
    ([](const crow::request& req, crow::response& res){
        auto ct = req.get_header_value("Content-Type");
        if (ct != "audio/wav" && ct != "application/octet-stream") {
            res.code = 415;
            res.write("Expected WAV body");
            return res.end();
        }

        auto wav = decodeWavToMonoFloats(req.body, 44100.f);
        auto midi = convert_wav_to_midi(wav);

        res.set_header("Content-Type", "audio/midi");
        res.body.assign((const char*)(midi.data()), midi.size());
        return res.end();
    });

    app.port(8080).multithreaded().run();
}

int main()
{
    // run_web_app();

    auto wav = read_wav("../../my_piano_test_stereo.wav");
    auto midi = convert_wav_to_midi(wav);
    // save_buf_to_file(midi, "../../test_output.mid");

    return 0;
}