#include "wav_to_midi.h"
#include "midi_to_wav.h"

#define CROW_JSON_USE_MAP
#include "crow.h"

using namespace std::literals;

struct av_packet_t
{
    cv::Mat carfac;
    cv::Mat input;
    cv::Mat columns;
    cv::Mat tm;
    std::vector<float> wav;
    double ts = 0; // in seconds

    static std::string mat_to_base64(cv::Mat const& img, std::string img_ext = ".png")
    {
        std::vector<uchar> img_bytes;
        cv::imencode(img_ext, img, img_bytes);
        return crow::utility::base64encode(img_bytes.data(), img_bytes.size());
    }

    crow::json::wvalue to_json_images() const
    {
        crow::json::wvalue result = {{"type", "avpacket"}};
        result["value"]["carfac"] = mat_to_base64(carfac);
        result["value"]["input"] = mat_to_base64(input);
        result["value"]["columns"] = mat_to_base64(columns);
        result["value"]["tm"] = mat_to_base64(tm);
        result["value"]["ts"] = ts;
        return result;
    }

    std::string to_wav_string() const
    {
        auto audio = wav;
        // I think values in this system are in range of 0:1, we need -1:+1
        for(auto& frame : audio)
            frame = std::clamp(frame * 2.0f - 1.0f, -1.0f, 1.0f);
        std::string audio_bin;
        audio_bin.assign(reinterpret_cast<const char*>(audio.data()), audio.size() * sizeof(float));
        return audio_bin;
    }
};

struct runner_t
{
    midi_labeler_t labeler;
    note_model_params_t params;
    note_model_t model;
    av_packet_t last_packet;
    
    uint prev_pred = 0;

    runner_t()
    {
        load_model();
    }

    void load_model()
    {
        params.models_path = "../../models/carfac_latest";
        model.setup(params);
        model.load();
    }

    void load_audio(std::vector<float> wav)
    {
        model.load_audio(wav);
        model.tm.reset();
        labeler.reset();
    }

    void load_midi(std::string file_path)
    {
        model.load_audio_file_and_notes(file_path+".wav");
        model.tm.reset();
        labeler.reset();
    }

    bool is_finished() const { return model.carfac_reader.get_render_pos() >= model.audio.total_bytes(); }
    float progress() const { return std::min(100.f, model.carfac_reader.get_render_pos() / float(model.audio.total_bytes()) * 100); }

    av_packet_t get_packet(note_image_t& note_image, cv::Mat preproc_input, uint pred_midi)
    {
        av_packet_t result;
        result.columns = sdr3DToColorMap(model.columns);
        result.tm = sdr3DToColorMap(model.outTM);
        result.input = preproc_input;
    
        draw_notes_as_keys(note_image);
        if(pred_midi > 0){
            if(prev_pred == pred_midi){
                note_image.midi.clear();
                note_image.midi.push_back(str_note_event_t::from_int(pred_midi));
                draw_notes_as_keys(note_image, note_image.mat.rows - 30);
            }
            prev_pred = pred_midi;
        }
        result.carfac = note_image.mat;
        result.wav = note_image.wav_chunk;
        result.ts = note_image.midi_ts / 1000.;

        return result;
    }

    void step(bool with_av_packet = false)
    {
        if(is_finished())
            return;
        auto note_image = model.carfac_reader.next();
        
        auto img = model.preproc_input(note_image.mat);
        auto labels = midi_to_labels(note_image.midi);
        model.feedforward(img, false);
        auto pdf = model.clsr.infer(model.outTM);
        auto pred_midi = argmax(pdf);
        if(pred_midi > 0)
            labeler.add_new(pred_midi, note_image.midi_ts);
        else
            labeler.skip();

        if(with_av_packet)
            last_packet = get_packet(note_image, img, pred_midi);
    }
};

struct messenger_t
{
    std::set<crow::websocket::connection*> clients;
    std::mutex clients_mtx;

    void add(crow::websocket::connection* client)
    {
        std::lock_guard<std::mutex> guard(clients_mtx);
        clients.insert(client);
    }

    void remove(crow::websocket::connection* client)
    {
        std::lock_guard<std::mutex> guard(clients_mtx);
        clients.erase(client);
    }

    void send_progress(float progress)
    {
        std::lock_guard<std::mutex> guard(clients_mtx);
        crow::json::wvalue msg({{"type", "progress"}, {"value", progress}});
        auto text = msg.dump();
        for(auto& client : clients){
            client->send_text(text);
            break;
        }
    }

    void send_demo_pause()
    {
        std::lock_guard<std::mutex> guard(clients_mtx);
        crow::json::wvalue msg({{"type", "demo_state"}, {"value", "paused"}});
        auto text = msg.dump();
        for(auto& client : clients){
            client->send_text(text);
            break;
        }
    }

    void send_demo_ended()
    {
        std::lock_guard<std::mutex> guard(clients_mtx);
        crow::json::wvalue msg({{"type", "demo_state"}, {"value", "ended"}});
        auto text = msg.dump();
        for(auto& client : clients){
            client->send_text(text);
            break;
        }
    }

    void send_packet(av_packet_t const& packet)
    {
        std::lock_guard<std::mutex> guard(clients_mtx);
        auto images = packet.to_json_images().dump();
        // auto audio = packet.to_wav_string();
        for(auto& client : clients){
            client->send_text(images);
            // client->send_binary(audio);
            break;
        }
    }

    void send_song(std::vector<float> const& wav)
    {
        std::lock_guard<std::mutex> guard(clients_mtx);
        av_packet_t packet;
        packet.wav = wav;
        auto audio = packet.to_wav_string();
        for(auto& client : clients){
            client->send_binary(audio);
            break;
        }
    }

    int64_t client_count()
    {
        std::lock_guard<std::mutex> guard(clients_mtx);
        return clients.size();
    }

};

void run_web_app() {
    crow::SimpleApp app;
    messenger_t messenger;
    runner_t runner;

    CROW_ROUTE(app, "/ping")([](){ return "pong"; });

    CROW_ROUTE(app, "/").methods("GET"_method)([](const crow::request& req, crow::response& res){
        res.set_header("Content-Type", "text/html");
        res.body = read_text_file("../web/app.html");
        return res.end();
    });

    CROW_ROUTE(app, "/audio_test").methods("GET"_method)([](const crow::request& req, crow::response& res){
        res.set_header("Content-Type", "text/html");
        res.body = read_text_file("../web/audio_test.html");
        return res.end();
    });

    CROW_ROUTE(app, "/wav-to-midi")
        .methods("POST"_method)
    ([&](const crow::request& req, crow::response& res){
        auto ct = req.get_header_value("Content-Type");
        if (ct != "audio/wav" && ct != "audio/x-wav" && ct != "application/octet-stream") {
            res.code = 415;
            res.write("Expected WAV body");
            return res.end();
        }
        auto wav = readWavBuffer(req.body);
        runner.load_audio(wav);
        while(!runner.is_finished()){
            runner.step();
            messenger.send_progress(runner.progress());
        }
        auto notes = runner.labeler.get_stable_notes();
        auto midi = note_events_to_midi_file(notes);

        res.set_header("Content-Type", "audio/midi");
        res.body.assign((const char*)(midi.data()), midi.size());
        return res.end();
    });

    std::atomic_bool demo_paused = false;
    std::atomic_bool stop_demo = false;
    std::atomic_int64_t packet_counter = 0;
    const int buffering_packets = 20;
    std::thread demo;
    smf::MidiFile midi_file;
    std::atomic_bool is_midi_demo = false;

    CROW_ROUTE(app, "/demo")
        .methods("POST"_method)
    ([&](const crow::request& req, crow::response& res){
        auto ct = req.get_header_value("Content-Type");
        
        is_midi_demo = (ct == "audio/midi" || ct == "audio/mid");
        if(is_midi_demo){
            std::cout << "GOT MIDI FILE! " << req.body.size() << std::endl;
            midi_file.clear();
            read_midi_from_buffer(midi_file, req.body);
            create_wav_and_labels(midi_file, ".", "midi_demo");
        }

        if (!is_midi_demo && ct != "audio/wav" && ct != "audio/x-wav" && ct != "application/octet-stream") {
            res.code = 415;
            res.write("Expected WAV body");
            return res.end();
        }

        if(demo.joinable()){
            stop_demo = true;
            demo.join();
        }
        stop_demo = false;
        demo = std::thread([&, data = req.body]{
            if(!is_midi_demo)
                runner.load_audio(readWavBuffer(data));
            else
                runner.load_midi("./midi_demo");
            demo_paused = false;
            packet_counter = 0;
            messenger.send_song(runner.model.audio.buffer);
            while(!stop_demo && !runner.is_finished()){
                runner.step(true);
                messenger.send_packet(runner.last_packet);
                if(++packet_counter > buffering_packets){
                    demo_paused = true;
                    packet_counter = 0;
                    messenger.send_demo_pause();
                    while(!stop_demo && demo_paused && messenger.client_count() > 0){
                        std::this_thread::sleep_for(10ms);
                    }
                }
            }
            messenger.send_demo_ended();
        });

        return res.end();
    });

    CROW_ROUTE(app, "/ws")
        .websocket(&app)
        .onopen([&](crow::websocket::connection& conn){
            messenger.add(&conn);
        })
        .onclose([&](crow::websocket::connection& conn, const std::string& reason, uint16_t err){
            messenger.remove(&conn);
            stop_demo = true;
        })
        .onmessage([&](crow::websocket::connection& conn, const std::string& data, bool is_binary){
            if(is_binary){
                std::cout << "skipping binary messages..." << std::endl;
            }
            else{
                try{
                    auto js = crow::json::load(data);
                    if(js["type"] == "demo_state" && js["value"] == "continue")
                        demo_paused = false;
                }
                catch(std::exception& e){
                    std::cout << "failed to parse json " << e.what() << std::endl;
                }
            }
        });

    app.port(8080).multithreaded().run();
}

int main()
{
    run_web_app();
    return 0;
}