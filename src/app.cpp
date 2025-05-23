#include "wav_to_midi.h"
#include "midi_to_wav.h"
#include "tbt_model.h"

#define CROW_JSON_USE_MAP
#include "crow.h"

using namespace std::literals;

struct av_packet_t
{
    cv::Mat sai;
    cv::Mat activations;
    cv::Mat voting;

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
        result["value"]["sai"] = mat_to_base64(sai);
        result["value"]["activations"] = mat_to_base64(activations);
        result["value"]["voting"] = mat_to_base64(voting);
        result["value"]["ts"] = ts;
        return result;
    }

    std::string to_wav_string() const
    {
        auto audio = wav;
        std::string audio_bin;
        audio_bin.assign(reinterpret_cast<const char*>(audio.data()), audio.size() * sizeof(float));
        return audio_bin;
    }
};

struct runner_t
{
    midi_labeler_t labeler;
    av_packet_t last_packet;

    std::map<std::string, ptr<tbt_model_t>> models;
    std::string current_model;

    runner_t()
    {
        load_model();
    }

    void load_model()
    {
        auto models_dir = "../../stable_models"s;
        for(auto model_dir : fs::directory_iterator(models_dir)){
            auto model_name = fs::path(model_dir).stem();
            if(model_name != "many_eyes")
                continue;
            if(model_dir.path().stem().string().starts_with("."))
                continue;
            auto model = std::make_shared<tbt_model_t>();
            model->params.core.models_path = model_dir.path();
            model->loadv2();
            models[model_name] = model;
        }

        if(models.empty()){
            std::cerr << "Failed to load models from: " << models_dir << "!!!" << std::endl;
            std::exit(1);
        }

        current_model = models.begin()->first;
    }

    std::vector<std::string> get_model_names() const
    {
        std::vector<std::string> result;
        for(auto& [key, v] : models)
            result.push_back(key);
        return result;
    }

    void load_audio(std::vector<float> wav)
    {
        auto model = models.at(current_model);
        model->core.load_audio(wav);
        model->reset_tms();
        labeler.reset();
    }

    void load_midi(std::string file_path)
    {
        auto model = models.at(current_model);
        model->core.load_audio_file_and_notes(file_path+".wav");
        model->reset_tms();
        labeler.reset();
    }

    bool is_finished() const {
        auto model = models.at(current_model);
        return model->core.carfac_reader.get_render_pos() >= model->core.audio.total_bytes();
    }
    float progress() const {
        auto model = models.at(current_model);
        return std::min(100.f, model->core.carfac_reader.get_render_pos() / float(model->core.audio.total_bytes()) * 100);
    }

    av_packet_t get_packet(note_image_t& note_image, std::vector<int> pred_midi)
    {
        auto model = models.at(current_model);

        av_packet_t result;
        result.activations = model->get_activations_image();
        result.voting = model->voting.get_voting_image();

        draw_notes_as_keys(note_image);
        model->draw_regions(note_image);
        if(!pred_midi.empty() && pred_midi != std::vector<int>{0}){
            note_image.midi.clear();
            for(auto& note : pred_midi)
                note_image.midi.push_back(str_note_event_t::from_int(note));
            draw_notes_as_keys(note_image, note_image.mat.rows - 30);
        }
        result.sai = note_image.mat;

        result.wav = note_image.wav_chunk;
        result.ts = note_image.midi_ts / 1000.;

        return result;
    }

    void step(bool with_av_packet = false)
    {
        if(is_finished())
            return;

        auto model = models.at(current_model);
        auto note_image = model->core.carfac_reader.next();

        std::vector<int> pred_midi;

        if(!model->params.use_voting_tm)
            pred_midi = model->infer(note_image);
        else
            pred_midi = model->infer_voting(note_image);
        
        if(!pred_midi.empty()){
            for(auto note : pred_midi)
                labeler.add_new(note, note_image.midi_ts);
        }
        else
            labeler.skip();

        if(with_av_packet)
            last_packet = get_packet(note_image, pred_midi);
    }

    std::vector<float> get_full_audio()
    {
        auto model = models.at(current_model);
        return model->core.audio.buffer;
    }

    void reset()
    {
        for(auto [model_name, model] : models)
            model->reset();
        labeler.reset();
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

    CROW_ROUTE(app, "/model_names").methods("GET"_method)([&](const crow::request& req, crow::response& res){
        res.set_header("Content-Type", "application/json");
        crow::json::wvalue v;
        v = runner.get_model_names();
        res.body = v.dump();
        return res.end();
    });

    CROW_ROUTE(app, "/get_current_model").methods("GET"_method)([&](const crow::request& req, crow::response& res){
        res.set_header("Content-Type", "application/json");
        crow::json::wvalue v;
        v = runner.current_model;
        res.body = v.dump();
        return res.end();
    });

    CROW_ROUTE(app, "/set_model").methods("GET"_method)([&](const crow::request& req, crow::response& res){
        try{
            auto js = crow::json::load(req.body);
            auto model_name = js.s();
            if(!runner.models.contains(model_name)){
                res.code = 415;
                res.write("Expected valid model name");
            }
            else{
                runner.reset();
                runner.current_model = model_name;
            }
        }
        catch(std::exception& e){
            res.code = 415;
            res.write("Expected valid model name");
        }
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
    const int buffering_packets = int(2.5 / (1024./44100.)) * 3;
    std::thread demo;
    smf::MidiFile midi_file;
    std::atomic_bool is_midi_demo = false;

    CROW_ROUTE(app, "/demo")
        .methods("POST"_method)
    ([&](const crow::request& req, crow::response& res){
        auto ct = req.get_header_value("Content-Type");
        
        is_midi_demo = (ct == "audio/midi" || ct == "audio/mid");
        if(is_midi_demo){
            auto gain = 2.f;
            if(req.headers.contains("audio-gain"))
                gain = std::stof(req.headers.find("audio-gain")->second);
            std::cout << "GOT MIDI FILE! " << req.body.size() << " gain: " << gain << std::endl;
            midi_file.clear();
            read_midi_from_buffer(midi_file, req.body);
            create_wav_and_labels(midi_file, ".", "midi_demo", gain);
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
            runner.reset();
            if(!is_midi_demo)
                runner.load_audio(readWavBuffer(data));
            else
                runner.load_midi("./midi_demo");
            demo_paused = false;
            packet_counter = 0;

            messenger.send_song(runner.get_full_audio());
            while(!stop_demo && !runner.is_finished()){
                using namespace std::chrono;
                auto start = high_resolution_clock::now();
                runner.step(true);
                messenger.send_packet(runner.last_packet);
                if(++packet_counter > buffering_packets){
                    auto end = high_resolution_clock::now();
                    auto elapsed = duration_cast<milliseconds>(end - start).count();
                    std::cout << "it took: " << elapsed << " ms ";
                    std::cout << "to generate: " << buffering_packets << " packets ";
                    std::cout << "last_ts: " << runner.last_packet.ts << std::endl;
                    demo_paused = true;
                    packet_counter = 0;
                    messenger.send_demo_pause();
                    while(!stop_demo && demo_paused && messenger.client_count() > 0){
                        std::this_thread::sleep_for(1ms);
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