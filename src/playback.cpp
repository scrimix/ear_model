#include "carfac_reader.h"


int main(int argc, char *argv[]) {
    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }
    signal(SIGINT, exit_on_ctrl_c);

    auto sample_rate = 44100;
    auto buffer_size = 1024;

    // std::string file_path = "/Users/scrimix/Music/lofi.wav";
    // auto loudness_coef = 100;

    // std::string file_path = "../../sound_data/rnd_single.wav";
    // auto loudness_coef = 70;

    std::string file_path = "../../dataset/tests/arpegio_test_c6_c4.wav";
    auto loudness_coef = 80;

    carfac_reader_t carfac_reader;
    carfac_reader.init(file_path);
    carfac_reader.set(sample_rate, buffer_size, loudness_coef);

    AudioData audio = {read_wav(file_path, sample_rate)};

    // Set up audio specifications
    SDL_AudioSpec spec;
    spec.freq = sample_rate;         // Sample rate
    spec.format = AUDIO_F32; // 16-bit PCM ?? float32
    spec.channels = 1;         // Mono
    spec.samples = buffer_size;
    spec.callback = audio_callback;
    spec.userdata = &audio;

    // Open the audio device
    if (SDL_OpenAudio(&spec, NULL) < 0) {
        fprintf(stderr, "SDL_OpenAudio Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    // Start playback
    SDL_PauseAudio(false);

    // Wait for playback to finish
    while (audio.get_pos() < audio.total_bytes()) {
        while(audio.get_pos() - carfac_reader.get_render_pos() > 0){
            auto note_image = carfac_reader.next();
            if(!note_image.midi.empty())
                std::cout << "note? " << note_image.midi.front().to_midi_int() << std::endl;

            draw_notes(note_image);

            // cv::resize(note_image.mat, note_image.mat, cv::Size(64,64));
            // cv::cvtColor(note_image.mat, note_image.mat, cv::COLOR_BGR2GRAY);
            // cv::threshold(note_image.mat, note_image.mat, 80, 255, cv::THRESH_BINARY);

            cv::namedWindow("nap", 2);
            cv::imshow("nap", note_image.mat);
            cv::waitKey(1);
        }
        SDL_Delay(1);
    }
    std::cout << "done? " << audio.get_pos() << " " << audio.total_bytes() << std::endl;

    // Clean up
    SDL_CloseAudio();
    SDL_Quit();

    return 0;
}
