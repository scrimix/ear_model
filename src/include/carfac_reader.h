#pragma once

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <filesystem>
#include <iostream>
#include <cassert>
#include <mutex>
#include <carfac/carfac.h>
#include <carfac/pitchogram_pipeline.h>
#include <carfac/image.h>
#include <iostream>
#include <filesystem>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <signal.h>
#include <thread>
#include <fstream>
#include "helpers.h"
#include "wav_reader.h"

using namespace std::literals;

// Structure to hold audio data
struct AudioData {
    std::vector<float> buffer;

    std::mutex pos_mut;
    int play_pos = 0;

    int get_pos(){  std::lock_guard lock(pos_mut); return play_pos; }
    void inc_pos(int by) { std::lock_guard lock(pos_mut); play_pos += by; }
    
    int64_t total_bytes() const { return buffer.size() * sizeof(float); }
};

// Audio callback function
inline void audio_callback(void *userdata, Uint8 *stream, int len) {
    AudioData *audio = (AudioData *)userdata;

    if (audio->get_pos() >= audio->total_bytes()) {
        memset(stream, 0, len);  // Fill buffer with silence if we are done
        return;
    }

    // Copy audio data to stream
    Uint32 remaining = (audio->total_bytes() - audio->get_pos());
    Uint32 bytes_to_copy = ((len > remaining) ? remaining : len);

    SDL_memcpy(stream, ((uint8_t*)audio->buffer.data()) + audio->get_pos(), bytes_to_copy);
    audio->inc_pos(bytes_to_copy);

    // If less data was copied, fill the rest with silence
    if (bytes_to_copy < len) {
        memset(stream + bytes_to_copy, 0, len - bytes_to_copy);
    }

    // std::this_thread::sleep_for(100ms);

    // std::cout << "playback: "  << audio->get_pos() << std::endl;
}

struct str_note_event_t
{
    std::string note_name;
    int64_t time_point = 0;
    std::string pos;
    int octave() { return note_name.back() - '0'; }
    int to_midi_int() const;
    static str_note_event_t from_int(int midi_value);
};

inline str_note_event_t str_note_event_t::from_int(int midi_value) {
    std::string notes[]={"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    auto octave = midi_value / 12 - 1;
    auto note_name = notes[midi_value % 12];
    std::stringstream ss;
    ss << note_name << octave;

    str_note_event_t result;
    result.note_name = ss.str();
    return result;
}

inline int str_note_event_t::to_midi_int() const {
    auto note = note_name;
    std::unordered_map<std::string, int> noteMap = {
        {"C", 0}, {"C#", 1}, {"Db", 1},
        {"D", 2}, {"D#", 3}, {"Eb", 3},
        {"E", 4}, {"Fb", 4}, {"E#", 5},
        {"F", 5}, {"F#", 6}, {"Gb", 6},
        {"G", 7}, {"G#", 8}, {"Ab", 8},
        {"A", 9}, {"A#", 10}, {"Bb", 10},
        {"B", 11}, {"Cb", 11}, {"B#", 0}
    };

    std::string pitch;
    int octave;

    // Extract pitch and octave
    if (note.length() >= 2 && std::isalpha(note[0])) {
        if (note[1] == '#' || note[1] == 'b') {
            pitch = note.substr(0, 2);
            octave = std::stoi(note.substr(2));
        } else {
            pitch = note.substr(0, 1);
            octave = std::stoi(note.substr(1));
        }
    } else {
        throw std::invalid_argument("Invalid note format");
    }

    if (noteMap.find(pitch) == noteMap.end())
        throw std::invalid_argument("Invalid pitch name");

    int midiNumber = 12 * (octave + 1) + noteMap[pitch];

    if (midiNumber < 0 || midiNumber > 127)
        throw std::out_of_range("MIDI note out of valid range (0-127)");

    return midiNumber;
}

inline std::vector<str_note_event_t> read_notes(std::string const& file_path)
{
    std::vector<str_note_event_t> result;
    std::ifstream file(file_path);

    std::string line;
    while(std::getline(file, line)){
        str_note_event_t note_event;
        auto parts = split(line, ",");
        note_event.note_name = parts[0];
        note_event.time_point = std::stoll(parts[1]);
        note_event.pos = parts[2];
        result.push_back(note_event);
    }

    return result;
}

inline std::vector<uint32_t> midi_to_labels(std::vector<str_note_event_t> midi)
{
  std::vector<uint32_t> labels;
    for(auto& note : midi)
      labels.push_back(note.to_midi_int());
  return labels;
}

class active_notes_t
{
public:
    std::vector<str_note_event_t> all_notes;
    std::vector<str_note_event_t> current_notes;
    std::vector<std::pair<int64_t,str_note_event_t>> fading_notes;
    
    int64_t current_pos = -1;
    int64_t echo_fade = 150;
    
    void advance(int64_t new_pos)
    {
        std::vector<str_note_event_t> new_events;
        for(auto& note : all_notes){
            if(note.time_point > current_pos && note.time_point <= new_pos){
                new_events.push_back(note);
            }
            if(note.time_point > new_pos)
                break;
        }
        for(auto& new_event : new_events){
            if(new_event.pos == "DOWN"){
                for(auto it = current_notes.begin(); it != current_notes.end(); it++){
                    if(it->note_name == new_event.note_name){
                        fading_notes.push_back({new_pos+echo_fade,*it});
                        current_notes.erase(it);
                        break;
                    }
                }
            }
            else{
                current_notes.push_back(new_event);
            }
        }
        for(auto it = fading_notes.begin(); it != fading_notes.end();){
            if(it->first < new_pos)
                it = fading_notes.erase(it);
            else
                it++;
        }
        current_pos = new_pos;
    }

    std::vector<str_note_event_t> get()
    {
        std::vector<str_note_event_t> result;
        result.insert(result.end(), current_notes.begin(), current_notes.end());
        for(auto& [timeout, note] : fading_notes)
            result.push_back(note);
        return result;
    }

    void reset()
    {
        current_notes.clear();
        fading_notes.clear();
        current_pos = -1;
    }
};


struct note_image_t {
    cv::Mat mat;
    std::vector<str_note_event_t> midi;
    std::vector<float> wav_chunk;
    int64_t midi_ts = 0;
    bool is_valid() const { return !mat.empty(); }
};

class carfac_reader_t {
public:
    void init(std::string file_path);
    void init(std::vector<float> const& wav);
    void set(int sample_rate, int buffer_size, int loudness_coef);
    int64_t get_render_pos() const;
    note_image_t next();
    void reset();
    int64_t total_note_count() const;

private:

    int sample_rate = 44100;
    int buffer_size = 1024;
    int loudness_coef = 70;
    int64_t render_pos = 0;
    std::vector<float> sample_data;
    active_notes_t active_notes;
    PitchogramPipeline* pipeline = nullptr;
};

inline int64_t carfac_reader_t::total_note_count() const
{
    return active_notes.all_notes.size();
}

inline void carfac_reader_t::set(int sample_rate_arg, int buffer_size_arg, int loudness_coef_arg)
{
    sample_rate = sample_rate_arg;
    buffer_size = buffer_size_arg;
    loudness_coef = loudness_coef_arg;
}

inline void carfac_reader_t::init(std::string file_path)
{
    if(pipeline){
        delete pipeline;
        pipeline = 0;
    }
    sample_data = readWavFile(file_path);
    std::string notes_path = replaced(file_path, ".wav", ".csv");
    if(std::filesystem::exists(notes_path))
        active_notes.all_notes = read_notes(notes_path);

    render_pos = 0;
    PitchogramPipelineParams params;
    params.num_frames = sample_data.size() / buffer_size;
    params.num_samples_per_segment = buffer_size;
    params.pitchogram_params.light_color_theme = false;
    pipeline = new PitchogramPipeline(sample_rate, params);
}

inline void carfac_reader_t::init(std::vector<float> const& wav)
{
    if(pipeline){
        delete pipeline;
        pipeline = 0;
    }
    sample_data = wav;
    render_pos = 0;
    PitchogramPipelineParams params;
    params.num_frames = sample_data.size() / buffer_size;
    params.num_samples_per_segment = buffer_size;
    params.pitchogram_params.light_color_theme = false;
    pipeline = new PitchogramPipeline(sample_rate, params);
}

inline note_image_t carfac_reader_t::next()
{
    float input[buffer_size];
    auto bytes_to_copy = buffer_size * sizeof(float);

    std::memcpy(input, (uint8_t*)sample_data.data() + render_pos, bytes_to_copy);
    for(auto i = 0; i < buffer_size; i++)
        input[i] /= loudness_coef; // adjusting volume for algorithms
    pipeline->ProcessSamples(input, buffer_size);
    auto& nap = pipeline->sai_output().transpose().eval();
    // auto& nap = pipeline.carfac_output().nap()[0].transpose().eval();
    cv::Mat mat(nap.rows(), nap.cols(), CV_32F, (void*)nap.data());
    cv::Mat rot_mat;
    cv::rotate(mat, rot_mat, cv::ROTATE_90_CLOCKWISE);
    cv::resize(rot_mat, rot_mat, cv::Size(800, 600));
    cv::cvtColor(rot_mat, rot_mat, cv::COLOR_GRAY2BGR);
    rot_mat.convertTo(rot_mat, CV_8U, 255);

    note_image_t result;
    result.mat = rot_mat;
    result.midi = active_notes.get();
    result.wav_chunk.assign(input, input + buffer_size);

    // advance bytes rendered
    render_pos += bytes_to_copy;

    // update active notes based on tick
    int64_t current_midi_ts = float(render_pos) / sizeof(float) / sample_rate * 1000;
    active_notes.advance(current_midi_ts);
    result.midi_ts = current_midi_ts;
    
    return result;
}

inline int64_t carfac_reader_t::get_render_pos() const
{
    return render_pos;
}

inline void draw_notes(note_image_t& note_image, int y_pos = 30)
{
    cv::Point p(5, y_pos);
    for(auto note : note_image.midi){
        auto color = cv::Scalar(0.6 * 255, 255, 255 - note.octave() * 0.07 * 255);
        cv::putText(note_image.mat, note.note_name, p, cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
        p.x += 100;
    }
}

inline cv::Scalar color_based_on_pitch(str_note_event_t const& note)
{
    int pitch_class = note.to_midi_int() % 12;
    cv::Scalar color;
    switch(pitch_class) {
        case 0: color = cv::Scalar(0, 0, 255); break;  // C -> Red
        case 1: color = cv::Scalar(0, 255, 0); break;  // C# -> Green
        case 2: color = cv::Scalar(255, 0, 0); break;  // D -> Blue
        case 3: color = cv::Scalar(0, 255, 255); break;  // D# -> Cyan
        case 4: color = cv::Scalar(255, 255, 0); break;  // E -> Yellow
        case 5: color = cv::Scalar(255, 0, 255); break;  // F -> Magenta
        case 6: color = cv::Scalar(0, 255, 255); break;  // F# -> Cyan
        case 7: color = cv::Scalar(255, 255, 255); break;  // G -> White
        case 8: color = cv::Scalar(128, 128, 128); break;  // G# -> Grey
        case 9: color = cv::Scalar(255, 128, 128); break;  // A -> Light Red
        case 10: color = cv::Scalar(128, 255, 128); break; // A# -> Light Green
        case 11: color = cv::Scalar(128, 128, 255); break; // B -> Light Blue
    }
    return color;
}

// Function to map a MIDI note to a color based on pitch class using HSV
inline cv::Scalar getColorForPitch(int midiNote) {
    // Map MIDI note to pitch class (mod 12)
    int pitchClass = midiNote % 12;

    // Map pitch class to hue (0-11 map to 0-180 in OpenCV HSV space)
    double hue = (pitchClass / 12.0) * 180.0;  // 0-11 -> 0-180 for OpenCV HSV
    double saturation = 255;  // Full saturation for vibrant colors
    double value = 255;  // Full brightness

    // Convert HSV to BGR
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, saturation, value));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    return cv::Scalar(bgr.at<cv::Vec3b>(0, 0)[0], bgr.at<cv::Vec3b>(0, 0)[1], bgr.at<cv::Vec3b>(0, 0)[2]);
}

inline void draw_notes_as_keys(note_image_t& note_image, int y_pos = 30)
{
    cv::Point p(5, y_pos);
    auto image_width = note_image.mat.cols;
    auto key_width = 8;
    auto midi_count = 100;
    auto step = image_width / double(key_width) / midi_count;
    
    for(auto note : note_image.midi){
        auto color = getColorForPitch(note.to_midi_int());
        p.x = note.to_midi_int() * step * key_width;
        auto black_key_offset = (note.note_name.size() > 2 ? -5 : 0);
        cv::putText(note_image.mat, note.note_name, p - cv::Point(9, -15 + black_key_offset), cv::FONT_HERSHEY_PLAIN, 1.2, color, 2);
        cv::rectangle(note_image.mat, cv::Rect(p.x+2, y_pos - 15 + black_key_offset, key_width-2, 15), color, -1);
    }
}

inline void carfac_reader_t::reset()
{
    render_pos = 0;
    active_notes.reset();
}