#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include "helpers.h"

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

inline std::string midi_array_to_string(std::vector<int> const& midi_notes)
{
    std::stringstream result;
    result << "[";
    std::string delim;
    for(auto& note : midi_notes){
        auto str_note = str_note_event_t::from_int(note);
        result << delim << str_note.note_name;
        delim = ",";
    }
    result << "]";
    return result.str();
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

inline std::vector<int> midi_to_labels(std::vector<str_note_event_t> midi)
{
  std::vector<int> labels;
    for(auto& note : midi)
      labels.push_back(note.to_midi_int());
  return labels;
}

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

inline float midi_to_freq(float midi) {
    return 440.0f * std::pow(2.0f, (midi - 69.0f) / 12.0f);
}

inline float freq_to_midi(float freq) {
    return 69.0f + 12.0f * std::log2(freq / 440.0f);
}

inline float log_interp(float y_norm, float f_min, float f_max) {
    return f_min * std::pow(f_max / f_min, y_norm);
}

inline std::pair<int, int> get_midi_range_for_region(
    int region_y,
    int region_height,
    int image_height,
    float f_min = 27.5f,
    float f_max = 8186.0f
) {
    float y0 = static_cast<float>(region_y) / image_height;
    float y1 = static_cast<float>(region_y + region_height) / image_height;

    // flip to match frequency axis
    y0 = 1.0f - y0;
    y1 = 1.0f - y1;

    float freq0 = f_min * std::pow(f_max / f_min, y0);
    float freq1 = f_min * std::pow(f_max / f_min, y1);

    float f_low = std::min(freq0, freq1);
    float f_high = std::max(freq0, freq1);

    int midi_low = static_cast<int>(std::floor(freq_to_midi(f_low)));
    int midi_high = static_cast<int>(std::ceil(freq_to_midi(f_high)));

    // midi_low = std::max(midi_low - 5, 21);
    // midi_high = std::min(midi_high + 5, 108);

    return {midi_low, midi_high};
}

inline std::vector<uint32_t> filter_and_remap_midi(
    const std::vector<uint32_t>& midi_notes,
    uint32_t midi_low,
    uint32_t midi_high
) {
    std::vector<uint32_t> local_labels;
    for (int note : midi_notes) {
        if (note >= midi_low && note <= midi_high) {
            int local_label = note - midi_low + 1; // 1 to have 0 value as empty label
            local_labels.push_back(local_label);
        }
    }
    return local_labels;
}

inline int remap_midi_back(uint32_t local_label, uint32_t midi_low) {
    return (local_label-1) + midi_low; // 1 to have 0 value as empty label
}


inline std::vector<uint32_t> labels_to_region_specific(std::vector<uint32_t> labels, cv::Rect region, cv::Size image_size)
{
    auto [midi_low, midi_high] = get_midi_range_for_region(region.y, region.height, image_size.height);
    auto region_specific = filter_and_remap_midi(labels, midi_low, midi_high);
    if(region_specific.empty())
        region_specific.push_back(0);
    return region_specific;
}

inline std::vector<int> labels_from_region_to_global(std::vector<int> region_specific, cv::Rect region, cv::Size image_size)
{
    auto [midi_low, midi_high] = get_midi_range_for_region(region.y, region.height, image_size.height);
    std::vector<int> result;
    for(auto& label : region_specific)
        result.push_back(remap_midi_back(label, midi_low));
    return result;
}