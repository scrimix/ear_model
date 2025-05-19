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