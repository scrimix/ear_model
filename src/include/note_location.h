#pragma once
#include <bitset>
#include <random>
#include <unordered_set>
#include <map>
#include <fstream>
#include <sstream>
#include <opencv2/core.hpp>

inline constexpr size_t note_location_resolution = 256;
using note_location_t = std::bitset<note_location_resolution>;
using note_map_t = std::map<int, note_location_t>;

inline note_map_t create_note_map() {
    note_map_t result;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (auto midi_note = 0; midi_note < 127; midi_note++) {
        note_location_t sdr;
        std::unordered_set<int> on_bits;

        while (on_bits.size() < note_location_resolution * 0.02) {
            int idx = gen() % note_location_resolution;
            on_bits.insert(idx);
        }

        for (int idx : on_bits) {
            sdr.set(idx);
        }

        result[midi_note] = sdr;
    }
    return result;
}

inline void write_note_map_to_file(note_map_t const& note_map, std::string const& file_path)
{
    std::ofstream file(file_path);
    for(auto [midi, location] : note_map)
        file << midi << ":" << location.to_string() << "\n";
    file.close();
}

inline note_map_t read_note_map_from_file(std::string const& file_path)
{
    note_map_t result;

    std::ifstream file(file_path);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string keyStr, sdrStr;
        std::getline(iss, keyStr, ':');
        std::getline(iss, sdrStr);

        int midi_note = std::stoi(keyStr);
        std::bitset<note_location_resolution> sdr(sdrStr);

        result[midi_note] = sdr;
    }

    return result;
}

inline std::vector<uint8_t> concat(std::vector<uint8_t> const& image, note_location_t const& note_location)
{
    std::vector<uint8_t> result;
    result.resize(image.size() + note_location.size());
    for(auto i = 0; i < image.size(); i++)
        result[i] = image[i];
    for(auto i = 0; i < note_location.size(); i++)
        result[image.size() + i] = note_location[i] ? 255 : 0;
    return result;
}

inline std::vector<uint8_t> note_location_to_vec(note_location_t const& location)
{
    std::vector<uint8_t> result;
    result.resize(location.size());
    for(auto i = 0; i < location.size(); i++)
        result[i] = location[i] ? 255 : 0;
    return result;
}

inline note_location_t midi_pred_to_location(note_map_t const& note_map, std::vector<int> const& pred)
{
    note_location_t result;
    for(auto& midi : pred)
        result |= note_map.at(midi);
    return result;
}

inline std::vector<uint32_t> to_sparse_indices(const std::vector<uint8_t>& dense_vec)
{
    std::vector<uint32_t> sparse;
    for (uint32_t i = 0; i < dense_vec.size(); ++i)
        if (dense_vec[i]) sparse.push_back(i);
    return sparse;
}
