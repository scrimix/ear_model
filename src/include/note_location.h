#pragma once
#include <bitset>
#include <random>
#include <unordered_set>
#include <map>
#include <fstream>
#include <sstream>
#include <opencv2/core.hpp>

inline constexpr size_t note_location_resolution = 4096;
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

inline note_location_t encode_note_shifted(
    int note_id,
    int region_id,
    int sparsity = 15,
    int shift = 1,
    int resolution = note_location_resolution // or whatever size your SDR expects
)
{
    note_location_t sdr;  // assumed to be a std::bitset<resolution>

    std::mt19937 rng(note_id * 997);  // Stable per note
    std::uniform_int_distribution<int> dist(0, resolution - 1);

    std::unordered_set<int> selected;
    while (selected.size() < sparsity)
    {
        int base = dist(rng);
        int shifted = base + region_id * shift;

        if (shifted >= resolution)
            continue; // skip out-of-bounds bit

        if (!sdr.test(shifted)) {
            sdr.set(shifted);
            selected.insert(shifted);
        }
    }

    return sdr;
}

inline note_location_t encode_note_region_bucketed(
    int note_id,
    int region_id,
    int note_count,    // e.g., 128
    int region_count,  // e.g., 16
    int sparsity       // e.g., 5
)
{
    note_location_t sdr;

    int bucket_size   = note_location_resolution / note_count;
    int region_shift  = bucket_size / region_count;

    int base = note_id * bucket_size;
    int shift = region_id * region_shift;

    for (int i = 0; i < sparsity; ++i) {
        int bit = (base + shift + i) % note_location_resolution;
        sdr.set(bit);
    }

    return sdr;
}

