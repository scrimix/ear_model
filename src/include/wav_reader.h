#pragma once

#include <sndfile.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>

// Map sample from [-1, +1] to [0, 1]
inline float normalize_sample(float sample) {
    return (sample + 1.0f) * 0.5f;
}

// Read WAV from file and merge to mono, normalized to [0,1]
inline std::vector<float> readWavFile(const std::string& path) {
    SF_INFO sfinfo = {};
    SNDFILE* sndfile = sf_open(path.c_str(), SFM_READ, &sfinfo);
    if (!sndfile) {
        throw std::runtime_error(std::string("Failed to open file: ") + sf_strerror(nullptr));
    }

    int channels = sfinfo.channels;
    sf_count_t frames = sfinfo.frames;
    std::vector<float> interleaved(frames * channels);
    sf_count_t readcount = sf_readf_float(sndfile, interleaved.data(), frames);
    sf_close(sndfile);

    if (readcount != frames) {
        throw std::runtime_error("Did not read expected number of frames");
    }

    std::vector<float> mono;
    mono.reserve(frames);
    if (channels == 1) {
        for (sf_count_t i = 0; i < frames; ++i) {
            mono.push_back(interleaved[i]);
            // mono.push_back(normalize_sample(interleaved[i]));
        }
    } else {
        for (sf_count_t i = 0; i < frames; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < channels; ++c) {
                sum += interleaved[i * channels + c];
            }
            mono.push_back(sum / channels);
            // mono.push_back(normalize_sample(sum / channels));
        }
    }
    return mono;
}

// Virtual file struct for in-memory WAV reading
typedef struct {
    const char*  data;   // pointer to WAV bytes
    sf_count_t   size;   // total length
    sf_count_t   pos;    // current read position
} VirtualFile;

static sf_count_t vio_get_filelen(void* user_data) {
    return static_cast<VirtualFile*>(user_data)->size;
}

static sf_count_t vio_seek(sf_count_t offset, int whence, void* user_data) {
    auto* vf = static_cast<VirtualFile*>(user_data);
    sf_count_t newpos;
    switch (whence) {
        case SEEK_SET: newpos = offset; break;
        case SEEK_CUR: newpos = vf->pos + offset; break;
        case SEEK_END: newpos = vf->size + offset; break;
        default:       return -1;
    }
    if (newpos < 0 || newpos > vf->size) return -1;
    vf->pos = newpos;
    return vf->pos;
}

static sf_count_t vio_read(void* ptr, sf_count_t count, void* user_data) {
    auto* vf = static_cast<VirtualFile*>(user_data);
    sf_count_t remain  = vf->size - vf->pos;
    sf_count_t to_read = (count > remain) ? remain : count;
    std::memcpy(ptr, vf->data + vf->pos, to_read);
    vf->pos += to_read;
    return to_read;
}

static sf_count_t vio_tell(void* user_data) {
    return static_cast<VirtualFile*>(user_data)->pos;
}

// Read WAV from memory buffer and merge to mono, normalized to [0,1]
inline std::vector<float> readWavBuffer(const std::string& buffer) {
    VirtualFile vf{ buffer.data(), static_cast<sf_count_t>(buffer.size()), 0 };
    SF_VIRTUAL_IO vio{
        vio_get_filelen,
        vio_seek,
        vio_read,
        /*write=*/nullptr,
        vio_tell
    };

    SF_INFO sfinfo = {};
    SNDFILE* sndfile = sf_open_virtual(&vio, SFM_READ, &sfinfo, &vf);
    if (!sndfile) {
        throw std::runtime_error(std::string("Failed to open virtual buffer: ") + sf_strerror(nullptr));
    }

    int channels = sfinfo.channels;
    sf_count_t frames = sfinfo.frames;
    std::vector<float> interleaved(frames * channels);
    sf_count_t readcount = sf_readf_float(sndfile, interleaved.data(), frames);
    sf_close(sndfile);

    if (readcount != frames) {
        throw std::runtime_error("Did not read expected number of frames from buffer");
    }

    std::vector<float> mono;
    mono.reserve(frames);
    if (channels == 1) {
        for (sf_count_t i = 0; i < frames; ++i) {
            // mono.push_back(normalize_sample(interleaved[i]));
            mono.push_back(interleaved[i]);
        }
    } else {
        for (sf_count_t i = 0; i < frames; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < channels; ++c) {
                sum += interleaved[i * channels + c];
            }
            // mono.push_back(normalize_sample(sum / channels));
            mono.push_back(sum / channels);
        }
    }
    return mono;
}
