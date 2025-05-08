include_directories(include/midifile)
file(GLOB midi_src_files "src/*.cpp")

add_library(midifile ${midi_src_files})

target_include_directories(midifile
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)