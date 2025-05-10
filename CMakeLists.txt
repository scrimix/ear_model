cmake_minimum_required(VERSION 3.20)
project(ear_model)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("/Library/Developer/CommandLineTools/SDKs/MacOSX15.sdk/usr/include/c++/v1")

find_package(PkgConfig REQUIRED)
pkg_check_modules(FLUIDSYNTH REQUIRED fluidsynth)

find_package(OpenCV REQUIRED)
find_package(SDL2 REQUIRED)
add_subdirectory(carfac)

include_directories(htm/include)
link_directories(htm/lib)

set(opencv_lib opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

execute_process(
    COMMAND pkg-config --libs-only-L fluidsynth
    OUTPUT_VARIABLE FLUIDSYNTH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REPLACE "-L" "" FLUIDSYNTH_LIB_DIR "${FLUIDSYNTH_LIB_DIR}")
link_directories(${FLUIDSYNTH_LIB_DIR})

# Set paths manually
set(FFTW3_INCLUDE_DIR /opt/homebrew/include)
set(FFTW3_LIBRARY /opt/homebrew/lib/libfftw3.dylib)

# Add checks to verify paths
if(NOT EXISTS ${FFTW3_INCLUDE_DIR}/fftw3.h)
    message(FATAL_ERROR "FFTW3 header not found at ${FFTW3_INCLUDE_DIR}")
endif()
if(NOT EXISTS ${FFTW3_LIBRARY})
    message(FATAL_ERROR "FFTW3 library not found at ${FFTW3_LIBRARY}")
endif()

find_package(SndFile REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS})

include_directories("src/include")

include_directories("crow/include")

include_directories("dr_libs")
include_directories(${FLUIDSYNTH_INCLUDE_DIRS})

add_subdirectory(midifile)


list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/crow")
find_package(Crow REQUIRED)


set(COMMON_LIBS carfac ${opencv_lib} ${SNDFILE_LIBRARIES} ${FLUIDSYNTH_LIBRARIES} htm_core midifile asio::asio)

add_executable(midi_gen src/midi_gen.cpp)
target_link_libraries(midi_gen ${COMMON_LIBS})

add_executable(spectro src/spectro.cpp)
target_include_directories(spectro PRIVATE ${FFTW3_INCLUDE_DIR})
target_link_libraries(spectro PRIVATE ${FFTW3_LIBRARY} ${COMMON_LIBS})

add_executable(playback src/playback.cpp)
target_link_libraries(playback ${SDL2_LIBRARIES} ${COMMON_LIBS})

add_executable(mnist_train src/mnist_train.cpp)
target_link_libraries(mnist_train ${COMMON_LIBS})

add_executable(notes_train src/notes_train.cpp)
target_link_libraries(notes_train ${COMMON_LIBS})

add_executable(midi_cvt src/midi_cvt.cpp)
target_link_libraries(midi_cvt ${COMMON_LIBS})

add_executable(app src/app.cpp)
target_link_libraries(app ${COMMON_LIBS})