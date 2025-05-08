include_directories(include/carfac)
file(GLOB carfac_src_files "src/*.cc")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(carfac ${carfac_src_files})
target_link_libraries(carfac Eigen3::Eigen)

target_include_directories(carfac
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)