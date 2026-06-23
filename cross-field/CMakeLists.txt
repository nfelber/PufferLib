cmake_minimum_required(VERSION 3.20)

project(crossfield_precompute LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# -------------------------------------------------------------------------
# Eigen
# -------------------------------------------------------------------------
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(eigen)

if (NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(
        Eigen3::Eigen
        INTERFACE
            ${eigen_SOURCE_DIR}
    )
endif()

# -------------------------------------------------------------------------
# Directional
#
# We only populate the source tree and include its headers.
# Do not call FetchContent_MakeAvailable(directional), because we are not
# building Directional's tutorial/example targets here.
# -------------------------------------------------------------------------
FetchContent_Declare(
    directional
    GIT_REPOSITORY https://github.com/avaxman/Directional.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(directional)

if (NOT directional_POPULATED)
    FetchContent_Populate(directional)
endif()

message(STATUS "Directional source dir: ${directional_SOURCE_DIR}")

if (NOT EXISTS "${directional_SOURCE_DIR}/include/directional/readOBJ.h")
    message(FATAL_ERROR
        "Could not find Directional header: "
        "${directional_SOURCE_DIR}/include/directional/readOBJ.h"
    )
endif()

# -------------------------------------------------------------------------
# zlib
# -------------------------------------------------------------------------
set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.3.1
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(zlib)

if (TARGET zlibstatic)
    set(CROSSFIELD_ZLIB_TARGET zlibstatic)
elseif (TARGET zlib)
    set(CROSSFIELD_ZLIB_TARGET zlib)
else()
    message(FATAL_ERROR "Could not find fetched zlib target.")
endif()

# -------------------------------------------------------------------------
# cnpy
#
# Do not call FetchContent_MakeAvailable(cnpy), because cnpy's own
# CMakeLists.txt is too old for recent CMake versions.
# -------------------------------------------------------------------------
FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(cnpy)

if (NOT cnpy_POPULATED)
    FetchContent_Populate(cnpy)
endif()

# -------------------------------------------------------------------------
# Executable
# -------------------------------------------------------------------------
add_executable(
    crossfield_precompute
    src/main.cpp
    ${cnpy_SOURCE_DIR}/cnpy.cpp
)

target_include_directories(
    crossfield_precompute
    PRIVATE
        ${directional_SOURCE_DIR}/include
        ${cnpy_SOURCE_DIR}
)

target_link_libraries(
    crossfield_precompute
    PRIVATE
        Eigen3::Eigen
        ${CROSSFIELD_ZLIB_TARGET}
)
