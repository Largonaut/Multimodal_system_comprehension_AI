# GREMLIN C++ - High-Performance Inference Engine
# Security: Zero hub dependencies, complete control
# Philosophy: Built from scratch, machine precision

cmake_minimum_required(VERSION 3.18)
project(GREMLIN_CPP VERSION 1.0.0 LANGUAGES CXX)

# C++20 required (concepts, ranges, etc.)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /WX /permissive-)
    # Optimization
    add_compile_options(/O2 /Oi /Ot /GL)
    # SIMD
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    # Optimization
    add_compile_options(-O3 -march=native -mtune=native)
    # SIMD
    add_compile_options(-mavx2 -mfma)
endif()

# Options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(USE_GEMMA_CPP "Use Gemma.cpp for model inference" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings (PyBind11)" ON)
option(ENABLE_PROFILING "Enable performance profiling" OFF)

# Find dependencies
find_package(Threads REQUIRED)

# Optional: PyTorch (for data loader extension)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, fetching from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# Optional: Gemma.cpp
if(USE_GEMMA_CPP)
    find_package(GemmaCpp REQUIRED)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Core library
add_library(gremlin_core STATIC
    src/gremlin_tokenizer.cpp
    src/semantic_dictionary.cpp
    src/trap_door.cpp
    src/gremlin_engine.cpp
)

target_link_libraries(gremlin_core
    PUBLIC Threads::Threads
)

# CLI tool
add_executable(gremlin-cli
    src/main.cpp
)

target_link_libraries(gremlin-cli
    PRIVATE gremlin_core
)

# Python bindings (optional)
if(BUILD_PYTHON_BINDINGS AND pybind11_FOUND)
    pybind11_add_module(gremlin_dataloader
        src/gremlin_dataloader.cpp
    )

    target_link_libraries(gremlin_dataloader
        PRIVATE gremlin_core
    )
endif()

# Tests (optional)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples (optional)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation
install(TARGETS gremlin_core gremlin-cli
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/GremlinCppConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/GremlinCppConfigVersion.cmake"
    DESTINATION lib/cmake/GremlinCpp
)

# Print configuration
message(STATUS "")
message(STATUS "GREMLIN C++ Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Use Gemma.cpp: ${USE_GEMMA_CPP}")
message(STATUS "  Python bindings: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Profiling: ${ENABLE_PROFILING}")
message(STATUS "")
