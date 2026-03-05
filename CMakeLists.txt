cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(align_angle LANGUAGES CXX)

find_package(HOOMD 5.0.0 REQUIRED)
message(STATUS "Found HOOMD: ${HOOMD_DIR}")

# Force install into the HOOMD Python site-packages tree
set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH
    "Installation prefix" FORCE)

add_subdirectory(src)
