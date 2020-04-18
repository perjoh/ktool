cmake_minimum_required(VERSION 3.12)
project(ktool) 

enable_testing()
add_subdirectory(external/kgfx)
add_subdirectory(src/${PROJECT_NAME})

