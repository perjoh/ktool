find_package(Catch2 CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(glm CONFIG REQUIRED)

set(IS_MSVC "$<CXX_COMPILER_ID:MSVC>")
set(IS_GCC "$<CXX_COMPILER_ID:GNU>")

function(enable_compile_options target)
    # Warning level (MSVC)
    target_compile_options(${target} PUBLIC "$<${IS_MSVC}:/W4>")
    target_compile_options(${target} PUBLIC "$<${IS_MSVC}:/bigobj>")

    # Optimizations GCC (Release)
    target_compile_options(${target} PUBLIC "$<$<AND:${IS_GCC},$<CONFIG:RELEASE>>:-O3>")
endfunction(enable_compile_options)

# Library
add_library(${PROJECT_NAME}lib STATIC   surface.cpp 
                                        surface_io.cpp 
                                        argparser.cpp)

target_link_libraries(${PROJECT_NAME}lib PUBLIC glm 
                                                kgfx 
                                        PRIVATE nlohmann_json 
                                                nlohmann_json::nlohmann_json)

target_compile_features(${PROJECT_NAME}lib PUBLIC cxx_std_17)
enable_compile_options(${PROJECT_NAME}lib)

# Executable
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}lib)
enable_compile_options(${PROJECT_NAME})

# Test executable
add_executable(${PROJECT_NAME}test  main.test.cpp 
                                    surface.test.cpp 
                                    argparser.test.cpp 
                                    surface_io.test.cpp)

target_link_libraries(${PROJECT_NAME}test ${PROJECT_NAME}lib Catch2::Catch2)
add_test(NAME ${PROJECT_NAME}test COMMAND ${PROJECT_NAME}test)
