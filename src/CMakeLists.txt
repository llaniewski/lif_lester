cmake_minimum_required( VERSION 3.24 )
project( lif )
find_package( pugixml REQUIRED )
add_subdirectory( ../lester L3STER-bin )
add_executable( ${CMAKE_PROJECT_NAME} main.cpp solver_2D_O3.cpp )
target_link_libraries( ${CMAKE_PROJECT_NAME} L3STER pugixml )


target_precompile_headers(${CMAKE_PROJECT_NAME} PRIVATE
    <vector>
    <print>
    "../lester/include/l3ster/l3ster.hpp"
)
