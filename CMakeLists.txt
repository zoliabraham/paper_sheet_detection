cmake_minimum_required(VERSION 3.15)
project(paper_sheet_detection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#[[TODO set opencv Directory]]
SET("OpenCV_DIR" "C:/opencv/mingw-build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(paper_sheet_detection main.cpp edge_detector.hpp edge_detector.cpp)

# linking
target_link_libraries(paper_sheet_detection ${OpenCV_LIBS})