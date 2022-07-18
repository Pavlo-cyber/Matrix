cmake_minimum_required(VERSION 3.22)
project(opencl_mac)

set(CMAKE_CXX_STANDARD 14)

include_directories(/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk/System/Library/Frameworks/OpenCL.framework/Headers)
find_package(OpenCL REQUIRED)


add_executable(opencl_mac main.cpp)

target_link_libraries(opencl_mac OpenCL::OpenCL)