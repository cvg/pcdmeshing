cmake_minimum_required(VERSION 3.6)

project(pcdmeshing)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CGAL_DIR ${CMAKE_CURRENT_LIST_DIR}/CGAL-5.3.1/lib/cmake/CGAL)
set(BOOST_ROOT ${CMAKE_CURRENT_LIST_DIR}/boost_1_77_0)

find_package(CGAL 5.3.1 REQUIRED)
message(STATUS "Found CGAL ${CGAL_VERSION}")
find_package(Boost 1.71 REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(pybind11)

pybind11_add_module(_pcdmeshing src/bindings.cc)
target_link_libraries(_pcdmeshing PUBLIC CGAL::CGAL OpenMP::OpenMP_CXX)

add_executable(meshing  src/exe.cc)
target_link_libraries(meshing PRIVATE CGAL::CGAL OpenMP::OpenMP_CXX)
