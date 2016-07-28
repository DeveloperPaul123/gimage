############################################################################
# <summary> CMakeLists.txt for gimage a gpu based image processing library. </summary>
# <date>    2016-05          </date>
# <author>  Paul T </author>
############################################################################
# minimum required cmake version
cmake_minimum_required(VERSION 2.8)
set(PROJECT_NAME gimage)
project(${PROJECT_NAME} C CXX)

#enable export header for dll
include (GenerateExportHeader)

#find required packages, gtest will be downloaded. 
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

#set directories.
set(CUDA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/cuda)
set(COMMON_INCLUDES ${PROJECT_SOURCE_DIR}/include)
set(TEST_SOURCES_DIR ${PROJECT_SOURCE_DIR}/test)

file(GLOB gimage_headers ${COMMON_INCLUDES}/*.h)
file(GLOB gimage_sources ${CUDA_SOURCE_DIR}/*.cu ${COMMON_SRC_DIR}/*.cpp)
file(GLOB test_source ${TEST_SOURCES_DIR}/*.cpp)

set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)

include_directories(${OpenCV_INCLUDE_DIRS} ${COMMON_INCLUDES})

#add gimage library.
cuda_add_library(gimage SHARED ${gimage_headers} ${gimage_sources})
GENERATE_EXPORT_HEADER(gimage
			 EXPORT_FILE_NAME ${PROJECT_SOURCE_DIR}/include/gimage_export.h
             EXPORT_MACRO_NAME GIMAGE_EXPORT
)
include_directories(".")

#define output path for the libraries.
set(LIBRARY_OUTPUT_PATH ${gimage_BINARY_DIR}/bin CACHE PATH "Directory for libraries.")
mark_as_advanced(LIBRARY_OUTPUT_PATH)
 
#set configuration file variables. 
set(gimage_INCLUDE_DIR "${gimage_SOURCE_DIR}/include")
set(gimage_LIBRARY_DIR "${LIBRARY_OUTPUT_PATH}/release")
set(gimage_DEBUG_LIBRARY_DIR "${LIBRARY_OUTPUT_PATH}/debug")

#Configure the file and generate a gimageConfig.cmake file for
#easy inclusion of this library with other libs.
set(gimage_PACKAGE_NAME "gimage")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/gimageConfig.cmake.in" 
			   "${gimage_BINARY_DIR}/gimageConfig.cmake"
			   @ONLY@)

option(INCLUDE_TESTING "Enable to including testing code and subproject." OFF)
if(INCLUDE_TESTING)
	#enable g testing.
	enable_testing()
	add_subdirectory(${EXT_PROJECTS_DIR}/gtest)
	include_directories(${GTEST_INCLUDE_DIRS})
	#setup testing.
	set(PROJECT_TEST_NAME ${PROJECT_NAME}_test)
	add_executable(${PROJECT_TEST_NAME} ${test_source})
	if(NOT WIN32)
		target_link_libraries(${PROJECT_TEST_NAME}
			${GTEST_LIBS_DIR}/libgtest.a
			${GTEST_LIBS_DIR}/libgtest_main.a
			)
	else()
		target_link_libraries(${PROJECT_TEST_NAME}
			debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
			optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${CMAKE_FIND_LIBRARY_SUFFIXES}
			)
		target_link_libraries(${PROJECT_TEST_NAME}
			debug ${GTEST_LIBS_DIR}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
			optimized ${GTEST_LIBS_DIR}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main${CMAKE_FIND_LIBRARY_SUFFIXES}
			)
	endif()
	target_link_libraries(${PROJECT_TEST_NAME} ${OpenCV_LIBS})
	target_link_libraries(${PROJECT_TEST_NAME} gimage)
	add_test(allTests ${PROJECT_TEST_NAME})
endif(INCLUDE_TESTING)			   



