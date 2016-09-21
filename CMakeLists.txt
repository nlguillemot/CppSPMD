## ======================================================================== ##
## Copyright 2016 Jefferson Amstutz                                         ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

project(CppSPMD)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
set(CPPSPMD_VERSION_MAJOR 0)
set(CPPSPMD_VERSION_MINOR 1)
set(CPPSPMD_VERSION_PATCH 0)
set(CPPSPMD_VERSION
  ${CPPSPMD_VERSION_MAJOR}.${CPPSPMD_VERSION_MINOR}.${CPPSPMD_VERSION_PATCH}
)

set(CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CONFIGURATION_TYPES})
endif()

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

include(ispc)

# Make CXX compiler visible in basic-mode GUIs
mark_as_advanced(CLEAR CMAKE_CXX_COMPILER)

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  include(${PROJECT_SOURCE_DIR}/cmake/icc.cmake)
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
  include(${PROJECT_SOURCE_DIR}/cmake/gcc.cmake)
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  include(${PROJECT_SOURCE_DIR}/cmake/clang.cmake)
#elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
#  include(${PROJECT_SOURCE_DIR}/cmake/msvc.cmake)
else()
  message(FATAL_ERROR "Unsupported compiler specified: '${CMAKE_CXX_COMPILER_ID}'")
endIF()

add_subdirectory(src)

