## ======================================================================== ##
## Copyright 2009-2016 Intel Corporation                                    ##
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

# unhide compiler to make it easier for users to see what they are using
mark_as_advanced(CLEAR CMAKE_CXX_COMPILER)

## Compiler configuration macro ##
macro(CONFIGURE_COMPILER)
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    include(${PROJECT_SOURCE_DIR}/cmake/icc.cmake)
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    include(${PROJECT_SOURCE_DIR}/cmake/gcc.cmake)
  elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    include(${PROJECT_SOURCE_DIR}/cmake/clang.cmake)
  else()
    message(FATAL_ERROR "Unsupported compiler specified: '${CMAKE_CXX_COMPILER_ID}'")
  endIF()
endmacro()

macro(CONFIGURE_ISA_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
endmacro()
