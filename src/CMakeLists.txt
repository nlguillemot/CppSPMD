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

set(TARGET_NAME "cppspmd")

## Figure out which mode to build ##
set(CPPSPMD_BUILD_MODE "CPPSPMD" CACHE STRING "What mode to build the code in?")
set_property(CACHE CPPSPMD_BUILD_MODE PROPERTY STRINGS CPPSPMD SCALAR ISPC)

add_definitions(-D${CPPSPMD_BUILD_MODE})

## Figure out which test to run ##
set(CPPSPMD_BUILD_TEST "SIMPLE" CACHE STRING "Which test do you want built?")
set_property(CACHE CPPSPMD_BUILD_MODE PROPERTY STRINGS
             SIMPLE NOISE MANDELBROT VOLUME OPTIONS)

add_definitions(-D${CPPSPMD_BUILD_TEST})

add_executable(${TARGET_NAME}
  avx_mathfun_tweaked.h
  spmd_avx2-i32x8.h
  main.cpp
)

