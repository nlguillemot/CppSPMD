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

## Figure out which mode to build ##
set(CPPSPMD_BUILD_MODE "CPPSPMD" CACHE STRING "What mode to build the code in?")
set_property(CACHE CPPSPMD_BUILD_MODE PROPERTY STRINGS CPPSPMD SCALAR ISPC)

add_definitions(-D${CPPSPMD_BUILD_MODE})

include_directories(common)

# Simple
add_executable(simple
  common/avx_mathfun_tweaked.h
  common/common.h
  common/spmd_avx2-i32x8.h
  simple.cpp
)

# Noise
add_executable(noise
  common/avx_mathfun_tweaked.h
  common/common.h
  common/spmd_avx2-i32x8.h
  noise.cpp
)

# Mandelbrot
add_executable(mandelbrot
  common/avx_mathfun_tweaked.h
  common/common.h
  common/spmd_avx2-i32x8.h
  mandelbrot.cpp
)

# Volume
add_executable(volume
  common/avx_mathfun_tweaked.h
  common/common.h
  common/spmd_avx2-i32x8.h
  volume.cpp
)

# Options
add_executable(options
  common/avx_mathfun_tweaked.h
  common/common.h
  common/spmd_avx2-i32x8.h
  options.cpp
)
