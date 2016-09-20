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

## Macro for printing CMake variables ##
macro(PRINT var)
  message("${var} = ${${var}}")
endmacro()

## Macro to print a warning message that only appears once ##
macro(JRAY_WARN_ONCE identifier message)
  set(INTERNAL_WARNING "JRAY_WARNED_${identifier}")
  if(NOT ${INTERNAL_WARNING})
    message(WARNING ${message})
    set(${INTERNAL_WARNING} ON CACHE INTERNAL "Warned about '${message}'")
  endif()
endmacro()

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

## Tasking system configuration macro ##
macro(CONFIGURE_TASKING_SYSTEM)
  # -------------------------------------------------------
  # Setup tasking system build configuration
  # -------------------------------------------------------

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(CILK_STRING "Cilk")
  endif()

  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(TASKING_DEFAULT TBB)
  else()
    set(TASKING_DEFAULT OpenMP)
  endif()

  set(JRAY_TASKING_SYSTEM ${TASKING_DEFAULT} CACHE STRING
      "Per-node thread tasking system [OpenMP,TBB,Cilk,Debug]")

  set_property(CACHE JRAY_TASKING_SYSTEM PROPERTY
               STRINGS TBB ${CILK_STRING} OpenMP Internal Debug)
  mark_as_advanced(JRAY_TASKING_SYSTEM)

  # NOTE(jda) - Make the JRAY_TASKING_SYSTEM build option case-insensitive
  string(TOUPPER ${JRAY_TASKING_SYSTEM} JRAY_TASKING_SYSTEM_ID)

  set(USE_TBB    FALSE)
  set(USE_CILK   FALSE)
  set(USE_OPENMP FALSE)

  if(${JRAY_TASKING_SYSTEM_ID} STREQUAL "TBB")
    set(USE_TBB TRUE)
  elseif(${JRAY_TASKING_SYSTEM_ID} STREQUAL "CILK")
    set(USE_CILK TRUE)
  elseif(${JRAY_TASKING_SYSTEM_ID} STREQUAL "OPENMP")
    set(USE_OPENMP TRUE)
  endif()

  unset(TASKING_SYSTEM_LIBS)
  unset(TASKING_SYSTEM_LIBS_MIC)

  if(USE_TBB)
    find_package(TBB REQUIRED)
    add_definitions(-DJRAY_USE_TBB)
    include_directories(${TBB_INCLUDE_DIRS})
    set(TASKING_SYSTEM_LIBS ${TBB_LIBRARIES})
    set(TASKING_SYSTEM_LIBS_MIC ${TBB_LIBRARIES_MIC})
  else(USE_TBB)
    unset(TBB_INCLUDE_DIR          CACHE)
    unset(TBB_LIBRARY              CACHE)
    unset(TBB_LIBRARY_DEBUG        CACHE)
    unset(TBB_LIBRARY_MALLOC       CACHE)
    unset(TBB_LIBRARY_MALLOC_DEBUG CACHE)
    unset(TBB_INCLUDE_DIR_MIC      CACHE)
    unset(TBB_LIBRARY_MIC          CACHE)
    unset(TBB_LIBRARY_MALLOC_MIC   CACHE)
    if(USE_OPENMP)
      find_package(OpenMP)
      if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS
            "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        add_definitions(-DJRAY_USE_OMP)
      endif()
    elseif(USE_CILK)
      add_definitions(-DJRAY_USE_CILK)
    else()#Debug
      # Do nothing, will fall back to scalar code (useful for debugging)
    endif()
  endif(USE_TBB)
endmacro()

macro(CONFIGURE_ISA_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endmacro()
