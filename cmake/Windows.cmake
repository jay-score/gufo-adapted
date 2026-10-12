# Native Windows source builds use TheRock's clang with the MSVC STL/SDK.
# Keep the POSIX headers and libraries confined to Gufo-owned targets.
if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR NOT CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  message(FATAL_ERROR "Windows builds require clang targeting the MSVC ABI; use tools/windows/build.ps1")
endif()
if (ENGINE_ENABLE_HIP OR NOT DEFINED ENGINE_ENABLE_HIP)
  message(FATAL_ERROR "Windows HIP support is a later integration stage; configure ENGINE_ENABLE_HIP=OFF")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_library(gufo_posix_compat STATIC
  ${PROJECT_SOURCE_DIR}/compat/win32/posix.cpp
  ${PROJECT_SOURCE_DIR}/compat/win32/directory.cpp
  ${PROJECT_SOURCE_DIR}/compat/win32/spawn.cpp)
target_include_directories(gufo_posix_compat SYSTEM BEFORE PUBLIC
  ${PROJECT_SOURCE_DIR}/compat/win32/include)
target_compile_definitions(gufo_posix_compat PUBLIC
  _CRT_DECLARE_NONSTDC_NAMES=0 WIN32_LEAN_AND_MEAN NOMINMAX
  _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
target_link_libraries(gufo_posix_compat PUBLIC ws2_32 psapi)
file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/win32-compat-links-$<CONFIG>.txt CONTENT
  "$<TARGET_PROPERTY:gufo_posix_compat,LINK_LIBRARIES>\n$<TARGET_PROPERTY:gufo_posix_compat,INTERFACE_LINK_LIBRARIES>\n")

function(gufo_windows_targets directory)
  get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_target_property(type ${target} TYPE)
    if (NOT target STREQUAL "gufo_posix_compat" AND NOT type STREQUAL "UTILITY")
      target_link_libraries(${target} PRIVATE gufo_posix_compat)
      if (type STREQUAL "EXECUTABLE")
        target_link_options(${target} PRIVATE LINKER:/STACK:8388608)
      endif()
    endif()
  endforeach()
  get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(child IN LISTS children)
    gufo_windows_targets("${child}")
  endforeach()
endfunction()
cmake_language(DEFER CALL gufo_windows_targets "${PROJECT_SOURCE_DIR}")

if (BUILD_TESTING)
  add_executable(win32_compat_test ${PROJECT_SOURCE_DIR}/tests/platform/win32_compat_test.cpp)
  target_compile_options(win32_compat_test PRIVATE -UNDEBUG)
  target_link_libraries(win32_compat_test PRIVATE gufo_cache)
  add_test(NAME win32_compat_test COMMAND win32_compat_test)
  set_tests_properties(win32_compat_test PROPERTIES LABELS "cpu;windows" TIMEOUT 30)
endif()
