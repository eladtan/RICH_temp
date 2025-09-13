# --- Config you can tweak ---
# Which file lists patterns (relative to project root)

set(DEBUG_FILES_LIST "${CMAKE_SOURCE_DIR}/../${DEBUG_FILES}")

if(${MIXED_BUILD})
    # if not defined "DEBUG_FILES"
    if(NOT DEFINED DEBUG_FILES)
        message(FATAL_ERROR "MIXED build requires DEBUG_FILES to be set")
    endif()
    # make sure the file exists
    if(NOT EXISTS "${DEBUG_FILES_LIST}")
        message(FATAL_ERROR "DEBUG_FILES file not found: ${DEBUG_FILES_LIST}")
    endif()
endif()


# Which file extensions are considered "compile units"
set(DEBUG_SOURCE_EXTS c;cc;cxx;cpp)

# --- Helpers ---

# Expand one pattern (relative or absolute) into a list of source files
function(_expand_pattern_to_sources OUT_LIST PATTERN)
  # Normalize to absolute path relative to source dir
  if(NOT IS_ABSOLUTE "${PATTERN}")
    set(PATTERN "${CMAKE_SOURCE_DIR}/${PATTERN}")
  endif()

  # If it's a directory, treat like "dir/**"
  if(IS_DIRECTORY "${PATTERN}")
    set(_patterns)
    foreach(ext IN LISTS DEBUG_SOURCE_EXTS)
      list(APPEND _patterns "${PATTERN}/**/*.${ext}")
    endforeach()
  else()
    # Wildcards?
    string(FIND "${PATTERN}" "*" has_star)
    string(FIND "${PATTERN}" "?" has_q)
    if(has_star GREATER -1 OR has_q GREATER -1)
      set(_patterns)
      # If user wrote "dir/*", expand recursively to sources
      foreach(ext IN LISTS DEBUG_SOURCE_EXTS)
        list(APPEND _patterns "${PATTERN}/**/*.${ext}" "${PATTERN}.${ext}")
      endforeach()
      # Also include the literal pattern (in case it already includes *.cpp etc.)
      list(APPEND _patterns "${PATTERN}")
    else()
      # Looks like a single file path (maybe without extension)
      if(EXISTS "${PATTERN}")
        set(${OUT_LIST} "${PATTERN}" PARENT_SCOPE)
        return()
      else()
        # Try appending known source extensions
        set(candidates)
        foreach(ext IN LISTS DEBUG_SOURCE_EXTS)
          if(EXISTS "${PATTERN}.${ext}")
            list(APPEND candidates "${PATTERN}.${ext}")
          endif()
        endforeach()
        if(candidates)
          set(${OUT_LIST} "${candidates}" PARENT_SCOPE)
        else()
          # Not found — return empty quietly
          set(${OUT_LIST} "" PARENT_SCOPE)
        endif()
        return()
      endif()
    endif()
  endif()

  # GLOB_RECURSE with CONFIGURE_DEPENDS so changes to the tree re-trigger CMake
  set(found)
  foreach(globpat IN LISTS _patterns)
    file(GLOB_RECURSE hits CONFIGURE_DEPENDS "${globpat}")
    if(hits)
      list(APPEND found ${hits})
    endif()
  endforeach()
  if(found)
    list(REMOVE_DUPLICATES found)
  endif()
  set(${OUT_LIST} "${found}" PARENT_SCOPE)
endfunction()

# Read debug_files.txt -> list of *absolute* source files to mark
function(_collect_debug_sources OUT_LIST)
  if(NOT EXISTS "${DEBUG_FILES_LIST}")
    set(${OUT_LIST} "" PARENT_SCOPE)
    return()
  endif()

  file(READ "${DEBUG_FILES_LIST}" _raw)
  # Normalize line endings
  string(REPLACE "\r\n" "\n" _raw "${_raw}")
  string(REPLACE "\r"   "\n" _raw "${_raw}")
  string(REGEX REPLACE "^[ \t]*#.*$" "" _raw "${_raw}")           # strip full-line comments
  string(REGEX REPLACE "\n[ \t]*#.*$" "\n" _raw "${_raw}")        # strip trailing comments
  string(REGEX REPLACE "^[ \t]*\n" "" _raw "${_raw}")             # drop leading blanks
  string(REGEX REPLACE "\n[ \t]*\n" "\n" _raw "${_raw}")          # collapse blank lines
  string(STRIP "${_raw}" _raw)

  if(_raw STREQUAL "")
    set(${OUT_LIST} "" PARENT_SCOPE)
    return()
  endif()

  separate_arguments(_lines UNIX_COMMAND "${_raw}")

  set(all_debug_srcs)
  foreach(item IN LISTS _lines)
    string(STRIP "${item}" item)
    if(item STREQUAL "")
      continue()
    endif()
    _expand_pattern_to_sources(expanded "${item}")
    if(expanded)
      list(APPEND all_debug_srcs ${expanded})
    endif()
  endforeach()

  if(all_debug_srcs)
    list(REMOVE_DUPLICATES all_debug_srcs)
  endif()
  set(${OUT_LIST} "${all_debug_srcs}" PARENT_SCOPE)
endfunction()

# Public: mark the collected sources on a given target
function(mark_debug_sources target)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "mark_debug_sources: target '${target}' does not exist")
  endif()

  _collect_debug_sources(srcs)
  if(NOT srcs)
    # Nothing to do
    return()
  endif()

  # Apply per-file compile options
  foreach(f IN LISTS srcs)
    message(STATUS "${target} — marking '${f}' for debug")
    # Only set on files that are part of this target; harmless otherwise,
    # but the following guard avoids noisy diagnostics on some generators.
    get_target_property(t_sources "${target}" SOURCES)
    if(t_sources)
      list(FIND t_sources "${f}" _idx)
      if(_idx EQUAL -1)
        # File not explicitly listed — still set; many projects add sources later via target_sources.
        # You can comment the next line if you prefer strictness.
      endif()
    endif()

    set_source_files_properties("${f}" PROPERTIES
      COMPILE_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}"
    )
  endforeach()

  # (Optional) show a short summary
  message(STATUS "mark_debug_sources: ${target} — ${_list_len} files in debug mode")
endfunction()
