# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Compile single file: `g++ -std=c++17 filename.cpp -o output_name`
- Build all: `g++ -std=c++17 *.cpp -o main`
- Run PSNN: `./PSNN`
- Run tester: `./tester`

## Code Style Guidelines
- Use 4-space indentation
- Follow standard C++ naming conventions: camelCase for variables/functions
- Place opening braces on the same line for functions and control structures
- Use `std::` prefix for standard library components (vs. using namespace)
- Include error handling with appropriate error messages
- Comment complex algorithms and functions with purpose and parameters
- Prefer modern C++ features (range-based for loops, auto, etc.)
- Use vector operations instead of raw arrays when possible
- Keep functions focused on a single task
- Use appropriate spacing around operators for readability