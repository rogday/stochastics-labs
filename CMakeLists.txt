cmake_minimum_required (VERSION 3.10)

project (stohastics)

set(CMAKE_CXX_STANDARD 17)

set(LAB1 lab1.cpp)
set(LAB2-test lab2.cpp)
set(LAB2 gachibass.cpp)

add_executable(lab1 ${LAB1})
add_executable(lab2-test ${LAB2-test})
add_executable(lab2 ${LAB2})