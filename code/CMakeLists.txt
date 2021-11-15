cmake_minimum_required(VERSION 3.0)
set(PROJECT_NAME "MuseeSecuriseVirtuel-MethodeNaive")
project( ${PROJECT_NAME} )

# add opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# compile
add_executable( ${PROJECT_NAME}.out catmap.cpp )
target_link_libraries( ${PROJECT_NAME}.out ${OpenCV_LIBS} )