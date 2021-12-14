cmake_minimum_required(VERSION 3.0)
set(PROJECT_NAME "ValidOpenCVInstall")
project( ${PROJECT_NAME} )

# add opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# compile
add_executable( ${PROJECT_NAME}.out valid_installation_opencv.cpp )
target_link_libraries( ${PROJECT_NAME}.out ${OpenCV_LIBS} )

# compile
add_executable( "FindSquare" FindSquare.cpp )
target_link_libraries( "FindSquare" ${OpenCV_LIBS} )

# compile
add_executable( "ResultAnalysis" resultAnalysis.cpp )
target_link_libraries( "ResultAnalysis" ${OpenCV_LIBS} )