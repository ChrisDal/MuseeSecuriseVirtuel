cmake_minimum_required(VERSION 3.0)
set(PROJECT_NAME "MuseeSecuriseVirtuel")
project( ${PROJECT_NAME} )

# add opencv 
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# compile
add_executable( ${PROJECT_NAME}-MethodeNaive.out blockPermutation.cpp )
target_link_libraries( ${PROJECT_NAME}-MethodeNaive.out ${OpenCV_LIBS} )

#compile 
add_executable( ${PROJECT_NAME}-Detection.out oeuvredetection.cpp )
target_link_libraries( ${PROJECT_NAME}-Detection.out ${OpenCV_LIBS} )

#compile 
add_executable( ${PROJECT_NAME}-ExportImage.out createOeuvre.cpp )
target_link_libraries( ${PROJECT_NAME}-ExportImage.out ${OpenCV_LIBS} )