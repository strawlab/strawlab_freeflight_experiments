cmake_minimum_required(VERSION 2.8.3)
project(strawlab_freeflight_experiments)
find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
  freemovr_engine
)

MESSAGE(STATUS "Setting CMAKE_MODULE_PATH: ${CMAKE_SOURCE_DIR}/strawlab_freeflight_experiments/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/strawlab_freeflight_experiments/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/strawlab_freeflight_experiments/cmake/Modules/cuda")

SET(CUDA_TOOLKIT_ROOT "/usr/local/cuda")
FIND_PACKAGE(OsgCompute)
INCLUDE(FindCuda)

FIND_PACKAGE(OpenThreads)
FIND_PACKAGE(osg)
FIND_PACKAGE(osgUtil)
FIND_PACKAGE(osgDB)
FIND_PACKAGE(osgText)
FIND_PACKAGE(osgParticle)
FIND_PACKAGE(osgTerrain)
FIND_PACKAGE(osgViewer)
FIND_PACKAGE(osgGA)
FIND_PACKAGE(osgWidget)
FIND_PACKAGE(osgParticle)

find_package(geometry_msgs)

SET(POCO_FIND_REQUIRED 1)
FIND_PACKAGE(Poco)

SET(JANSSON_FIND_REQUIRED 1)
FIND_PACKAGE(jansson)

SET(OSG_LIBS ${OPENTHREADS_LIBRARIES} ${OSG_LIBRARIES} ${OSGVIEWER_LIBRARIES} ${OSGGA_LIBRARIES} ${OSGDB_LIBRARIES} ${OSGWIDGET_LIBRARIES} ${OSGUTIL_LIBRARIES} ${OSGTEXT_LIBRARIES} ${OSGPARTICLE_LIBRARIES})

catkin_python_setup()

add_message_files(
  FILES
  CylinderGratingInfo.msg
)

generate_messages(DEPENDENCIES geometry_msgs)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  )

add_library(StimulusAltitudeEdge SHARED src/StimulusAltitudeEdge.cpp)
target_link_libraries(StimulusAltitudeEdge ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES})

add_library(StimulusOSGFile SHARED src/StimulusOSGFile.cpp src/json2osg.cpp)
target_link_libraries(StimulusOSGFile ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES})

add_library(StimulusStarField SHARED src/StimulusStarField.cpp)
target_link_libraries(StimulusStarField ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES})

add_library(StimulusStarFieldAndModel SHARED src/StimulusStarFieldAndModel.cpp src/json2osg.cpp)
target_link_libraries(StimulusStarFieldAndModel ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES})

add_library(StimulusCylinder SHARED src/StimulusCylinder.cpp src/json2osg.cpp)
target_link_libraries(StimulusCylinder ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES})

add_library(StimulusCylinderAndModel SHARED src/StimulusCylinderAndModel.cpp src/json2osg.cpp)
target_link_libraries(StimulusCylinderAndModel ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES})

add_library(StimulusCylinderGrating SHARED src/StimulusCylinderGrating.cpp src/json2osg.cpp)
target_link_libraries(StimulusCylinderGrating ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES})

SET(MY_CUDA_SOURCE_FILES
	src/PtclKernels.cu
)

IF(OSG_COMPUTE_FOUND)
  SET(CUDA_NVCC_FLAGS "--compiler-options '-fPIC'")
  CUDA_COMPILE( CUDA_FILES ${MY_CUDA_SOURCE_FILES} )

  # collect the sources
  SET(TARGET_SRC
    src/StimulusCUDAStarFieldAndModel.cpp src/json2osg.cpp
    ${CUDA_FILES}
  )

  MESSAGE(STATUS "TARGET_SRC: ${TARGET_SRC}")
  MESSAGE(STATUS "CUDA_FILES: ${CUDA_FILES}")

  add_library(StimulusCUDAStarFieldAndModel SHARED ${TARGET_SRC})

  INCLUDE_DIRECTORIES(
    ${OSG_COMPUTE_INCLUDE_PATHS}
    ${CUDA_TOOLKIT_INCLUDE}
    ${FREEMOVR_ENGINE_INCLUDE_PATH}
  )
  target_link_libraries(StimulusCUDAStarFieldAndModel
    ${OSG_LIBS} ${POCO_LIBRARIES} ${JANSSON_LIBRARIES} ${OSG_COMPUTE_LIBRARIES}
    )
ELSE(OSG_COMPUTE_FOUND)
  INCLUDE_DIRECTORIES(
    ${FREEMOVR_ENGINE_INCLUDE_PATH}
  )
ENDIF(OSG_COMPUTE_FOUND)

install(TARGETS StimulusAltitudeEdge StimulusOSGFile StimulusStarField StimulusStarFieldAndModel StimulusCylinder StimulusCylinderAndModel StimulusCylinderGrating StimulusCUDAStarFieldAndModel
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
