cmake_minimum_required(VERSION 2.4.6)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/cuda")

SET(CUDA_TOOLKIT_ROOT "/usr/local/cuda")
FIND_PACKAGE(OsgCompute)
INCLUDE(FindCuda)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

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

SET(OSG_LIBS ${OPENTHREADS_LIBRARIES} ${OSG_LIBRARIES} ${OSGVIEWER_LIBRARIES} ${OSGGA_LIBRARIES} ${OSGDB_LIBRARIES} ${OSGWIDGET_LIBRARIES} ${OSGUTIL_LIBRARIES} ${OSGTEXT_LIBRARIES} ${OSGPARTICLE_LIBRARIES})

SET(POCO_FIND_REQUIRED 1)
FIND_PACKAGE(Poco)

SET(JANSSON_FIND_REQUIRED 1)
FIND_PACKAGE(jansson)

SET(FLYVR_PATH "../flyvr")
SET(FLYVR_INCLUDE_PATH "${FLYVR_PATH}/include")

FIND_LIBRARY(FLYVR_LIBRARY
	NAMES flyvr
	PATHS "${FLYVR_PATH}/lib")

IF (FLYVR_LIBRARY)
  MESSAGE(STATUS "Found flyvr: ${FLYVR_LIBRARY}")
ELSE(FLYVR_LIBRARY)
  MESSAGE(FATAL_ERROR "Could not find flyvr library")
ENDIF(FLYVR_LIBRARY)

add_library(StimulusAltitudeEdge SHARED src/StimulusAltitudeEdge.cpp)
target_link_libraries(StimulusAltitudeEdge ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES} ${FLYVR_LIBRARY})

add_library(StimulusOSGFile SHARED src/StimulusOSGFile.cpp src/json2osg.cpp)
target_link_libraries(StimulusOSGFile ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES} ${FLYVR_LIBRARY})

add_library(StimulusStarField SHARED src/StimulusStarField.cpp)
target_link_libraries(StimulusStarField ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES} ${FLYVR_LIBRARY})

add_library(StimulusStarFieldAndModel SHARED src/StimulusStarFieldAndModel.cpp src/json2osg.cpp)
target_link_libraries(StimulusStarFieldAndModel ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES} ${FLYVR_LIBRARY})

add_library(StimulusCylinder SHARED src/StimulusCylinder.cpp src/json2osg.cpp)
target_link_libraries(StimulusCylinder ${OSG_LIBS} ${POCO_LIBRARIES}
  ${JANSSON_LIBRARIES} ${FLYVR_LIBRARY})

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

  MESSAGE(STATUS "CUDA_FILES: ${CUDA_FILES}")

  add_library(StimulusCUDAStarFieldAndModel SHARED ${TARGET_SRC})

  INCLUDE_DIRECTORIES(
    ${OSG_COMPUTE_INCLUDE_PATHS}
    ${CUDA_TOOLKIT_INCLUDE}
    ${FLYVR_INCLUDE_PATH}
  )
  target_link_libraries(StimulusCUDAStarFieldAndModel
    ${OSG_LIBS} ${POCO_LIBRARIES} ${JANSSON_LIBRARIES} ${OSG_COMPUTE_LIBRARIES}
    ${FLYVR_LIBRARY})
ENDIF(OSG_COMPUTE_FOUND)
