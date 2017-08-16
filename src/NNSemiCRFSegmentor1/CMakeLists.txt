
include_directories(
  model
)

add_executable(NNSemiCRFSegmentor1 NNSemiCRFSegmentor.cpp)
target_link_libraries(NNSemiCRFSegmentor1 ${LIBS})
