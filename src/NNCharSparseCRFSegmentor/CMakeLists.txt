
include_directories(
  model
)

add_executable(NNCharSparseCRFSegmentor NNCRFSegmentor.cpp)
target_link_libraries(NNCharSparseCRFSegmentor ${LIBS})
