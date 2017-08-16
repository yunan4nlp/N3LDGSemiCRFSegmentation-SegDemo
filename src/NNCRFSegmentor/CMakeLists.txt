
include_directories(
  model
)

add_executable(NNCRFSegmentor NNCRFSegmentor.cpp)
target_link_libraries(NNCRFSegmentor ${LIBS})
