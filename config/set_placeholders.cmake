set(R3D_MAX_VERTS "256")

configure_file(
    "${PROJECT_SOURCE_DIR}/opt/r3d/config/r3d-config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/opt/r3d/config/r3d-config.h"
    @ONLY
)
