#include "Snapshot3D.hpp"

Snapshot3D::Snapshot3D(void) : mesh_points(),
#ifdef RICH_MPI
    proc_points(),
#endif
    volumes(), cells(), time(), cycle(), tracerstickernames(), ll(Vector3D()), ur(Vector3D())
{
}

Snapshot3D::Snapshot3D(const Snapshot3D &source) : mesh_points(source.mesh_points),
#ifdef RICH_MPI
    proc_points(source.proc_points),
#endif
    volumes(source.volumes), cells(source.cells), time(source.time), cycle(source.cycle), tracerstickernames(source.tracerstickernames), ll(source.ll), ur(source.ur)
{
}
