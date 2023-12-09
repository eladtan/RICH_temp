#ifndef SNAPSHOT_3D_HPP
#define SNAPSHOT_3D_HPP

#include <vector>
#include <string>
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "3D/elementary/Vector3D.hpp"

using namespace std;

//! \brief Container for snapshot data
class Snapshot3D
{
public:

	//! \brief Default constructor
	Snapshot3D(void);

	//! \brief Copy constructor
	//! \param source Source
	Snapshot3D(const Snapshot3D &source);

	//! \brief Mesh points
	std::vector<Vector3D> mesh_points;

#ifdef RICH_MPI
	//! \brief Processors Mesh points
	std::vector<Vector3D> proc_points; // TODO: maybe not necessary?
#endif

	//! \brief Volume of cells
	std::vector<double> volumes;

	//! \brief Computational cells
	std::vector<ComputationalCell3D> cells;

	//! \brief Time
	double time;

	//! \brief Cycle number
	int cycle;

	//! \brief THe names of the tracers and stickers
    std::pair<std::vector<std::string>, std::vector<std::string>> tracerstickernames;

	//! \brief Lower left corner
	Vector3D ll;

  //! \brief Upper right corner
	Vector3D ur;
};

#endif // SNAPSHOT_3D_HPP