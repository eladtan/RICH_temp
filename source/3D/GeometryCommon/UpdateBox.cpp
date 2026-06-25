#include <cassert>
#include "UpdateBox.hpp"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <MeshDecomposer3D/kernels/Rectangle.hpp>

void UpdateBox(Voronoi3D &tess, Simulation &sim, double const min_velocity, double const volume_fraction, ComputationalCell3D const& reference_cell)
{
	std::vector<ComputationalCell3D>& cells = sim.getCells();
	std::vector<Conserved3D>& extensives = sim.getExtensives();
    size_t const N = tess.GetPointNo();
    Vector3D maxv(-1e200, -1e200, -1e200), minv(1e200, 1e200, 1e200);
    double maxR = 0;
    for (size_t i = 0; i < N; ++i)
    {
        if (fastabs(cells[i].velocity) > min_velocity)
        {
            Vector3D const& p = tess.GetMeshPoint(i);
            maxv.x = std::max(maxv.x, p.x);
            maxv.y = std::max(maxv.y, p.y);
            maxv.z = std::max(maxv.z, p.z);
            minv.x = std::min(minv.x, p.x);
            minv.y = std::min(minv.y, p.y);
            minv.z = std::min(minv.z, p.z);
            maxR = std::max(maxR, tess.GetWidth(i));
        }
    }
    std::array<double, 7> tempvec, temprecv;
    tempvec[0] = maxR;
    tempvec[1] = maxv.x;
    tempvec[2] = maxv.y;
    tempvec[3] = maxv.z;
    tempvec[4] = -minv.x;
    tempvec[5] = -minv.y;
    tempvec[6] = -minv.z;
	temprecv= tempvec;
	int rank = 0;
#ifdef RICH_MPI
    MPI_Allreduce(&tempvec[0], &temprecv[0], 7, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    Vector3D const cur_max = tess.GetBoxCoordinates().second; 
	Vector3D const cur_min = tess.GetBoxCoordinates().first;
    double const oldv = (cur_max.x - cur_min.x) * (cur_max.y - cur_min.y) * (cur_max.z - cur_min.z);
    temprecv[0] = std::max(temprecv[0], std::pow(oldv, 0.333333333) * 0.03);
    // Do we need to resize?
    Vector3D recvmax(temprecv[1] + 5 * temprecv[0], temprecv[2] + 5 * temprecv[0],
        temprecv[3] + 5 * temprecv[0]), recvmin(-temprecv[4] - 5 * temprecv[0],
            -temprecv[5] - 5 * temprecv[0], -temprecv[6] - 5 * temprecv[0]);
    bool recalc = false;
		
	if (recvmax.x > cur_max.x)
	{
		recalc = true;
		recvmax.x = cur_max.x + 5 * temprecv[0];
	}
	if (recvmax.y > cur_max.y)
	{
		recalc = true;
		recvmax.y = cur_max.y + 5 * temprecv[0];
	}
	if (recvmax.z > cur_max.z)
	{
		recalc = true;
		recvmax.z = cur_max.z + 5 * temprecv[0];
	}
	if (recvmin.x < cur_min.x)
	{
		recalc = true;
		recvmin.x = cur_min.x - 5 * temprecv[0];
	}
	if (recvmin.y < cur_min.y)
	{
		recalc = true;
		recvmin.y = cur_min.y - 5 * temprecv[0];
	}
	if (recvmin.z < cur_min.z)
	{
		recalc = true;
		recvmin.z = cur_min.z - 5 * temprecv[0];
	}
	if (recalc)
	{
		recvmax.x = std::max(recvmax.x, cur_max.x);
		recvmax.y = std::max(recvmax.y, cur_max.y);
		recvmax.z = std::max(recvmax.z, cur_max.z);
		recvmin.x = std::min(recvmin.x, cur_min.x);
		recvmin.y = std::min(recvmin.y, cur_min.y);
		recvmin.z = std::min(recvmin.z, cur_min.z);
		// Get how many points to add
		double const newv = (recvmax.x - recvmin.x) * (recvmax.y - recvmin.y) * (recvmax.z - recvmin.z);
		size_t const Np = static_cast<size_t>((newv - oldv) / (volume_fraction * newv));
		if (rank == 0)
		{
			std::cout << "Doing resize rank " << rank << std::endl;
			std::cout << "Old box ll = " << cur_min.x << "," << cur_min.y << "," << cur_min.z << " ur = " << cur_max.x << "," << cur_max.y << "," << cur_max.z << std::endl;
			std::cout << "New box ll = " << recvmin.x << "," << recvmin.y << "," << recvmin.z << " ur = " << recvmax.x << "," << recvmax.y << "," << recvmax.z << std::endl;
			std::cout << "Max cell size " << temprecv[0] << std::endl;
			std::cout << "Old vol " << oldv << " New vol " << newv << std::endl;
			std::cout << "Point number " << Np << std::endl;
		}
		tess.SetBox(recvmin, recvmax);
		// tess.SetKernel(new Rectangle(recvmin, recvmax));		
		std::vector<Vector3D> mypoints = tess.getMeshPoints();
		mypoints.resize(N);
		cells.resize(N);
		// GetNewPoints
		size_t& MaxID = sim.GetMaxID();
		if(rank == 0)
		{
			boost::random::mt19937_64 generator(sim.GetCycle());
			boost::random::uniform_real_distribution<> dist;
			std::vector<Vector3D> newpoints;
			double ran[3];
			Vector3D point;
			size_t counter = 0;
			while (counter < Np)
			{
				ran[0] = dist(generator);
				ran[1] = dist(generator);
				ran[2] = dist(generator);
				point.x = ran[0] * (recvmax.x - recvmin.x) + recvmin.x;
				point.y = ran[1] * (recvmax.y - recvmin.y) + recvmin.y;
				point.z = ran[2] * (recvmax.z - recvmin.z) + recvmin.z;
				if (point.x<cur_min.x || point.y<cur_min.y || point.z<cur_min.z || point.x>cur_max.x || point.y>cur_max.y || point.z>cur_max.z)
				{
					mypoints.push_back(point);
					cells.push_back(reference_cell);
					cells.back().ID = MaxID + 1 + counter;
					++counter;
				}
			}
		}
		MaxID += Np;
		assert(N>0);
		
#ifdef RICH_MPI
		tess.BuildParallel(mypoints);
		MPI_exchange_data(tess, cells, false);
		MPI_exchange_data(tess, cells, true);
#else // RICH_MPI
		tess.Build(mypoints);
#endif // RICH_MPI

		extensives.resize(tess.GetPointNo());		
		for (size_t i = 0; i < tess.GetPointNo(); ++i)
			PrimitiveToConserved(cells.at(i), tess.GetVolume(i), extensives.at(i));
	}
}
