#include "TimeStepUtils.hpp"

void CalcFaceVelocities(const Tessellation3D &tess, const vector<Vector3D> &point_vel, vector<Vector3D> &res)
{
	size_t N = tess.GetTotalFacesNumber();
	res.resize(N);
	for (size_t i = 0; i < N; ++i)
	{
		if (tess.BoundaryFace(i))
			res[i] = Vector3D();
		else
		{
			try
			{
				res[i] = tess.CalcFaceVelocity(i, point_vel[tess.GetFaceNeighbors(i).first], point_vel[tess.GetFaceNeighbors(i).second]);
			}
			catch (UniversalError & /*eo*/)
			{
				throw;
			}
		}
	}
}