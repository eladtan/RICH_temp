#ifndef GRAVITY_ACC_3D_HPP
#define GRAVITY_ACC_3D_HPP

#ifdef RICH_MPI
    #include "3D/gravity/DistributedGravityCalculator.hpp"
#else // RICH_MPI
    #include "3D/gravity/GravityTree.hpp"
#endif // RICH_MPI

#include "newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "misc/memory_profile.hpp"

class GravityAcceleration3D : public Acceleration3D
{
public:
    GravityAcceleration3D(double theta, bool quadrupole = false, double G = 1): theta(theta), quadrupole(quadrupole), G(G), lastWalkTime_(0){};

    double getLastWalkTime() const { return lastWalkTime_; }

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& fluxes, const double time, vector<Vector3D> &acc) const
    {
        (void) fluxes;
        (void) time;
        MEMORY_PROFILE_SCOPE("gravity source");

        size_t N = tess.GetPointNo();
        this->points_.resize(N);
        this->masses_.resize(N);
        for(size_t cellIdx = 0; cellIdx < N; cellIdx++)
        {
            this->points_[cellIdx] = tess.GetCellCM(cellIdx);
            this->masses_[cellIdx] = (cells[cellIdx].density) * (tess.GetVolume(cellIdx));
        }
        #ifdef RICH_MPI
            DistributedGravityCalculator agent(tess, this->masses_, this->theta, this->quadrupole);
            acc = agent.getAcceleration(this->points_);
            lastWalkTime_ = agent.getWalkTime();
        #else // RICH_MPI
            std::pair<Vector3D, Vector3D> boundaries = tess.GetBoxCoordinates();
            GravityTree<Vector3D> gravTree(boundaries.first, boundaries.second, this->theta, this->quadrupole);
            this->massed_points_.clear();
            this->massed_points_.reserve(N);
            for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
            {
                this->massed_points_.emplace_back(MassedPoint<Vector3D>(this->points_[pointIdx], this->masses_[pointIdx]));
            }
            {
                MEMORY_PROFILE_SCOPE("gravity tree build");
                gravTree.build(this->massed_points_);
            }
            
            double wt0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            acc.resize(N);
            for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
            {
                acc[pointIdx] = gravTree.gravity(this->points_[pointIdx]);
            }
            lastWalkTime_ = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() - wt0;
        #endif // RICH_MPI

        for(size_t i = 0; i < N; ++i)
            acc[i] *= this->G;
    }

private:
    double theta, G;
    bool quadrupole;
    mutable double lastWalkTime_;
    mutable std::vector<Vector3D> points_;
    mutable std::vector<gravity_result_t> masses_;
    #ifndef RICH_MPI
        mutable std::vector<MassedPoint<Vector3D>> massed_points_;
    #endif // RICH_MPI
};

#endif // GRAVITY_ACC_3D_HPP
