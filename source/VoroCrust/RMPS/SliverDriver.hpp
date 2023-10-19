#ifndef SLIVER_DRIVER
#define SLIVER_DRIVER

#include "../trees.hpp"
#include <array>

enum class Dim {
    CORNER,
    EDGE,
    FACE
};

using Ball = std::pair<Vector3D, double const>; // center, radius
inline double getR(Ball const& b) { return b.second; }

struct Seed {
    Vector3D p;
    double radius;

    Seed() : p(0.0, 0.0, 0.0), radius(-1.0) {}
    Seed(Vector3D const& p_, double radius_) : p(p_), radius(radius_) {}
    Seed(Ball const& ball) : p(ball.first), radius(ball.second) {}
};

struct BallInfo {
    std::size_t index;
    Dim dim;

    BallInfo(std::size_t const index_, Dim const dim_) : index(index_), dim(dim_) {}

    friend bool operator==(BallInfo const& lhs, BallInfo const& rhs) { return (lhs.index == rhs.index) && (lhs.dim == rhs.dim); }
};

using Triplet = std::pair<std::size_t, std::size_t>; // Triplet is p, Triplet.first, Triplet.second
using InfoQuartet = std::array<BallInfo const, 4>;
using BallQuartet = std::array<Ball const, 4>;

// part of calculating the intersection seed algorithm
std::tuple<double const, double const, double const, double const> 
getLineCoeff(double const x1, 
             double const y1, 
             double const z1, 
             double const r1, 
             double const x2, 
             double const y2, 
             double const z2, 
             double const r2);

// part of calculating the intersection seed algorithm
std::pair<double const, double const> 
getZDependency(double const a1, 
               double const b1, 
               double const c1, 
               double const k1, 
               double const a3, 
               double const b3, 
               double const c3, 
               double const k3);


class SliverDriver {
    public:
        SliverDriver(double const L_Lipschitz_);

        bool eliminateSlivers(Trees &trees);

        std::vector<Seed> getSeeds(Trees const& trees) const;

    private:
        double const L_Lipschitz;
        std::vector<double> r_new_corner_balls;
        std::vector<double> r_new_edge_balls;
        std::vector<double> r_new_face_balls;

        std::size_t number_of_slivers_eliminated;
        mutable double max_radius_corner;
        mutable double max_radius_edge;
        
        /*! \brief eliminate Slivers created from a specific ball tree. in practice we only use this function on the faces ball tree
        */
        void eliminateSliversForBallsInBallTree(Dim const dim, Trees const& trees);

        void dealWithBall(BallInfo const& ball_info, Trees const& trees);

        void dealWithTriplets(BallInfo const& ball_info_1, Triplet const& triplet, std::vector<BallInfo> const& overlapping_balls, Trees const& trees);

        std::tuple<bool, Vector3D, Vector3D> calculateIntersectionSeeds(Ball const& ball_1, Ball const& ball_2, Ball const& ball_3) const;

        std::vector<BallInfo> groupOverlappingBalls(BallInfo const& ball, Trees const& trees) const;

        std::vector<Triplet> formTripletsOfOverlappingBalls(std::vector<BallInfo> const& overlapping_balls, Trees const& trees) const;

        void dealWithHalfCoveredSeeds(InfoQuartet const& balls_info, BallQuartet const& balls);

        Ball getBall(BallInfo const& ball_info, Trees const& trees) const;

        void setRadiusOfBall(double const r_new, BallInfo const& ball_info);

};

#endif // SLIVER_DRIVER