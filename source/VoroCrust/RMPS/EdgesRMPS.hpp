#ifndef EDGES_RMPS
#define EDGES_RMPS

#include "../PLC/PL_Complex.hpp"
#include "source/3D/elementary/Vector3D.hpp"
#include "../VoroCrust_kd_tree/VoroCrust_kd_tree.hpp"
#include "../trees.hpp"
#include <vector>
#include <array>
#include <boost/random.hpp>

struct EligbleEdge {
    std::array<Vector3D, 2> edge;
    std::size_t crease_index;
    std::size_t plc_index;

    EligbleEdge() : edge(), crease_index(0), plc_index(0) {}
    EligbleEdge(Vector3D const& v1, 
                Vector3D const& v2, 
                std::size_t const crease_index_, 
                std::size_t const plc_index_) 
                    : edge({v1, v2}), 
                      crease_index(crease_index_), 
                      plc_index(plc_index_) {}

    Vector3D const& operator [] (std::size_t const index) const {
        return edge[index];
    }

    Vector3D& operator [] (std::size_t const index)  {
        return edge[index];
    }
};

class EdgesRMPS {
    public:
        EdgesRMPS(double const maxRadius_, double const L_Lipschitz_, double const alpha_, double const sharpTheta_, std::shared_ptr<PL_Complex> const& plc_);

        //! \brief loads the sharp edges vector to the eligble edge vector
        void loadEdges(std::vector<VoroCrust::Edge> const& sharp_edges);

        //! \brief do the RMPS sampling stage until there are no more eligble edges 
        bool doSampling(VoroCrust_KD_Tree_Ball &edges_ball_tree, Trees const& tree);

    private:
        double const maxRadius;
        double const L_Lipschitz;
        double const alpha;
        double const sharpTheta;
        double const rejection_probability = 0.1;

        boost::variate_generator<boost::mt19937, boost::uniform_01<>> uni01_gen;
        
        std::shared_ptr<PL_Complex const> plc;
        std::vector<EligbleEdge> eligble_edges;
        std::vector<bool> isDeleted;

        /*! \brief calculates the current total length of the eligble edges and the start len for each edge
            \return returns a pair <total_length, start_length> where start_length is a vector with the same size as the eligble edges
        */
        std::pair<double const, std::vector<double> const> calculateTotalLengthAndStartLengthOfEligbleEdges() const;

        //! \brief divides the eligble edges to two eligble edges at the midpoint
        void divideEligbleEdges();

        //! \brief checks if `p` is deeply covered by any ball in edges_ball_tree
        bool checkIfPointIsDeeplyCovered(Vector3D const& p, Trees const& trees) const;

        /*! \brief sample the eligble edges
            \return <success, edge_index, sample> where success is a bool flag and is false if sampling failed
        */
        std::tuple<bool, std::size_t const, Vector3D const> sampleEligbleEdges(double const total_len, std::vector<double> const& start_len);

        //! \brief discard any eligble edges that meet the critrea for discardtion 
        bool discardEligbleEdges(Trees const& trees);

        //! \brief discard any eligble edge fully contained inside a corner ball
        void discardEligbleEdgesContainedInCornerBalls(Trees const& trees);

        //! \brief returns the maximal radius satisfying the cosmoothness limitation
        double calculateSmoothnessLimitation(Vector3D const& p, EligbleEdge const& edge_sampled, Trees const& trees) const;

        //! \brief return true if eligble edge is deeply covered by edge ball 
        bool isEligbleEdgeDeeplyCoveredInEdgeBall(EligbleEdge const& edge, Trees const& trees, std::size_t const ball_index) const;

        //! \brief retunrs the initial radius of a new sample defined by point
        double calculateInitialRadius(Vector3D const& point, std::size_t const edge_index, Trees const& trees) const;

};

#endif // EDGES_RMPS