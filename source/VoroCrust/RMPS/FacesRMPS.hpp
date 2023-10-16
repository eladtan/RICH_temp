#ifndef FACES_RMPS
#define FACES_RMPS

#include <vector>
#include <boost/random.hpp>

#include "../PLC/PL_Complex.hpp"
#include "source/3D/elementary/Vector3D.hpp"
#include "../VoroCrust_kd_tree/VoroCrust_kd_tree.hpp"
#include "../trees.hpp"

struct EligbleFace {
    std::vector<Vector3D> face;
    std::size_t patch_index;
    std::size_t plc_index;
    double area;

    EligbleFace() :  face(), patch_index(0), plc_index(0), area(0.0) {}
    EligbleFace(std::vector<Vector3D> const& face_, 
                std::size_t const patch_index_, 
                std::size_t const plc_index_, 
                double const area_) 
                    : face(face_), 
                      patch_index(patch_index_), 
                      plc_index(plc_index_), 
                      area(area_) {}

    Vector3D const& operator [] (std::size_t const index) const {
        return face[index];
    }

    Vector3D& operator [] (std::size_t const index) {
        return face[index];
    }

    /*! \brief returns true if Face is contained in the ball `<center, r>`
    */
    bool isContainedInBall(Vector3D const& center, double const r) const;
};

class FacesRMPS {
    public:
        FacesRMPS(double const maxRadius_, double const L_Lipschitz_, double const alpha_, double const sharpTheta_, std::shared_ptr<PL_Complex> const& plc_);

        void loadFaces(std::vector<VoroCrust::Face> const& faces);
        
        bool doSampling(VoroCrust_KD_Tree_Ball &faces_ball_tree, Trees const& trees);

    private:
        double const maxRadius;
        double const L_Lipschitz;
        double const alpha;
        double const sharpTheta;
        double const rejection_probability = 0.1;

        boost::variate_generator<boost::mt19937, boost::uniform_01<>> uni01_gen;

        std::shared_ptr<PL_Complex const> plc;
        std::vector<EligbleFace> eligble_faces;
        std::vector<int> isDeleted;

        std::pair<double const, std::vector<double> const> calculateTotalAreaAndStartAreaOfEligbleFaces() const;

        /*! \brief divides the eligble faces, each triangle is divided into four triangles by connecting the midpoints of the edges. They new triangles maintaine the same orientation as the face 
        */
        void divideEligbleFaces();

        /*! \brief check if `p` is deeply covered by a ball in trees
        */
        bool checkIfPointIsDeeplyCovered(Vector3D const& p, Trees const& trees) const;

        /*! \brief sample a point from the eligble faces
        */
        std::tuple<bool, std::size_t const, Vector3D const> sampleEligbleFaces(double const total_area, std::vector<double> const& start_area);

        bool discardEligbleFaces(Trees const& trees);

        void discardEligbleFacesContainedInCornerBalls(Trees const& trees);

        void discardEligbleFacesContainedInEdgeBalls(Trees const& trees);

        double calculateSmoothnessLimitation(Vector3D const& p, EligbleFace const& face_sampled, Trees const& trees) const;

        bool isEligbleFaceDeeplyCoveredInFaceBall(EligbleFace const& face, Trees const& trees, std::size_t ball_index) const;

        double calculateInitialRadius(Vector3D const& point, std::size_t const face_index, Trees const& trees) const;
};
#endif // FACES_RMPS
