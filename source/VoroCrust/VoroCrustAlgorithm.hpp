#ifndef VOROCRUST_ALGORITHM
#define VOROCRUST_ALGORITHM

#include "PLC/PL_Complex.hpp"
#include "trees.hpp"
#include "RMPS/CornersRMPS.hpp"
#include "RMPS/EdgesRMPS.hpp"
#include "RMPS/FacesRMPS.hpp"
#include "RMPS/SliverDriver.hpp"
#include <sstream>
#include <filesystem>
#include <memory>


class VoroCrustAlgorithm {
    public:
        std::shared_ptr<PL_Complex> plc;
        Trees trees;

        VoroCrustAlgorithm( PL_Complex const& plc_,
                            double const sharpTheta_,
                            double const flatTheta_,
                            double const maxRadius_,
                            double const L_Lipschitz_,
                            double const alpha_,
                            std::size_t const maximal_num_iter_,
                            std::size_t const num_of_samples_edges_,
                            std::size_t const num_of_samples_faces);

        /*! \brief runs the VoroCrust Algorithm*/
        void run();

        std::vector<Seed> getSeeds() const;

        std::string repr() const;

        std::vector<std::vector<Seed>> randomSampleSeeds(std::vector<PL_Complex> const& zones_plcs, std::vector<std::vector<Seed>> const& zones_boundary_seeds, double const maxSize);

        void dump(std::filesystem::path const& dirname) const;

        void load_dump(std::filesystem::path const& dirname);

        bool pointOutSidePLC(PL_Complex const& plc, Vector3D const& p);

        bool pointInSidePLC(PL_Complex const& plc, Vector3D const& p);
        
    private:
        double const sharpTheta;
        double const flatTheta;
        double const maxRadius;
        double const L_Lipschitz;
        double const alpha;
        std::size_t const maximal_num_iter;
        std::size_t const num_of_samples_edges;
        std::size_t const num_of_samples_faces;

        CornersRMPS cornersDriver;
        EdgesRMPS edgesDriver;
        FacesRMPS facesDriver;
        SliverDriver sliverDriver;

        /*! \brief enforces the Lipschitzness for a strata ball_tree
            \return true if some ball shrunk
        */
        bool enforceLipschitzness(VoroCrust_KD_Tree_Ball& ball_tree);

        bool sliverElimination();
};

VoroCrust_KD_Tree_Ball makeSeedBallTree(std::vector<Seed> const& seeds);

std::vector<std::vector<Seed>> determineZoneOfSeeds(std::vector<Seed> const& seeds, std::vector<PL_Complex> const& zone_plcs);

std::vector<Seed> getSeedsFromBallTree(VoroCrust_KD_Tree_Ball const& ball_tree);

void dumpSeeds(std::filesystem::path const& dirname, std::vector<Seed> const& seeds);

std::vector<Seed> load_dumpSeeds(std::filesystem::path const& dirname);

#endif /* VOROCRUST_ALGORITHM */