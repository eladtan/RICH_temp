#include "CornersRMPS.hpp"
#include <boost/random.hpp>

CornersRMPS::CornersRMPS(double const maxRadius_, double const L_Lipschitz_, double const sharpTheta_, std::shared_ptr<PL_Complex> const& plc_) : maxRadius(maxRadius_), L_Lipschitz(L_Lipschitz_), sharpTheta(sharpTheta_), plc(plc_), eligble_corners() {}


void CornersRMPS::loadCorners(std::vector<VoroCrust::Vertex> const& sharp_corners){
    eligble_corners.clear();
    for(VoroCrust::Vertex const& corner_ptr : sharp_corners)
        eligble_corners.push_back(corner_ptr);
}

void CornersRMPS::doSampling(VoroCrust_KD_Tree_Ball &corner_ball_tree, Trees const& trees){

    Vector3D const empty_vector(0, 0, 0);

    while(not eligble_corners.empty()){

        // sample a vertex
        EligbleCorner const sample = newSample();

        double const radius = calculateInitialRadius(sample, trees);
        
        corner_ball_tree.insert(sample->vertex, empty_vector, radius, corner_ball_tree.size(), sample->index);
    }

    std::cout << "num of corner samples: " << corner_ball_tree.size() << std::endl;
}

EligbleCorner CornersRMPS::newSample(){
    // create a random number generator to sample from 0 - (eligble_corners.size()-1)
    boost::mt19937 rng(std::time(nullptr));
    boost::random::uniform_int_distribution<> int_distribution(0, eligble_corners.size() - 1);
    boost::variate_generator rand_gen(rng, int_distribution);

    // sample a random index;
    std::size_t const index = rand_gen();
    
    auto it = eligble_corners.cbegin();
    
    // advance iterator to that point
    for(std::size_t i=0; i < index; ++it, ++i);
    
    // erased sampled corner
    EligbleCorner sample = *it;
    eligble_corners.erase(it);

    return sample;
}

double CornersRMPS::calculateInitialRadius(EligbleCorner const& corner, Trees const& trees) const {
    VoroCrust_KD_Tree_Ball const& corner_ball_tree = trees.ball_kd_vertices;

    double r_q = std::numeric_limits<double>::max();
    double dist_q = 0.0;
    
    if(not corner_ball_tree.empty()){
        // find nearest ball center
        auto const& [nearestBallCenter, neareset_r] = corner_ball_tree.getBallNearestNeighbor(corner->vertex);

        r_q = neareset_r;
        dist_q = distance(corner->vertex, nearestBallCenter); // ||p-q||
    }
    
    double const dist_q_prime = calculateSmoothnessLimitation(corner, trees);

    return std::min({maxRadius, 0.49*dist_q_prime, r_q + L_Lipschitz*dist_q});
}

double CornersRMPS::calculateSmoothnessLimitation(EligbleCorner const& corner, Trees const& trees) const {
    VoroCrust_KD_Tree_Boundary const& corner_boundary_tree = trees.VC_kd_sharp_corners;
    VoroCrust_KD_Tree_Boundary const& edges_boundary_tree = trees.VC_kd_sharp_edges;
    VoroCrust_KD_Tree_Boundary const& faces_boundary_tree = trees.VC_kd_faces;

    // find nearest sharp corner, taking [1] beacause corner is a point in the tree so obviously it is the closest to itself
    auto const nearestSharpCorner_index = corner_boundary_tree.kNearestNeighbors(corner->vertex, 2)[1];
    Vector3D const& nearestSharpCorner = corner_boundary_tree.points[nearestSharpCorner_index];

    double const dist_nearest_corner = distance(corner->vertex, nearestSharpCorner); //||p-q^*||

    double min_dist = std::min(dist_nearest_corner, maxRadius/0.49); // maxRadius is divided because r_smooth is multiplied by 0.49

    // find neareset non cosmooth point on the creases current corner is part of 
    std::vector<std::size_t> creases_exclude;
    for(VoroCrust::Edge const& edge : corner->edges){
        if(not edge->isSharp) continue;

        std::size_t const crease_index = edge->crease_index;
        creases_exclude.push_back(crease_index);
        
        std::vector<Vector3D> parallel({edge->vertex2->vertex - edge->vertex1->vertex});

        min_dist = edges_boundary_tree.distanceToNearestNonCosmoothPoint(corner->vertex, parallel, crease_index, sharpTheta, min_dist);
    }

    // find the nearest point on the boundary tree that is on a crease current corner is not part of and limit min_dist
    min_dist = edges_boundary_tree.distanceToNearestNeighborExcludingFeatures(corner->vertex, creases_exclude, min_dist);

    // find the nearest non cosmooth point on the patches current corner is part of and limit min_dist 
    std::vector<std::size_t> patches_to_exclude;
    for(auto const& patch_faces : corner->divided_faces){
        std::size_t const patch_index = patch_faces[0]->patch_index;
        patches_to_exclude.push_back(patch_index);
        
        std::vector<Vector3D> normals;
        normals.reserve(patch_faces.size());        
        for(VoroCrust::Face const& face : patch_faces){
            normals.push_back(face->getNormal());
        }

        // find the nearest point on the patch not cosmooth to current corner and limit min_dist
        min_dist = faces_boundary_tree.distanceToNearestNonCosmoothPoint(corner->vertex, normals, patch_index, sharpTheta, min_dist);
    }
    
    // find the nearest point on a surface patch current corner is not part of and limit min_dist
    min_dist = faces_boundary_tree.distanceToNearestNeighborExcludingFeatures(corner->vertex, patches_to_exclude, min_dist);

    return min_dist;
}