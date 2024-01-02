#include "SliverDriver.hpp"
#include <iostream>

std::vector<Seed> unsorted_unique(VoroCrust_KD_Tree_Ball const& seed_tree, double const tol);

SliverDriver::SliverDriver(double const L_Lipschitz_) : L_Lipschitz(L_Lipschitz_), 
                                                        r_new_corner_balls(), 
                                                        r_new_edge_balls(), 
                                                        r_new_face_balls(), 
                                                        number_of_slivers_eliminated(0), 
                                                        max_radius_corner(0.0),
                                                        max_radius_edge(0.0) {}

std::vector<BallInfo> SliverDriver::groupOverlappingBalls(BallInfo const& ball_info, Trees const& trees) const {
    auto const& [p, radius] = getBall(ball_info, trees);

    std::vector<BallInfo> overlapping_balls;

    double const r_max = (2.0 / (1.0 - L_Lipschitz)) * radius;

    //! EPSILONTICA:
    double const r_corner = (ball_info.dim == Dim::FACE) ? (max_radius_corner+2.0*radius)*(1.+ 1e-5) : r_max;
    double const r_edge   = (ball_info.dim == Dim::FACE) ? (max_radius_edge+2.0*radius)*(1.+ 1e-5) : r_max;

    // this lambda get the overlapping balls indices and push them into overlapping balls vector
    auto getAndPushOverlapping = [&overlapping_balls](VoroCrust_KD_Tree_Ball const& ball_tree, 
                                                      Vector3D const& center, 
                                                      double const b_radius, 
                                                      double const r_max_overlap, 
                                                      Dim const dim){
                                                        
        std::vector<std::size_t> const& overlapping_balls_indices = ball_tree.getOverlappingBalls(center, b_radius, r_max_overlap);
        for(std::size_t const ball_index : overlapping_balls_indices){
            overlapping_balls.push_back(BallInfo(ball_index, dim));
        }
    };

    getAndPushOverlapping(trees.ball_kd_vertices, p, radius, r_corner, Dim::CORNER);
    getAndPushOverlapping(trees.ball_kd_edges, p, radius, r_edge, Dim::EDGE);
    getAndPushOverlapping(trees.ball_kd_faces, p, radius, r_max, Dim::FACE);

    auto const& it = std::find(overlapping_balls.begin(), overlapping_balls.end(), ball_info);

    if(it == overlapping_balls.end()){
        std::cout << "current ball is not in overlapping balls!" << std::endl;
        exit(1);
    }

    // erase the current ball from the overlapping balls
    auto const index = it - overlapping_balls.begin();
    overlapping_balls.erase(overlapping_balls.begin() + index);

    return overlapping_balls;
}

std::vector<Triplet> SliverDriver::formTripletsOfOverlappingBalls(std::vector<BallInfo> const& overlapping_balls, Trees const& trees) const {
    std::vector<std::pair<std::size_t, std::size_t>> triplets;

    std::size_t const overlapping_balls_size = overlapping_balls.size();

    for(std::size_t i=0 ; i < overlapping_balls_size; ++i){
        auto const& [p_i, r_i] = getBall(overlapping_balls[i], trees);
        for(std::size_t j=i+1; j < overlapping_balls_size; ++j){
            auto const& [p_j, r_j] = getBall(overlapping_balls[j], trees);
            
            if(distance(p_i, p_j) < r_i + r_j) triplets.push_back(Triplet(i, j));
        }
    }

    return triplets;
}

Ball SliverDriver::getBall(BallInfo const& ball_info, Trees const& trees) const {
    std::size_t const index = ball_info.index;

    if(ball_info.dim == Dim::FACE){
        return Ball(trees.ball_kd_faces.points[index], trees.ball_kd_faces.ball_radii[index]);
    }

    
    if(ball_info.dim == Dim::EDGE){
        return Ball(trees.ball_kd_edges.points[index], trees.ball_kd_edges.ball_radii[index]);
    }

    
    if(ball_info.dim == Dim::CORNER){
        return Ball(trees.ball_kd_vertices.points[index], trees.ball_kd_vertices.ball_radii[index]);
    }

    std::cout << "ERROR: getBall" << std::endl;
    exit(1);
}

void SliverDriver::eliminateSliversForBallsInBallTree(Dim const dim, Trees const& trees) {
    VoroCrust_KD_Tree const * ball_tree_ptr = nullptr;

    switch(dim){
        case Dim::CORNER:
            ball_tree_ptr = &trees.ball_kd_vertices;
            break;
        case Dim::EDGE:
            ball_tree_ptr = &trees.ball_kd_edges;
            break;
        case Dim::FACE:
            ball_tree_ptr = &trees.ball_kd_faces;
            break;
        default:
            std::cout << "error in dim in eliminateSliversForBallsInBallTree" << std::endl;
            break;
    }
    
    std::size_t const num_of_balls = ball_tree_ptr->points.size();
    for(std::size_t i=0; i < num_of_balls; ++i){
        dealWithBall(BallInfo(i, dim), trees);
    }
}

void SliverDriver::dealWithBall(BallInfo const& ball_info, Trees const& trees) {
    std::vector<BallInfo> const& overlapping_balls = groupOverlappingBalls(ball_info, trees);
    std::vector<Triplet>  const& triplets = formTripletsOfOverlappingBalls(overlapping_balls, trees);

    for(Triplet const& triplet : triplets){
        dealWithTriplets(ball_info, triplet, overlapping_balls, trees);
    }
}

void SliverDriver::dealWithTriplets(BallInfo const& ball_info_1, Triplet const& triplet, std::vector<BallInfo> const& overlapping_balls, Trees const& trees) {
    BallInfo const& ball_info_2 = overlapping_balls[triplet.first];
    BallInfo const& ball_info_3 = overlapping_balls[triplet.second];

    Ball const& ball_1 = getBall(ball_info_1, trees);
    Ball const& ball_2 = getBall(ball_info_2, trees);
    Ball const& ball_3 = getBall(ball_info_3, trees);

    auto const& [success, seed_p, seed_m] = calculateIntersectionSeeds(ball_1, ball_2, ball_3);

    if(not success) return;

    for(BallInfo const& ball_info_4 : overlapping_balls){
        if(ball_info_4 == ball_info_2 || ball_info_4 == ball_info_3) continue;
        Ball const& ball_4 = getBall(ball_info_4, trees);
        auto const& [p4, r4] = ball_4;

        bool const is_seed_p_covered = distance(seed_p, p4) < r4; // check if seed_p is covered by ball_4
        bool const is_seed_m_covered = distance(seed_m, p4) < r4; // check if seed_m is covered by ball_4

        bool const half_covered_seed = (is_seed_p_covered && !is_seed_m_covered) || (is_seed_m_covered && !is_seed_p_covered);

        InfoQuartet info_quartet = {ball_info_1, ball_info_2, ball_info_3, ball_info_4};
        BallQuartet ball_quartet = {ball_1, ball_2, ball_3, ball_4};

        if(half_covered_seed) dealWithHalfCoveredSeeds(info_quartet, ball_quartet);
    }
}

void SliverDriver::dealWithHalfCoveredSeeds(InfoQuartet const& balls_info, BallQuartet const& balls){
    int least_shrinkage = -1;
    double r_new = -1.0;

    int l = -1;
    // run over all triplets in quartet
    for(int i=0; i<4; ++i){
        for(int j=i+1; j<4; ++j){
            for(int k=j+1; k<4; ++k){
                // find intersection seeds of triplets
                auto const& [success, seed_p, seed_m] = calculateIntersectionSeeds(balls[i], balls[j], balls[k]);

                // check if intersection of spheres is not disjoint
                if(not success) continue;

                // find the fourth index in the quartet
                for(int m=0; m < 4; ++m){
                    if((i != m) && (j != m) && (k != m)){
                        l=m;
                        break;
                    }
                }
                
                auto const& [p4, r4] = balls[l];
        
                bool const is_seed_p_covered = distance(seed_p, p4) < r4; // check if seed_p is covered by ball_4
                bool const is_seed_m_covered = distance(seed_m, p4) < r4; // check if seed_m is covered by ball_4

                double r_temp = r_new;

                // check if seeds are half covered
                if(is_seed_p_covered && !is_seed_m_covered){
                    r_temp = distance(seed_p, p4);
                } else if(is_seed_m_covered && !is_seed_p_covered) {
                    r_temp = distance(seed_m, p4);
                }

                if(r_temp > r_new){
                    least_shrinkage = l;
                    r_new = r_temp;
                }
            }
        }
    }

    // shrink ball needing the least amount to uncover the seeds 
    if(least_shrinkage>=0) setRadiusOfBall(r_new, balls_info[least_shrinkage]);
}

void SliverDriver::setRadiusOfBall(double const r_new, BallInfo const& ball_info) {
    if(ball_info.dim == Dim::CORNER){
        if(r_new < r_new_corner_balls[ball_info.index]) number_of_slivers_eliminated++;
        r_new_corner_balls[ball_info.index] = std::min(r_new, r_new_corner_balls[ball_info.index]);
        return;
    }

    if(ball_info.dim == Dim::EDGE) {
        if(r_new < r_new_edge_balls[ball_info.index]) number_of_slivers_eliminated++;
        r_new_edge_balls[ball_info.index] = std::min(r_new, r_new_edge_balls[ball_info.index]);
        return;
    }

    if(ball_info.dim == Dim::FACE) {
        if(r_new < r_new_face_balls[ball_info.index]) number_of_slivers_eliminated++;
        r_new_face_balls[ball_info.index] = std::min(r_new, r_new_face_balls[ball_info.index]);
        return;
    }

    std::cout << "ERROR: setRadiusOfBall" << std::endl;
    exit(1);
}

std::tuple<bool, Vector3D, Vector3D> SliverDriver::calculateIntersectionSeeds(Ball const& ball_1, Ball const& ball_2, Ball const& ball_3) const {
    // https://gist.github.com/ahcurrier/1a03faa8b0c2420ec6fd
    auto const& [p1, r1] = ball_1;
    auto const& [p2, r2] = ball_2;
    auto const& [p3, r3] = ball_3;

    auto const [a1, b1, c1, k1] = getLineCoeff(p1.x, p1.y, p1.z, r1, p2.x, p2.y, p2.z, r2);
    auto const [a3, b3, c3, k3] = getLineCoeff(p3.x, p3.y, p3.z, r3, p2.x, p2.y, p2.z, r2);

    auto const [e, f] = getZDependency(a1, b1, c1, k1, a3, b3, c3, k3);

    if(std::abs(a1) < 1e-10 && std::abs(a3) < 1e-10){
        double const z_both = (k3 - f*b3)/(b3*e + c3);
        double const y_both = z_both*e + f;

        double const val = r1*r1 - (y_both - p1.y)*(y_both - p1.y) - (z_both-p1.z)*(z_both-p1.z);

        if(val < 0){
            return std::tuple(false, Vector3D(0.0, 0.0, 0.0),  Vector3D(0.0, 0.0, 0.0));
        }

        double const val_sqrt = sqrt(val);
        double const x_plus = p1.x + val_sqrt;
        double const x_minus = p1.x - val_sqrt;

        Vector3D const seed_plus(x_plus, y_both, z_both);
        Vector3D const seed_minus(x_minus, y_both, z_both);

        return std::tuple(true, seed_plus, seed_minus);
    } else {
        
        auto const [g, h] = getZDependency(b1, a1, c1, k1, b3, a3, c3, k3);
        double const A = (1.0 + g*g + e*e);
        double const B = -2.0*(g*(p1.x-h) + e*(p1.y-f) + p1.z);
        double const C = ((p1.x-h)*(p1.x-h) + (p1.y-f) * (p1.y-f) + p1.z*p1.z - r1*r1);
        
        double const disc = B*B - 4*A*C;
        
        // if there is no intersection or the centers are linear or perpenicular (all of which should not preduce seeds)
        if(std::isnan(disc) || disc < 0){
            return std::tuple(false, Vector3D(0.0, 0.0, 0.0), Vector3D(0.0, 0.0, 0.0));
        }

        double const z_plus = (-B + sqrt(disc))/(2.0*A);
        double const x_plus = g*z_plus + h;
        double const y_plus = e*z_plus + f;

        Vector3D const seed_plus(x_plus, y_plus, z_plus);
        
        double const z_minus = (-B - sqrt(disc))/(2.0*A);
        double const x_minus = g*z_minus + h;
        double const y_minus = e*z_minus + f;

        Vector3D const seed_minus(x_minus, y_minus, z_minus);

        return std::tuple(true, seed_plus, seed_minus);
    }
}

std::tuple<double const, double const, double const, double const> 
getLineCoeff(double const x1, 
             double const y1, 
             double const z1, 
             double const r1, 
             double const x2, 
             double const y2, 
             double const z2, 
             double const r2) {
    
    double const a = 2.0*(x2-x1);
    double const b = 2.0*(y2-y1);
    double const c = 2.0*(z2-z1);
    double const k = r1*r1 - r2*r2 + x2*x2 - x1*x1 + y2*y2 - y1*y1 + z2*z2 - z1*z1;

    return std::tuple(a, b, c, k);
}

std::pair<double const, double const> 
getZDependency(double const a1, 
               double const b1, 
               double const c1, 
               double const k1, 
               double const a3, 
               double const b3, 
               double const c3, 
               double const k3) {
    
    //! WARNING: EPSILONTICA
    if(std::abs(a1) < 1e-10){
        return std::pair(-c1/b1, k1/b1);
    }
    
    if(std::abs(a3) < 1e-10){
        return std::pair(-c3/b3, k3/b3);
    }

    double const a31 = a3 / a1;
    double const denom = (b1*a31 - b3);

    if(std::abs(denom) < 1e-10){
        return std::pair(std::nan(""), std::nan(""));
    }

    double const e = (c3 - c1*a31) / denom;
    double const f = (k1*a31 - k3) / denom;

    return std::pair(e, f);
}

bool SliverDriver::eliminateSlivers(Trees &trees){
    // initialize radii of balls
    r_new_corner_balls = trees.ball_kd_vertices.ball_radii;
    r_new_edge_balls = trees.ball_kd_edges.ball_radii;
    r_new_face_balls = trees.ball_kd_faces.ball_radii;
    
    max_radius_corner = trees.ball_kd_vertices.empty() ? std::numeric_limits<double>::max()/1e100 : trees.ball_kd_vertices.getMaxRadius();
    max_radius_edge = trees.ball_kd_edges.empty() ? std::numeric_limits<double>::max()/1e100 : trees.ball_kd_edges.getMaxRadius();
    
    number_of_slivers_eliminated = 0;
    
    eliminateSliversForBallsInBallTree(Dim::FACE, trees);

    // update ball radii
    trees.ball_kd_vertices.ball_radii = r_new_corner_balls;
    trees.ball_kd_edges.ball_radii    = r_new_edge_balls;
    trees.ball_kd_faces.ball_radii    = r_new_face_balls;

    std::cout << "number of slivers eliminated in this iteration = " << number_of_slivers_eliminated << std::endl;

    return number_of_slivers_eliminated != 0;
}

std::vector<Seed> SliverDriver::getSeeds(Trees const& trees) const {
    //! CODEDUPLICATION:
    max_radius_corner = trees.ball_kd_vertices.empty() ? std::numeric_limits<double>::max()/1e100 : trees.ball_kd_vertices.getMaxRadius();
    max_radius_edge = trees.ball_kd_edges.empty() ? std::numeric_limits<double>::max()/1e100 : trees.ball_kd_edges.getMaxRadius();

    VoroCrust_KD_Tree_Ball seed_tree;
    Vector3D empty(0.0, 0.0, 0.0);

    VoroCrust_KD_Tree_Ball const& faces_ball_tree = trees.ball_kd_faces;
    for (std::size_t i = 0; i < faces_ball_tree.size(); ++i)
    {
        if(i % 100000 == 0) std::cout << "ball num: " << i << std::endl;
        BallInfo ball_info(i, Dim::FACE);
        std::vector<BallInfo> const& overlapping_balls = groupOverlappingBalls(ball_info, trees);
        std::vector<Triplet>  const& triplets = formTripletsOfOverlappingBalls(overlapping_balls, trees);
        
        for(Triplet const& triplet : triplets){
            BallInfo const& ball_info_2 = overlapping_balls[triplet.first];
            BallInfo const& ball_info_3 = overlapping_balls[triplet.second];

            Ball const& ball_1 = getBall(ball_info, trees);
            Ball const& ball_2 = getBall(ball_info_2, trees);
            Ball const& ball_3 = getBall(ball_info_3, trees);

            auto const& [success, seed_1, seed_2] = calculateIntersectionSeeds(ball_1, ball_2, ball_3);

            if(not success) continue;

            bool is_seed_1_covered = false;
            bool is_seed_2_covered = false;
            for(BallInfo const& ball_info_4 : overlapping_balls){
                if(ball_info_4 == ball_info_2 || ball_info_4 == ball_info_3) continue;

                auto const& [p4, r4] = getBall(ball_info_4, trees);

                is_seed_1_covered = is_seed_1_covered || distance(seed_1, p4) < r4; // check if seed_p is covered by ball_4
                is_seed_2_covered = is_seed_2_covered || distance(seed_2, p4) < r4; // check if seed_m is covered by ball_4

            }
            
            double const seed_radius = 0.3333 * (getR(ball_1) + getR(ball_2) + getR(ball_3));

            if(not is_seed_1_covered) seed_tree.insert(seed_1, empty, seed_radius, 0, 0);
            if(not is_seed_2_covered) seed_tree.insert(seed_2, empty, seed_radius, 0, 0);
        }
    }
    
    seed_tree.remakeTree();
    std::vector<Seed> const& seeds = unsorted_unique(seed_tree, 1e-5);
    
    return seeds;
}

std::vector<Seed> unsorted_unique(VoroCrust_KD_Tree_Ball const& seed_tree, double const tol) {
    std::size_t const seed_tree_size = seed_tree.size();

    std::vector<int> isDeleted(seed_tree_size, 0);

    std::size_t size_deleted = 0;
    for(std::size_t i=0; i<seed_tree_size; ++i){
        if(i % 100000 == 0) std::cout << "unsorted_unique seed num: " << i << std::endl;

        if(isDeleted[i]) continue;

        Vector3D const& seed_p = seed_tree.points[i];
        auto const& suspects = seed_tree.radiusSearch(seed_p, tol);
        for(auto const j : suspects){
            if(i==j) continue;

            isDeleted[j] = 1;
            ++size_deleted;
        }
    }



    std::vector<Seed> seed_vec;
    seed_vec.reserve(seed_tree_size - size_deleted + 1);
    std::cout << "size: " << seed_tree_size - size_deleted << std::endl;

    for(std::size_t i=0; i<seed_tree_size; ++i){
        if(isDeleted[i]) continue;

        seed_vec.emplace_back(seed_tree.getBall(i));
    }

    return seed_vec;
}
