#include "VoroCrustAlgorithm.hpp"
#include <cmath>
#include <iostream>
#include <boost/random.hpp>

VoroCrustAlgorithm::VoroCrustAlgorithm( PL_Complex const& plc_, 
                                        double const sharpTheta_, 
                                        double const flatTheta_, 
                                        double const maxRadius_,
                                        double const L_Lipschitz_,
                                        double const alpha_,
                                        std::size_t const maximal_num_iter_,
                                        std::size_t const num_of_samples_edges_,
                                        std::size_t const num_of_samples_faces_): plc(std::make_shared<PL_Complex>(plc_)), 
                                                              trees(),
                                                              sharpTheta(sharpTheta_),
                                                              flatTheta(flatTheta_),
                                                              maxRadius(maxRadius_),
                                                              L_Lipschitz(L_Lipschitz_),
                                                              alpha(alpha_),
                                                              maximal_num_iter(maximal_num_iter_),
                                                              num_of_samples_edges(num_of_samples_edges_),
                                                              num_of_samples_faces(num_of_samples_faces_),
                                                              cornersDriver(maxRadius_, L_Lipschitz_, sharpTheta_, plc),
                                                              edgesDriver(maxRadius_, L_Lipschitz_, alpha_, sharpTheta_, plc),
                                                              facesDriver(maxRadius_, L_Lipschitz_, alpha_, sharpTheta_, plc),
                                                              sliverDriver(L_Lipschitz_) {

    if(sharpTheta > M_PI_2){
        std::cout << "ERROR: sharpTheta > pi/2" << std::endl;
        exit(1);
    }           

    if(L_Lipschitz >= 1){
        std::cout << "ERROR: L_Lipschitz >= 1" << std::endl;
        exit(1);
    }

    if(L_Lipschitz <= 0){
        std::cout << "ERROR: L_Lipschitz <= 0 " << std::endl;
        exit(1);
    }

    //! TODO: Maybe all this need to be in the plc under detect features?
    if(not plc->checkAllVerticesAreUnique()) exit(1);

    if(not plc->checkAllVerticesAreOnFace()) exit(1);

    plc->detectFeatures(sharpTheta, flatTheta);
}

void VoroCrustAlgorithm::run() {
    //! TODO: make sampling size a user input!
    //! IMPORTANT: sampling size can effect the convergence of the algorithm because the radius is determined using proximity to the sampled points on different features. Make sure that the sampling size is compatible to the size of the smallest polygon in the data. One wants the sampling to be "dense" in the edges and faces.
    trees.loadPLC(*plc, num_of_samples_edges, num_of_samples_faces);    

    //! TODO: init eligable edges vertices and faces
    cornersDriver.loadCorners(plc->sharp_corners);
    cornersDriver.doSampling(trees.ball_kd_vertices, trees);
    trees.ball_kd_vertices.remakeTree();
    // sliver elimination loop
    for(std::size_t iteration = 0; iteration < maximal_num_iter; ++iteration){
        // enfore lipchitzness on vertices
        std::cout << "\nCorners Lipchitzness\n--------------\n" << std::endl;
        enforceLipschitzness(trees.ball_kd_vertices);    
        
        do {
            std::cout << "\nEdgesRMPS\n--------------\n" << std::endl;
            edgesDriver.loadEdges(plc->sharp_edges);
            edgesDriver.doSampling(trees.ball_kd_edges, trees);
            trees.ball_kd_edges.remakeTree();
        } while(enforceLipschitzness(trees.ball_kd_edges));
            
        do {
            std::cout << "\nFacesRMPS\n--------------\n" << std::endl;
            facesDriver.loadFaces(plc->faces);
            facesDriver.doSampling(trees.ball_kd_faces, trees);
            trees.ball_kd_faces.remakeTree();

        } while(enforceLipschitzness(trees.ball_kd_faces));

        // We can't eliminate slivers and enforce Lipchitzness since that would uncover parts of the PLC
        if(iteration+1 >= maximal_num_iter) break;

        if(not sliverDriver.eliminateSlivers(trees)) break;

        enforceLipschitzness(trees.ball_kd_edges);
        enforceLipschitzness(trees.ball_kd_faces);
    }
}

bool VoroCrustAlgorithm::enforceLipschitzness(VoroCrust_KD_Tree_Ball& ball_tree){
    std::size_t const num_of_points = ball_tree.size();
    
    std::size_t number_of_balls_shrunk = 0;

    std::cout << "Running enforceLipschitzness " << std::endl;
    // For each ball i enforce lipschitzness (going through all balls up to distance r_i/L_Lipschitz)
    for(std::size_t i = 0; i<num_of_points; ++i){
        auto const [p_i, r_i] = ball_tree.getBall(i);
        
        auto const& suspects = ball_tree.radiusSearch(p_i, r_i/L_Lipschitz);
        for(auto const j : suspects){

            auto const& [p_j, r_j] = ball_tree.getBall(j);
            double const dist = distance(p_i, p_j);
            
            if(r_i > r_j + L_Lipschitz*dist){
                number_of_balls_shrunk++;
                ball_tree.ball_radii[i] = r_j + L_Lipschitz*dist;
            }
        }
    }

    std::cout << "enforceLipschitzness : " << number_of_balls_shrunk << " balls shrunk" << std::endl;
    return number_of_balls_shrunk > 0;
}

std::vector<Seed> VoroCrustAlgorithm::getSeeds() const {
    return sliverDriver.getSeeds(trees);
}

VoroCrust_KD_Tree_Ball makeSeedBallTree(std::vector<Seed> const& seeds){
    
    std::size_t const seeds_size = seeds.size();

    std::vector<Vector3D> points;
    std::vector<double> radii;

    points.reserve(seeds_size+1);
    radii.reserve(seeds_size+1);

    for(auto const& seed : seeds){
        points.push_back(seed.p);
        radii.push_back(seed.radius);
    }

    return VoroCrust_KD_Tree_Ball(points, 
                                  std::vector<Vector3D>(seeds_size, Vector3D(0.0, 0.0, 0.0)), 
                                  std::vector<std::size_t>(seeds_size, 0), 
                                  std::vector<std::size_t>(seeds_size, 0),
                                  radii);
}

std::string VoroCrustAlgorithm::repr() const {
    std::ostringstream s;
    
    s << "VoroCrustAlgorithm : \n--------------------------------\n\n";
    s << "PLC : \n------------\n" << plc->repr() << std::endl;

    return s.str();
}

std::vector<std::vector<Seed>> determineZoneOfSeeds(std::vector<Seed> const& seeds, std::vector<PL_Complex> const& zone_plcs) {
    if(seeds.empty()){
        std::cout << "ERROR: seeds is empty" << std::endl;
        exit(1);
    }

    // the last index is for the outside seeds
    std::vector<std::vector<Seed>> zone_seeds(zone_plcs.size() + 1, std::vector<Seed>());

    for(auto& zone_seed_vec : zone_seeds){
        zone_seed_vec.reserve(seeds.size());
    }

    std::size_t seed_num = 0;
    for(auto const& seed : seeds){
        if(seed_num % 100000 == 0) std::cout << ++seed_num << std::endl;
        std::size_t i = 0;
        for(i=0; i < zone_plcs.size(); ++i){
            auto const& zone_plc = zone_plcs[i];

            if(zone_plc.determineLocation(seed.p) == PL_Complex::Location::IN) {
                zone_seeds[i].push_back(seed);
                break;
            }
        }

        if(i == zone_plcs.size()){
            zone_seeds[zone_plcs.size()].push_back(seed);
        }
    }
    return zone_seeds;
}

void VoroCrustAlgorithm::dump(std::filesystem::path const& dirname) const {
    //
    //
    //
    //

    std::filesystem::create_directories(dirname);

    trees.dump(dirname);
}

void VoroCrustAlgorithm::load_dump(std::filesystem::path const& dirname) {
    if(not std::filesystem::is_directory(dirname)){
        std::cout << "ERROR: dump directory does not exist" << std::endl;
        exit(1);
    }

    trees.load_dump(dirname);
}

bool VoroCrustAlgorithm::pointOutSidePLC(PL_Complex const& plc, Vector3D const& p){
    if(plc.determineLocation(p) != PL_Complex::Location::OUT)
        return false;
    bool in_boundary_ball = false;
    if(not trees.ball_kd_vertices.empty()) in_boundary_ball = in_boundary_ball || trees.ball_kd_vertices.isContainedInBall(p);
    if(not trees.ball_kd_edges.empty()) in_boundary_ball = in_boundary_ball || trees.ball_kd_edges.isContainedInBall(p);
    if(not trees.ball_kd_faces.empty()) in_boundary_ball = in_boundary_ball || trees.ball_kd_faces.isContainedInBall(p);

    if(in_boundary_ball)
        return false;
    return true;
}

bool VoroCrustAlgorithm::pointInSidePLC(PL_Complex const& plc, Vector3D const& p){
    if(plc.determineLocation(p) == PL_Complex::Location::OUT)
        return false;

    bool in_boundary_ball = false;
    if(not trees.ball_kd_vertices.empty()) in_boundary_ball = in_boundary_ball || trees.ball_kd_vertices.isContainedInBall(p);
    if(not trees.ball_kd_edges.empty()) in_boundary_ball = in_boundary_ball || trees.ball_kd_edges.isContainedInBall(p);
    if(not trees.ball_kd_faces.empty()) in_boundary_ball = in_boundary_ball || trees.ball_kd_faces.isContainedInBall(p);

    if(in_boundary_ball)
        return false;
    return true;
}

std::vector<std::vector<Seed>> VoroCrustAlgorithm::randomSampleSeeds(std::vector<PL_Complex> const& zones_plcs, std::vector<std::vector<Seed>> const& zones_boundary_seeds, double const maxSize) {
    std::size_t const num_of_zones = zones_plcs.size();

    Vector3D const empty_vec(0.0, 0.0, 0.0);
    // if(num_of_zones != zones_boundary_seeds.size()){
    //     std::cout << "ERROR: zones_plcs and zones_boundary_seeds size differ! " <<num_of_zones<<" "<< zones_boundary_seeds.size()<<std::endl;
    //     exit(1);
    // }

    return std::vector<std::vector<Seed>>();
    std::vector<std::vector<Seed>> zone_seeds;

    for(std::size_t zone_num=0; zone_num<num_of_zones; ++zone_num){
        PL_Complex const& plc = zones_plcs[zone_num];

        auto const& seeds = zones_boundary_seeds[zone_num];

        if(seeds.empty()){
            zone_seeds.push_back(seeds);

            continue;
        }

        auto const [ll_x, ll_y, ll_z, ur_x, ur_y, ur_z] = plc.getBoundingBox();
        auto const len_x = ur_x - ll_x;
        auto const len_y = ur_y - ll_y;
        auto const len_z = ur_z - ll_z;

        auto const& zone_seeds_tree = makeSeedBallTree(seeds);

        VoroCrust_KD_Tree_Ball volume_seeds_tree;

        // lightweight dart-throwing
        boost::random::variate_generator uni01_gen(boost::mt19937(std::time(nullptr)), boost::random::uniform_01<>());

        std::size_t num_of_samples = 0;
        do {
            std::size_t miss_counter = 0;
            while(miss_counter < 1000){
                Vector3D const p(ll_x + len_x*uni01_gen(),
                                 ll_y + len_y*uni01_gen(),
                                 ll_z + len_z*uni01_gen());
                
                if(plc.determineLocation(p) == PL_Complex::Location::OUT){
                    ++miss_counter;
                    continue;
                }

                bool in_boundary_ball = zone_seeds_tree.isContainedInBall(p) || trees.ball_kd_faces.isContainedInBall(p);

                if(not trees.ball_kd_vertices.empty()) in_boundary_ball = in_boundary_ball || trees.ball_kd_vertices.isContainedInBall(p);
                if(not trees.ball_kd_edges.empty()) in_boundary_ball = in_boundary_ball || trees.ball_kd_edges.isContainedInBall(p);

                if(in_boundary_ball){
                    ++miss_counter;
                    continue;
                }

                auto const& [p_s, r_s] = zone_seeds_tree.getBallNearestNeighbor(p);

                double r_volume = std::numeric_limits<double>::max();
                if(not volume_seeds_tree.empty()){
                    if(volume_seeds_tree.isContainedInBall(p)){
                        ++miss_counter;
                        continue;
                    }

                    auto const& [p_nearest, r_nearest] = volume_seeds_tree.getBallNearestNeighbor(p);

                    r_volume = r_nearest + L_Lipschitz*distance(p, p_nearest);
                }

                double const r = std::min({maxSize, r_s + L_Lipschitz*distance(p, p_s), r_volume});

                volume_seeds_tree.insert(p, empty_vec, r, 0, 0);
                num_of_samples++;
                
                if(num_of_samples % 10000 == 0){
                    std::cout << "sample: " << num_of_samples << ", miss_counter: " << miss_counter << std::endl;
                    volume_seeds_tree.remakeTree();
                }

                miss_counter = 0;
            }
            volume_seeds_tree.remakeTree();
        } while(enforceLipschitzness(volume_seeds_tree));

        std::vector<Seed> total_zone_seeds = zones_boundary_seeds[zone_num];
        std::vector<Seed> volume_temp_seeds = getSeedsFromBallTree(volume_seeds_tree);

        total_zone_seeds.insert(total_zone_seeds.end(), volume_temp_seeds.begin(), volume_temp_seeds.end());
        
        zone_seeds.push_back(total_zone_seeds);
    }

    return zone_seeds;
}

std::vector<Seed> getSeedsFromBallTree(VoroCrust_KD_Tree_Ball const& ball_tree){
    auto const tree_size = ball_tree.size();

    std::vector<Seed> seeds;
    seeds.reserve(tree_size + 1);

    for(std::size_t i=0; i<tree_size; ++i){
        seeds.emplace_back(ball_tree.getBall(i));
    }

    return seeds;
}

void dumpSeeds(std::filesystem::path const& dirname, std::vector<Seed> const& seeds){
    std::filesystem::create_directory(dirname);

    auto const num_of_seeds = seeds.size();

    std::vector<double> x, y, z, r;

    x.reserve(num_of_seeds+1);
    y.reserve(num_of_seeds+1);
    z.reserve(num_of_seeds+1);
    r.reserve(num_of_seeds+1);

    for(auto const& seed : seeds){
        auto const& p = seed.p;
        x.push_back(p.x);
        y.push_back(p.y);
        z.push_back(p.z);
        r.push_back(seed.radius);
    }

    dump_vector(dirname / "x.txt", x);
    dump_vector(dirname / "y.txt", y);
    dump_vector(dirname / "z.txt", z);
    dump_vector(dirname / "r.txt", r);
}

std::vector<Seed> load_dumpSeeds(std::filesystem::path const& dirname){
    if(not std::filesystem::is_directory(dirname)){
        std::cout << "ERROR: Seeds dump directory does not exist" << std::endl;
        exit(1); 
    }

    auto const x = load_dump_vector<double>(dirname / "x.txt");
    auto const y = load_dump_vector<double>(dirname / "y.txt");
    auto const z = load_dump_vector<double>(dirname / "z.txt");
    auto const r = load_dump_vector<double>(dirname / "r.txt");

    auto const num_of_seeds = x.size();

    if(y.size() != num_of_seeds || z.size() != num_of_seeds || r.size() != num_of_seeds){
        std::cout << "ERROR: x,y,z,r sizes do not match!" << std::endl;
        exit(1);
    }

    std::vector<Seed> seeds;
    seeds.reserve(num_of_seeds+1);

    for(std::size_t i=0; i<num_of_seeds; ++i){
        seeds.emplace_back(Vector3D(x[i], y[i], z[i]), r[i]);
    }

    return seeds;
}

