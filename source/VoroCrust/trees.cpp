
#include "trees.hpp"
#include <boost/random.hpp>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

Trees::Trees(): VC_kd_sharp_corners(),
                VC_kd_sharp_edges(),
                VC_kd_faces(),
                ball_kd_vertices(),
                ball_kd_edges(),
                ball_kd_faces() {}

void Trees::loadPLC(PL_Complex const& plc, std::size_t const Nsample_edges, std::size_t const Nsample_faces){
    
    auto const& [sharp_corners_points, i_corners] = pointsFromVertices(plc.sharp_corners);
    auto const& [p_edges, v_edges, i_feature_edges, i_plc_edges] = superSampleEdges(plc.sharp_edges, Nsample_edges);
    auto const& [p_faces, v_faces, i_feature_faces, i_plc_faces] = superSampleFaces(plc.faces, Nsample_faces);
    
    VC_kd_sharp_corners = VoroCrust_KD_Tree_Boundary(sharp_corners_points);
    VC_kd_sharp_edges = VoroCrust_KD_Tree_Boundary(p_edges, v_edges, i_feature_edges, i_plc_edges);
    VC_kd_faces = VoroCrust_KD_Tree_Boundary(p_faces, v_faces, i_feature_faces, i_plc_faces);
    
    std::cout << "built edges tree with: " << Nsample_edges << " points" << std::endl;
    std::cout << "built faces tree with: " << Nsample_faces << " points" << std::endl;
}

std::tuple<std::vector<Vector3D>, std::vector<std::size_t>> 
Trees::pointsFromVertices(std::vector<VoroCrust::Vertex> const& vertices){
    std::size_t const Npoints = vertices.size();
    std::vector<Vector3D> points(Npoints, {0, 0, 0});
    std::vector<std::size_t> plc_index(Npoints, 0);

    for(std::size_t i = 0; i<Npoints; ++i){
        points[i] = vertices[i]->vertex;
        plc_index[i] = vertices[i]->index;
    }
    
    return std::tuple(points, plc_index);
}

std::tuple<std::vector<Vector3D>, std::vector<Vector3D>, std::vector<std::size_t>, std::vector<std::size_t>> 
Trees::superSampleEdges(std::vector<VoroCrust::Edge> const& edges, std::size_t const Nsample){
    // if there are no sharp edges
    if(edges.empty()){
        return std::tuple(std::vector<Vector3D>(), std::vector<Vector3D>(), std::vector<std::size_t>(), std::vector<std::size_t>());
    }

    // generate a random number generator
    boost::mt19937 rng(std::time(nullptr));
    boost::random::uniform_01<> zeroone;
    boost::variate_generator<boost::mt19937, boost::uniform_01<>> rand_gen(rng, zeroone);

    std::vector<double> start_len(edges.size(), 0.0);

    // calculate the total length of all edges and the start length of each individual edge
    // i.e. the total length up to it. 
    double total_len = 0;
    for(std::size_t i=0; i<edges.size(); ++i){
        VoroCrust::Edge const& edge = edges[i];
        start_len[i] = total_len;

        double const edge_len = abs(edge->vertex2->vertex - edge->vertex1->vertex);
        total_len += edge_len;
    }

    std::cout << "\nEdge Samples : \n---------------------\n";
    std::vector<Vector3D> points(Nsample, {0, 0, 0});
    std::vector<Vector3D> parallel(Nsample, {0, 0, 0});
    std::vector<std::size_t> feature_index(Nsample, 0);
    std::vector<std::size_t> plc_index(Nsample, 0);

    for(std::size_t i=0; i<Nsample; ++i){
        double const sample = rand_gen()*total_len; // sample a point uniformly [0, total_length)

        // find the index of the edge it lies
        auto const iter_lower_bound = std::lower_bound(start_len.begin(), start_len.end(), sample);
        std::size_t edge_index = std::distance(start_len.begin(), iter_lower_bound) - 1;

        // if sample is on a Vertex resample
        if(std::abs(start_len[edge_index] - sample) < 1e-14){
            //! TODO: make eps a user given parameter. 
            i--;
            continue;
        }

        // find point on edge
        VoroCrust::Edge const& edge = edges[edge_index];
        Vector3D const& edge_vec = (edge->vertex2->vertex - edge->vertex1->vertex);
        double const factor = (sample - start_len[edge_index]) / abs(edge_vec);
        Vector3D const& point = edge->vertex1->vertex + factor*edge_vec;

        points[i] = point;
        parallel[i] = (edge_vec / abs(edge_vec));
        feature_index[i] = edge->crease_index;
        plc_index[i] = edge->index;
    }

    return std::tuple(points, parallel, feature_index, plc_index);
}

std::tuple<std::vector<Vector3D>, std::vector<Vector3D>, std::vector<std::size_t>, std::vector<std::size_t>> 
Trees::superSampleFaces(std::vector<VoroCrust::Face> const& faces, std::size_t const Nsample){
    // generate a random number generator
    boost::mt19937 rng(std::time(nullptr));
    boost::random::uniform_01<> zeroone;
    boost::variate_generator<boost::mt19937, boost::uniform_01<>> rand_gen(rng, zeroone);

    std::vector<double> start_area(faces.size(), 0.0);

    // calculate the total area and the start area of each face
    // i.e. the total area up to the face
    double total_area = 0.0;
    for(std::size_t i=0; i<faces.size(); ++i){
        VoroCrust::Face const& face = faces[i];
        start_area[i] = total_area;

        double const face_area = face->calcArea();
        total_area += face_area;
    }

    std::cout << "\nFace Samples: \n---------------------\n";
    std::vector<Vector3D> points(Nsample, {0, 0, 0});
    std::vector<Vector3D> normals(Nsample, {0, 0, 0});
    std::vector<std::size_t> feature_index(Nsample, 0);
    std::vector<std::size_t> plc_index(Nsample, 0);

    for(std::size_t i = 0; i<Nsample; ++i){
        double const sample_area = rand_gen()*total_area; // sample a number uniformly in [0, total_area)

        // find the face it lies on
        auto const iter_lower_bound = std::lower_bound(start_area.begin(), start_area.end(), sample_area);
        auto face_index = std::distance(start_area.begin(), iter_lower_bound) - 1;

        VoroCrust::Face const& face = faces[face_index];
        
        if(face->vertices.size() != 3){
            std::cout << "ERROR: Algorithm supports only triangular meshes for now" << std::endl;
            exit(1);
        }

        // sample a point in a triangle
        double const sqrt_r1 = std::sqrt(rand_gen());
        double const r2 = rand_gen();

        if(sqrt_r1 < 1e-14 || r2 < 1e-14){
            //! TODO: make eps a user given parameter. 
            i--;
            continue;
        }

        Vector3D const& A = face->vertices[0]->vertex;
        Vector3D const& B = face->vertices[1]->vertex;
        Vector3D const& C = face->vertices[2]->vertex;

        // sample point formula is taken from https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle-in-3d
        Vector3D const& point = (1.0-sqrt_r1)*A + (sqrt_r1*(1.0-r2))*B + (r2*sqrt_r1)*C;

        points[i] = point;
        normals[i] = face->getNormal();
        feature_index[i] = face->patch_index;
        plc_index[i] = face->index;
    }

    return std::tuple(points, normals, feature_index, plc_index);
}

void Trees::dump(std::filesystem::path const& dirname) const {
    dump_boundary_tree(dirname / "boundary_tree_vertices", VC_kd_sharp_corners);
    dump_boundary_tree(dirname / "boundary_tree_edges", VC_kd_sharp_edges);
    dump_boundary_tree(dirname / "boundary_tree_faces", VC_kd_faces);

    dump_ball_tree(dirname / "ball_tree_vertices", ball_kd_vertices);
    dump_ball_tree(dirname / "ball_tree_edges", ball_kd_edges);
    dump_ball_tree(dirname / "ball_tree_faces", ball_kd_faces);
}

void dump_points(std::filesystem::path const& dirname, std::vector<Vector3D> const& points){
    std::filesystem::create_directory(dirname);
    
    std::ofstream p_x(dirname / "x.txt");
    std::ofstream p_y(dirname / "y.txt");
    std::ofstream p_z(dirname / "z.txt");

    p_x << std::setprecision(15);
    p_y << std::setprecision(15);
    p_z << std::setprecision(15);

    for(Vector3D const& p : points){
        p_x << p.x << "\n";
        p_y << p.y << "\n";
        p_z << p.z << "\n";
    }
}

void dump_vector(std::filesystem::path const& filename, std::vector<double> const& vec){
    std::ofstream output_file(filename);

    output_file << std::setprecision(15);

    for(auto const val : vec){
        output_file << val << "\n";
    }
}

void dump_vector_size_t(std::filesystem::path const& filename, std::vector<std::size_t> const& vec){
    std::ofstream output_file(filename);

    for(auto const val : vec){
        output_file << val << "\n";
    }
}

void dump_boundary_tree(std::filesystem::path const& dirname, VoroCrust_KD_Tree_Boundary const& boundary_tree){
    std::filesystem::create_directory(dirname);
    
    auto const dir_points = dirname / "points";
    dump_points(dir_points, boundary_tree.points);

    auto const dir_vectors = dirname / "vectors";
    dump_points(dir_vectors, boundary_tree.vectors);

    dump_vector_size_t(dirname / "feature_index.txt", boundary_tree.feature_index);
    dump_vector_size_t(dirname / "plc_index.txt", boundary_tree.plc_index);
}

void dump_ball_tree(std::filesystem::path const& dirname, VoroCrust_KD_Tree_Ball const& ball_tree){
    dump_boundary_tree(dirname, ball_tree);
    dump_vector(dirname / "ball_radii.txt", ball_tree.ball_radii);
}

void Trees::load_dump(std::filesystem::path const& dirname){
    auto const& dump_boundary_vertices = load_dump_boundary_tree(dirname / "boundary_tree_vertices");
    auto const& dump_boundary_edges = load_dump_boundary_tree(dirname / "boundary_tree_edges");
    auto const& dump_boundary_faces = load_dump_boundary_tree(dirname / "boundary_tree_faces");

    auto const& dump_ball_vertices = load_dump_ball_tree(dirname / "ball_tree_vertices");
    auto const& dump_ball_edges = load_dump_ball_tree(dirname / "ball_tree_edges");
    auto const& dump_ball_faces = load_dump_ball_tree(dirname / "ball_tree_faces");

    VC_kd_sharp_corners = VoroCrust_KD_Tree_Boundary(std::get<0>(dump_boundary_vertices));

    VC_kd_sharp_edges = VoroCrust_KD_Tree_Boundary(std::get<0>(dump_boundary_edges),
                                                   std::get<1>(dump_boundary_edges),
                                                   std::get<2>(dump_boundary_edges),
                                                   std::get<3>(dump_boundary_edges));
    
    VC_kd_faces = VoroCrust_KD_Tree_Boundary(std::get<0>(dump_boundary_faces),
                                             std::get<1>(dump_boundary_faces),
                                             std::get<2>(dump_boundary_faces),
                                             std::get<3>(dump_boundary_faces));
    
    ball_kd_vertices = VoroCrust_KD_Tree_Ball(std::get<0>(dump_ball_vertices),
                                              std::get<1>(dump_ball_vertices),
                                              std::get<2>(dump_ball_vertices),
                                              std::get<3>(dump_ball_vertices),
                                              std::get<4>(dump_ball_vertices));

    ball_kd_edges = VoroCrust_KD_Tree_Ball(std::get<0>(dump_ball_edges),
                                           std::get<1>(dump_ball_edges),
                                           std::get<2>(dump_ball_edges),
                                           std::get<3>(dump_ball_edges),
                                           std::get<4>(dump_ball_edges));
    
    ball_kd_faces = VoroCrust_KD_Tree_Ball(std::get<0>(dump_ball_faces),
                                           std::get<1>(dump_ball_faces),
                                           std::get<2>(dump_ball_faces),
                                           std::get<3>(dump_ball_faces),
                                           std::get<4>(dump_ball_faces));
}

std::vector<Vector3D> load_dump_points(std::filesystem::path const& dirname){
    if(not std::filesystem::is_directory(dirname)){
        std::cout << "ERROR: in load_dump_points no such directory: " << dirname.string() << std::endl;
        exit(1);
    }

    std::vector<double> x = load_dump_vector<double>(dirname / "x.txt");
    std::vector<double> y = load_dump_vector<double>(dirname / "y.txt");
    std::vector<double> z = load_dump_vector<double>(dirname / "z.txt");

    std::vector<Vector3D> points;

    std::size_t num_points = x.size();

    if(num_points != y.size() || num_points != z.size()){
        std::cout << "ERROR: in load_dump_points x, y, z have different sizes" << std::endl;
        exit(1);
    }

    points.reserve(x.size());
    
    for(std::size_t i = 0; i < num_points; ++i){
        points.emplace_back(x[i], y[i], z[i]);
    }

    return points;
}

std::tuple<points, vecs, feature_index, plc_index>
load_dump_boundary_tree(std::filesystem::path const& dirname){
    points t_points = load_dump_points(dirname / "points");
    vecs t_vecs = load_dump_points(dirname / "vectors");
    feature_index t_feature_index = load_dump_vector<std::size_t>(dirname / "feature_index.txt");
    plc_index t_plc_index = load_dump_vector<std::size_t>(dirname / "plc_index.txt");

    return std::tuple(t_points, t_vecs, t_feature_index, t_plc_index);
}

std::tuple<points, vecs, feature_index, plc_index, ball_radii>
load_dump_ball_tree(std::filesystem::path const& dirname){
    auto const& [t_points, t_vecs, t_feature_index, t_plc_index] = load_dump_boundary_tree(dirname);
    ball_radii t_ball_radii = load_dump_vector<double>(dirname / "ball_radii.txt");

    return std::tuple(t_points, t_vecs, t_feature_index, t_plc_index, t_ball_radii);
}
