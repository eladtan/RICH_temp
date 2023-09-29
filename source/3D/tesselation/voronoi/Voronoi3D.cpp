#include "Voronoi3D.hpp"
#ifdef RICH_MPI
#include <mpi.h>
#endif
#include <vectorclass.h>
#include <algorithm>
#include <cfloat>
#include <stack>
#include "../../elementary/Mat33.hpp"
#include "../utils/Predicates3D.hpp"
#include "misc/utils.hpp"
#include "misc/io3D.hpp"
#include <fstream>
#include <iostream>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include "3D/GeometryCommon/Intersections.hpp"
#include "misc/int2str.hpp"
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/container/static_vector.hpp>
#include <omp.h>

#ifdef RICH_MPI

#include "3D/range/finders/BruteForce.hpp"
#include "3D/range/finders/RangeTree.hpp"
#include "3D/range/finders/OctTree.hpp"
#include "3D/range/finders/GroupRangeTree.hpp"
#include "3D/range/finders/HashBruteForce.hpp"
#include "3D/range/finders/SmartBruteForce.hpp"
#include "3D/environment/DistributedOctEnvAgent.hpp"
#include "3D/environment/HilbertEnvAgent.hpp"

#endif // RICH_MPI

// #define VORONOI_DEBUG

bool PointInPoly(Tessellation3D const &tess, Vector3D const &point, std::size_t index)
{
    face_vec const &faces = tess.GetCellFaces(index);
    vector<Vector3D> const &points = tess.GetFacePoints();
    std::size_t N = faces.size();
    std::array<Vector3D, 4> vec;
    for (std::size_t i = 0; i < N; ++i)
    {
        double R = fastsqrt(tess.GetArea(faces[i]));
        size_t N1 = 0;
        size_t N2 = 0;
        Vector3D V1, V2;
        size_t counter = 0;
        point_vec const &InFace = tess.GetPointsInFace(faces[i]);
        size_t NinFace = InFace.size();
        N1 = 1;
        V1 = points[InFace[(counter + 1) % NinFace]] - points[InFace[0]];
        while (fastabs(V1) < 0.01 * R)
        {
            ++counter;
            assert(counter < NinFace);
            V1 = points[InFace[(counter + 1) % NinFace]] - points[InFace[0]];
            ++N1;
        }
        V2 = points[InFace[(counter + 2) % NinFace]] - points[InFace[N1]];
        N2 = (counter + 2) % NinFace;
        while (fastabs(V2) < 0.01 * R || fastabs(CrossProduct(V1, V2)) < 0.0001 * tess.GetArea(faces[i]))
        {
            ++counter;
            if (counter > 2 * NinFace)
                break;
            V2 = points[InFace[(counter + 2) % NinFace]] - points[InFace[N1]];
            N2 = (counter + 2) % NinFace;
        }
        if (counter > 2 * NinFace)
        {
            std::cout << "Weird face in PointInPoly, cell " << index << " face " << faces[i] << " i " << i << " face area " << tess.GetArea(faces[i]) << std::endl;
            for (size_t j = 0; j < NinFace; ++j)
                std::cout << "Point j " << points[InFace[j]].x << "," << points[InFace[j]].y << "," << points[InFace[j]].z << std::endl;
            Vector3D normal = tess.GetFaceNeighbors(faces[i]).second == index ? tess.GetMeshPoint(tess.GetFaceNeighbors(faces[i]).second) - tess.GetMeshPoint(tess.GetFaceNeighbors(faces[i]).first) : tess.GetMeshPoint(tess.GetFaceNeighbors(faces[i]).first) - tess.GetMeshPoint(tess.GetFaceNeighbors(faces[i]).second);
            if (ScalarProd(normal, point - points[InFace[0]]) < 0)
                return false;
        }
        else
        {
            vec[0] = points[InFace[0]];
            vec[1] = points[InFace.at(N1)];
            vec[2] = points[InFace.at(N2)];
            vec[3] = tess.GetMeshPoint(index);
            double s1 = orient3d(vec);
            vec[3] = point;
            double s2 = orient3d(vec);
            if (s1 * s2 < -0)
                return false;
        }
    }
    return true;
}

bool PointInPoly(std::vector<Face> const& faces, Vector3D const &point)
{
    std::size_t const N = faces.size();
    std::array<Vector3D, 4> vec;
    vec[3] = point;
    for (std::size_t i = 0; i < N; ++i)
    {
        vec[0] = faces[i].vertices[0];
        vec[1] = faces[i].vertices[1];
        vec[2] = faces[i].vertices[2];
        double const s = orient3d(vec);
        if (s > 0)
            return false;
    }
    return true;
}

namespace
{
#ifdef RICH_MPI
    void GetPastDuplicate(size_t point, vector<size_t> &res, vector<vector<size_t>> const &sorted_to_duplicate,
                                                vector<size_t> const &procs)
    {
        res.clear();
        for (size_t i = 0; i < procs.size(); ++i)
        {
            if (std::binary_search(sorted_to_duplicate[i].begin(), sorted_to_duplicate[i].end(), point))
                res.push_back(procs[i]);
        }
    }
#endif
    boost::multiprecision::cpp_dec_float_50 Calc33Det(std::array<boost::multiprecision::cpp_dec_float_50, 9> const &points)
    {
        return points[0] * (points[4] * points[8] - points[5] * points[7]) + points[1] * (points[5] * points[6] - points[3] * points[8]) + points[2] * (points[3] * points[7] - points[4] * points[6]);
    }
}

namespace
{
    bool ShouldCalcTetraRadius(Tetrahedron const &T, size_t Norg)
    {
        for (size_t i = 0; i < 4; ++i)
            if (T.points[i] < Norg)
                return true;
        return false;
    }

    void FirstCheckList(std::stack<std::size_t> &check_stack, vector<unsigned char> &future_check, size_t Norg,
                                            Delaunay3D const &del, vector<tetra_vec> const &PointsInTetra)
    {
        //        check_stack.empty();
        future_check.resize(Norg, 0);
        size_t Ntetra = del.tetras_.size();
        vector<unsigned char> tetra_check(Ntetra, 0);

        for (size_t i = 0; i < Ntetra; ++i)
        {
            Tetrahedron const &tetra = del.tetras_[i];
            for (size_t j = 0; j < 4; ++j)
            {
                if (tetra.points[j] >= Norg)
                {
                    for (size_t k = 0; k < 4; ++k)
                    {
                        size_t tetcheck = tetra.points[k];
                        if (tetra.points[k] < Norg)
                        {
                            size_t ntet = PointsInTetra[tetcheck].size();
                            for (size_t z = 0; z < ntet; ++z)
                                tetra_check[PointsInTetra[tetcheck][z]] = 1;
                        }
                    }
                    break;
                }
            }
        }
        for (size_t i = 0; i < Ntetra; ++i)
        {
            if (tetra_check[i] == 1)
            {
                Tetrahedron const &tetra = del.tetras_[i];
                for (size_t j = 0; j < 4; ++j)
                {
                    if (tetra.points[j] < Norg)
                        future_check[tetra.points[j]] = 1;
                }
            }
        }
        for (size_t i = 0; i < Norg; ++i)
            if (future_check[i] == 1)
                check_stack.push(i);
    }

    vector<Face> BuildBox(Vector3D const &ll, Vector3D const &ur)
    {
        double dx = ur.x - ll.x;
        double dy = ur.y - ll.y;
        double dz = ur.z - ll.z;
        vector<Face> res(6);
        vector<Vector3D> points;
        points.push_back(ll);
        points.push_back(ll + Vector3D(dx, 0, 0));
        points.push_back(ll + Vector3D(dx, dy, 0));
        points.push_back(ll + Vector3D(0, dy, 0));
        points.push_back(ll + Vector3D(0, 0, dz));
        points.push_back(ll + Vector3D(dx, 0, dz));
        points.push_back(ll + Vector3D(dx, dy, dz));
        points.push_back(ll + Vector3D(0, dy, dz));
        points.push_back(ur);
        res[0].vertices.push_back(points[0]);
        res[0].vertices.push_back(points[1]);
        res[0].vertices.push_back(points[2]);
        res[0].vertices.push_back(points[3]);
        res[1].vertices.push_back(points[0]);
        res[1].vertices.push_back(points[4]);
        res[1].vertices.push_back(points[5]);
        res[1].vertices.push_back(points[1]);
        res[2].vertices.push_back(points[3]);
        res[2].vertices.push_back(points[7]);
        res[2].vertices.push_back(points[4]);
        res[2].vertices.push_back(points[0]);
        res[3].vertices.push_back(points[2]);
        res[3].vertices.push_back(points[6]);
        res[3].vertices.push_back(points[7]);
        res[3].vertices.push_back(points[3]);
        res[4].vertices.push_back(points[1]);
        res[4].vertices.push_back(points[5]);
        res[4].vertices.push_back(points[6]);
        res[4].vertices.push_back(points[2]);
        res[5].vertices.push_back(points[5]);
        res[5].vertices.push_back(points[4]);
        res[5].vertices.push_back(points[7]);
        res[5].vertices.push_back(points[6]);
        return res;
    }

#ifdef RICH_MPI
    vector<Vector3D> GetBoxNormals(Vector3D const &ll, Vector3D const &ur, vector<Face> const& box_faces_)
    {
        const vector<Face> faces = box_faces_.empty() ? BuildBox(ll, ur) : box_faces_;
        vector<Vector3D> res(faces.size());
        size_t N = res.size();
        for (size_t i = 0; i < N; ++i)
        {
            CrossProduct(faces[i].vertices[2] - faces[i].vertices[0], faces[i].vertices[1] - faces[i].vertices[0], res[i]);
            res[i] *= 1.0 /abs(res[i]);
        }
        return res;
    }

    size_t BoxIndex(vector<Vector3D> const &fnormals, const Vector3D &normal)
    {
        double max_angle = ScalarProd(fnormals[0], normal);
        size_t loc = 0;
        size_t N = fnormals.size();
        for (size_t i = 1; i < N; i++)
        {
            double temp = ScalarProd(fnormals[i], normal);
            if (temp > max_angle)
            {
                max_angle = temp;
                loc = i;
            }
        }
        return loc;
    }
#endif

    double CleanDuplicates(std::array<size_t, 128> const &indeces, const vector<Vector3D> &points,
                                                 boost::container::small_vector<size_t, 8> &res, double R,
                                                 std::array<double, 128> &diffs,
                                                 std::array<Vector3D, 128> &vtemp, const size_t N)
    {
        res.clear();
        for (size_t i = 0; i < N; ++i)
            vtemp[i] = points[indeces[i]];
        for (size_t i = N - 1; i > 0; --i)
        {
            vtemp[i].x -= vtemp[i - 1].x;
            vtemp[i].y -= vtemp[i - 1].y;
            vtemp[i].z -= vtemp[i - 1].z;
        }
        vtemp[0] -= points[indeces[N - 1]];
#ifdef __INTEL_COMPILER
#pragma omp simd reduction(max \
                                                     : R)
#endif
        for (size_t i = 0; i < N; ++i)
        {
            diffs[i] = ScalarProd(vtemp[i], vtemp[i]);
            R = std::max(R, diffs[i]);
        }
        for (size_t i = 0; i < N; ++i)
            if (diffs[i] > R * 1e-16)
                res.push_back(indeces[i]);
        return R;
    }

    size_t SetPointTetras(vector<tetra_vec> &PointTetras, size_t Norg, vector<Tetrahedron> &tetras,
                                                boost::container::flat_set<size_t> const &empty_tetras)
    {
        PointTetras.clear();
        PointTetras.resize(Norg);

        Vec4uq _Norg(Norg, Norg, Norg, Norg);

        size_t Ntetra = tetras.size();
        size_t bigtet(0);
        bool has_good, has_big;
        // change empty tetras to be not relevant
        for (boost::container::flat_set<size_t>::const_iterator it = empty_tetras.begin(); it != empty_tetras.end(); ++it)
        {
#ifdef __INTEL_COMPILER
#pragma omp simd early_exit
#endif
            for (size_t i = 0; i < 4; ++i)
            {
                tetras[*it].points[i] = std::numeric_limits<std::size_t>::max();
                tetras[*it].neighbors[i] = std::numeric_limits<std::size_t>::max();
            }
        }

        for (size_t i = 0; i < Ntetra; ++i)
        {
            has_good = false;
            has_big = false;

            const Tetrahedron &tet = tetras[i];
            Vec4uq _points(tet.points[0], tet.points[1], tet.points[2], tet.points[3]);
            Vec4qb cmp = (_points < _Norg);

            for(int j = 0; j < 4; ++j)
            {
                if(cmp[j])
                {
                    has_good = true;
                    PointTetras[_points[j]].push_back(i);
                }
                else
                {
                    has_big = true;
                }
            }
            if(has_big and has_good)
            {
                bigtet = i;
            }
            /*
            for (size_t j = 0; j < 4; ++j)
            {
                size_t temp = tetras[i].points[j];
                if (temp < Norg)
                {
                    PointTetras[temp].push_back(i);
                    has_good = true;
                }
                else
                    has_big = true;
            }
            if (has_big && has_good)
                bigtet = i;
            */
        }
        return bigtet;
    }

    bool CleanSameLine(boost::container::small_vector<size_t, 8> &indeces, vector<Vector3D> const& face_points, std::array<double, 128> &area_vec_temp)
    {
        point_vec old;
        size_t const N = indeces.size();
        double const small_fraction = 1e-14;
        double const medium_fraction = 3e-1;
        // Find correct normal
        Vector3D good_normal;
        for(size_t i = 0; i < N; ++i)
        {
            area_vec_temp[i] = fastabs(CrossProduct(face_points[indeces[i]] - face_points[indeces[(N + i - 1) % N]], face_points[indeces[(i + 1) % N]] 
            - face_points[indeces[(N + i - 1) % N]]));
            old.push_back(indeces[i]);
        }

        // ADDED

        double max_value = area_vec_temp[0];
        double second_max_value = max_value;
        size_t max_index = 0, second_max_index = 0;
        for(size_t i = 1; i < N; ++i)
        {
            if(area_vec_temp[i] > max_value)
            {
                second_max_value = max_value;
                max_value = area_vec_temp[i];
                second_max_index = max_index;
                max_index = i;
            }
            else
            {
                if(area_vec_temp[i] > second_max_value)
                {
                    second_max_value = area_vec_temp[i];
                    second_max_index = i;
                }
            }
        }

        // FINISHED ADDED
        /*
        double const area_scale = *std::max_element(area_vec_temp.begin(), area_vec_temp.begin() + N);
        good_normal = CrossProduct(face_points[indeces[0]] - face_points[indeces[N - 1]], face_points[indeces[1]] 
        - face_points[indeces[N - 1]]);
        good_normal *= 1.0 / (100 * std::numeric_limits<double>::min() + fastabs(good_normal));
        */

        double const area_scale = area_vec_temp[max_index];
        good_normal = CrossProduct(face_points[indeces[max_index]] - face_points[indeces[(N + max_index - 1) % N]], face_points[indeces[(max_index + 1) % N]] - face_points[indeces[(N + max_index - 1) % N]]);
        good_normal *= 1.0 / fastabs(good_normal);

        /*
        for(size_t i = 1; i < N; ++i)
        {
            if(area_vec_temp[i] > area_scale * medium_fraction)
            {
                good_normal = CrossProduct(face_points[indeces[i]] - face_points[indeces[i - 1]], face_points[indeces[(i + 1) % N]] 
                - face_points[indeces[i - 1]]);
                good_normal *= 1.0 / fastabs(good_normal);
                break;
            }
        }
        */
        size_t Nindeces = indeces.size();
        for(size_t i = 0; i < Nindeces; ++i)
        {
            Vector3D normal_temp = CrossProduct(face_points[indeces[i]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]], face_points[indeces[(i + 1) % Nindeces]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]]);
            double const area = fastabs(normal_temp);
            normal_temp *= 1.0 / (100 * std::numeric_limits<double>::min() + area);
            if(area < area_scale * small_fraction || ScalarProd(normal_temp, good_normal) < /*0.99998*/ 0.9999)
            {
                indeces.erase(indeces.begin() + i);
                if(i == indeces.size() - 1)
                    break;
                --i;
                Nindeces = indeces.size();
            }
        }

        if(Nindeces < 3)
        {
            indeces = old;
            max_index = second_max_index;
            good_normal = CrossProduct(face_points[indeces[max_index]] - face_points[indeces[(N + max_index - 1) % N]], face_points[indeces[(max_index + 1) % N]] - face_points[indeces[(N + max_index - 1) % N]]);
            good_normal *= 1.0 / fastabs(good_normal);

            for(size_t i = 0; i < Nindeces; i++)
            {
                Vector3D normal_temp = CrossProduct(face_points[indeces[i]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]], face_points[indeces[(i + 1) % Nindeces]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]]);
                double const area = fastabs(normal_temp);
                normal_temp *= 1.0 / area;
                if(area < area_scale * small_fraction || ScalarProd(normal_temp, good_normal) < /*0.99998*/ 0.9999)
                {
                    indeces.erase(indeces.begin() + i);
                    if(i == indeces.size() - 1)
                        break;
                    --i;
                    Nindeces = indeces.size();
                }
            }
        }

        if(Nindeces < 3)
        {
            // TODO HERE
            throw UniversalError("Bad CleanSameLine");
        }
        /*
        return false;
        else
        */
        return true;
    }


    void MakeRightHandFace(boost::container::small_vector<size_t, 8> &indeces, Vector3D const &point, vector<Vector3D> const &face_points,
                                                 std::array<size_t, 128> &temp, double areascale)
    {
        Vector3D V1, V2;
        size_t counter = 0;
        const size_t N = indeces.size();
        V1 = face_points[indeces[counter + 1]];
        V1 -= face_points[indeces[counter]];
        double AScale = 1e-14 * areascale;
        while (ScalarProd(V1, V1) < AScale)
        {
            ++counter;
            assert(counter < N);
            V1 = face_points[indeces[(counter + 1) % N]];
            V1 -= face_points[indeces[counter]];
        }
        V2 = face_points[indeces[(counter + 2) % N]];
        V2 -= face_points[indeces[(counter + 1) % N]];
        while (ScalarProd(V2, V2) < AScale)
        {
            ++counter;
            assert(counter < 2 * N);
            V2 = face_points[indeces[(counter + 2) % N]];
            V2 -= face_points[indeces[(counter + 1) % N]];
        }
        // Do we need to flip handness?
        if (ScalarProd(CrossProduct(V1, V2), point - face_points[indeces[0]]) > 0)
        {
            const size_t Ninner = indeces.size();
#ifdef __INTEL_COMPILER
#pragma omp simd early_exit
#endif
            for (size_t j = 0; j < Ninner; ++j)
                temp[j] = indeces[j];
#ifdef __INTEL_COMPILER
#pragma omp simd early_exit
#endif
            for (size_t i = 0; i < N; ++i)
                indeces[i] = temp[(N - i - 1)];
        }
    }

    size_t NextLoopTetra(Tetrahedron const &cur_tetra, size_t last_tetra, size_t N0, size_t N1)
    {
        size_t i = 0;
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
        for (; i < 4; i++)
        {
            size_t point = cur_tetra.points[i];
            if (point != N0 && point != N1 && cur_tetra.neighbors[i] != last_tetra)
                break;
        }
        if(i >= 4)
            throw UniversalError("Bad NextLoopTetra");
        return cur_tetra.neighbors[i];
    }

    void CalcFaceAreaCM(boost::container::small_vector<size_t, 8> const &indeces, std::vector<Vector3D> const &allpoints,
                                            std::array<Vector3D, 128> &points, double &Area, Vector3D &CM,
                                            std::array<double, 128> &Atemp)
    {
        //CM.Set(0.0, 0.0, 0.0);
        size_t Nloop = indeces.size();
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
        for (size_t i = 0; i < Nloop; i++)
            points[i] = allpoints[indeces[i]];
        Nloop -= 2;
        Area = 0;
        //Vector3D temp3, temp4, temp5;
        for (size_t i = 0; i < Nloop; i++)
        {
            //temp4.Set(points[i + 1].x - points[0].x, points[i + 1].y - points[0].y, points[i + 1].z - points[0].z);
            Vector3D temp4(points[i + 1].x - points[0].x, points[i + 1].y - points[0].y, points[i + 1].z - points[0].z);
            //temp5.Set(points[i + 2].x - points[0].x, points[i + 2].y - points[0].y, points[i + 2].z - points[0].z);
            Vector3D temp5(points[i + 2].x - points[0].x, points[i + 2].y - points[0].y, points[i + 2].z - points[0].z);
            Vector3D temp3;
            CrossProduct(temp4, temp5, temp3);
            Atemp[i] = 0.3333333333333333 * 0.5 * fastsqrt(ScalarProd(temp3, temp3));
        }
        double x = 0, y = 0, z = 0;
#ifdef __INTEL_COMPILER
#pragma vector aligned
        //#pragma omp simd reduction(+:x, y, z, Area)
#endif
        for (size_t i = 0; i < Nloop; i++)
        {
            double A = Atemp[i];
            x += A * points[0].x;
            y += A * points[0].y;
            z += A * points[0].z;
            x += A * points[i + 1].x;
            y += A * points[i + 1].y;
            z += A * points[i + 1].z;
            x += A * points[i + 2].x;
            y += A * points[i + 2].y;
            z += A * points[i + 2].z;
            Area += 3.0 * A;
        }
        CM.Set(x, y, z);
        CM *= (1.0 / (Area + DBL_MIN * 100)); //prevent overflow
    }

    bool PointInDomain(Vector3D const &ll, Vector3D const &ur, Vector3D const &point)
    {
        if (point.x > ll.x && point.x < ur.x && point.y > ll.y && point.y < ur.y && point.z > ll.z && point.z < ur.z)
            return true;
        else
            return false;
    }

    Vector3D MirrorPoint(Face const &face, Vector3D const &point)
    {
        Vector3D normal = CrossProduct(face.vertices[1] - face.vertices[0], face.vertices[2] - face.vertices[0]);
        normal = normal / abs(normal);
        return point - (2 * ScalarProd(point - face.vertices[0], normal)) * normal;
    }
}

Voronoi3D::Voronoi3D() : ll_(Vector3D()), ur_(Vector3D()), Norg_(0), bigtet_(0), set_temp_(std::set<int>()), stack_temp_(std::stack<int>()),
                        del_(Delaunay3D()), PointTetras_(vector<tetra_vec>()), R_(vector<double>()), tetra_centers_(vector<Vector3D>()),
                        FacesInCell_(vector<face_vec>()),
                        PointsInFace_(vector<point_vec>()),
                        FaceNeighbors_(vector<std::pair<std::size_t, std::size_t>>()),
                        CM_(vector<Vector3D>()), Face_CM_(vector<Vector3D>()),
                        volume_(vector<double>()), area_(vector<double>()), duplicated_points_(vector<vector<std::size_t>>()),
                        sentprocs_(vector<int>()), duplicatedprocs_(vector<int>()), sentpoints_(vector<vector<std::size_t>>()), Nghost_(vector<vector<std::size_t>>()),
                        self_index_(vector<std::size_t>()), temp_points_(std::array<Vector3D, 4>()), temp_points2_(std::array<Vector3D, 5>())
                        #ifdef RICH_MPI
                        , envAgent(nullptr), initialRadius(0.0), firstCall(true), pointsManager(PointsManager(this->ll_, this->ur_)), hilbertOrder(NULL_ORDER)
                        #endif // RICH_MPI
{
}

Voronoi3D::Voronoi3D(std::vector<Face> const& box_faces) : Norg_(0), bigtet_(0), set_temp_(std::set<int>()), stack_temp_(std::stack<int>()),
                                                        del_(Delaunay3D()), PointTetras_(vector<tetra_vec>()), R_(vector<double>()), tetra_centers_(vector<Vector3D>()),
                                                        FacesInCell_(vector<face_vec>()),
                                                        PointsInFace_(vector<point_vec>()),
                                                        FaceNeighbors_(vector<std::pair<std::size_t, std::size_t>>()),
                                                        CM_(vector<Vector3D>()), Face_CM_(vector<Vector3D>()),
                                                        volume_(vector<double>()), area_(vector<double>()), duplicated_points_(vector<vector<std::size_t>>()),
                                                        sentprocs_(vector<int>()), duplicatedprocs_(vector<int>()), sentpoints_(vector<vector<std::size_t>>()), Nghost_(vector<vector<std::size_t>>()),
                                                        self_index_(vector<std::size_t>()), temp_points_(std::array<Vector3D, 4>()), temp_points2_(std::array<Vector3D, 5>()), box_faces_(box_faces)
                                                        #ifdef RICH_MPI
                                                        , envAgent(nullptr), initialRadius(0.0), firstCall(true), pointsManager(PointsManager(this->ll_, this->ur_)), hilbertOrder(NULL_ORDER)
                                                        #endif // RICH_MPI
{
    size_t const Nfaces = box_faces.size();
    if(Nfaces < 4)
        throw UniversalError("Zero face vector in Voronoi3D constructor");
    ll_ = box_faces[0].vertices[0];
    ur_ = ll_;
    for(size_t i = 0; i < Nfaces; ++i)
    {
        size_t const Nvertices = box_faces[i].vertices.size();
        for(size_t j = 0; j < Nvertices; ++j)
        {
            ll_.x = std::min(ll_.x, box_faces[i].vertices[j].x);
            ll_.y = std::min(ll_.y, box_faces[i].vertices[j].y);
            ll_.z = std::min(ll_.z, box_faces[i].vertices[j].z);
            ur_.x = std::max(ur_.x, box_faces[i].vertices[j].x);
            ur_.y = std::max(ur_.y, box_faces[i].vertices[j].y);
            ur_.z = std::max(ur_.z, box_faces[i].vertices[j].z);
        }
    }
}

Voronoi3D::Voronoi3D(Vector3D const &ll, Vector3D const &ur) : ll_(ll), ur_(ur), Norg_(0), bigtet_(0), set_temp_(std::set<int>()), stack_temp_(std::stack<int>()),
                                                              del_(Delaunay3D()), PointTetras_(vector<tetra_vec>()), R_(vector<double>()), tetra_centers_(vector<Vector3D>()),
                                                              FacesInCell_(vector<face_vec>()),
                                                              PointsInFace_(vector<point_vec>()),
                                                              FaceNeighbors_(vector<std::pair<std::size_t, std::size_t>>()),
                                                              CM_(vector<Vector3D>()), Face_CM_(vector<Vector3D>()),
                                                              volume_(vector<double>()), area_(vector<double>()), duplicated_points_(vector<vector<std::size_t>>()),
                                                              sentprocs_(vector<int>()), duplicatedprocs_(vector<int>()), sentpoints_(vector<vector<std::size_t>>()), Nghost_(vector<vector<std::size_t>>()),
                                                              self_index_(vector<std::size_t>()), temp_points_(std::array<Vector3D, 4>()), temp_points2_(std::array<Vector3D, 5>()), box_faces_(std::vector<Face> ())
                                                              #ifdef RICH_MPI
                                                              , envAgent(nullptr), initialRadius(0.0), firstCall(true), pointsManager(PointsManager(this->ll_, this->ur_)), hilbertOrder(NULL_ORDER)
                                                              #endif // RICH_MPI
                                                              {}

void Voronoi3D::CalcRigidCM(std::size_t face_index)
{
    Vector3D normal = normalize(del_.points_[FaceNeighbors_[face_index].first] - del_.points_[FaceNeighbors_[face_index].second]);
    std::size_t real, other;
    if (FaceNeighbors_[face_index].first >= Norg_)
    {
        real = FaceNeighbors_[face_index].second;
        other = FaceNeighbors_[face_index].first;
    }
    else
    {
        real = FaceNeighbors_[face_index].first;
        other = FaceNeighbors_[face_index].second;
    }
    CM_[other] = CM_[real] - 2 * normal * ScalarProd(normal, CM_[real] - tetra_centers_[PointsInFace_[face_index][0]]);
}

vector<Vector3D> Voronoi3D::CreateBoundaryPoints(vector<std::pair<std::size_t, std::size_t>> const &to_duplicate,
                                                 vector<vector<size_t>> &past_duplicate)
{
    size_t Ncheck = to_duplicate.size();
    vector<std::pair<std::size_t, std::size_t>> to_add;
    to_add.reserve(Ncheck);
    vector<Face> faces = box_faces_.empty() ? BuildBox(ll_, ur_) : box_faces_;
    vector<Vector3D> res;
    bool first_time = past_duplicate.empty();
    if (first_time)
        past_duplicate.resize(faces.size());
    for (std::size_t i = 0; i < Ncheck; ++i)
    {
        if (first_time || !std::binary_search(past_duplicate[to_duplicate[i].first].begin(),
                                                                                    past_duplicate[to_duplicate[i].first].end(), to_duplicate[i].second))
        {
            res.push_back(MirrorPoint(faces[to_duplicate[i].first], del_.points_[to_duplicate[i].second]));
            to_add.push_back(to_duplicate[i]);
        }
    }
    for (size_t i = 0; i < to_add.size(); ++i)
        past_duplicate[to_add[i].first].push_back(to_add[i].second);
    for (size_t i = 0; i < past_duplicate.size(); ++i)
        std::sort(past_duplicate[i].begin(), past_duplicate[i].end());
    return res;
}

vector<vector<std::size_t>> const &Voronoi3D::GetGhostIndeces(void) const
{
    return Nghost_;
}

/**
 * gets a point index, and returns the maximal radius of the tetrahedra containing that point.
 * @param index the index of the point (within the points list)
*/
double Voronoi3D::GetMaxRadius(std::size_t index)
{
    std::size_t N = PointTetras_[index].size();
    double res = 0;
    #ifdef __INTEL_COMPILER
    #pragma ivdep
    #endif
    for(std::size_t i = 0; i < N; ++i)
    {
        res = std::max(res, GetRadius(PointTetras_[index][i]));
    }
    return res;
}

/**
 * gets a point index, and returns the minimal radius of the tetrahedra containing that point.
 * @param index the index of the point (within the points list)
*/
double Voronoi3D::GetMinRadius(std::size_t index)
{
    std::size_t N = PointTetras_[index].size();
    double res = std::numeric_limits<double>::max();
    #ifdef __INTEL_COMPILER
    #pragma ivdep
    #endif
    for(std::size_t i = 0; i < N; ++i)
    {
        res = std::min(res, GetRadius(PointTetras_[index][i]));
    }
    return res;
}

/**
 * if the initial box does not exist, builds its faces according to the leftmost and rightmost points.
 * If it does, does not build the faces again.
 * @return the normals to the faces
*/
void Voronoi3D::InitialBoxBuild(std::vector<Face> &box, std::vector<Vector3D> &normals)
{
    box = box_faces_.empty() ? BuildBox(this->ll_, this->ur_) : this->box_faces_;
    size_t Nfaces = box.size();
    normals.resize(Nfaces);

    // calculates the normals for each one of the box's faces
    for (size_t i = 0; i < Nfaces; ++i)
    {
        normals[i] = CrossProduct(box[i].vertices[1] - box[i].vertices[0], box[i].vertices[2] - box[i].vertices[0]);
        normals[i] *= (1.0 / fastsqrt(ScalarProd(normals[i], normals[i])));
    }
}

#ifdef RICH_MPI
namespace
{
    #ifdef VORONOI_DEBUG
    template<typename T>
    void reportDuplications(const std::vector<T> &vector)
    {
        for(size_t i = 0; i < vector.size(); i++)
        {
            for(size_t j = 0; j < vector.size(); j++)
            {
                if(i == j) continue;
                if(vector[i] == vector[j])
                {
                    std::cout << "duplication found in indices " << i << " and " << j << ": " << vector[i] << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 2050);
                }
            }
        }
    }
    #endif // VORONOI_DEBUG
}

/**
 * \author Maor Mizrachi
 * \brief Initializes data structures, for the voronoi build
*/
void Voronoi3D::BuildInitialize(size_t num_points)
{
    // assert(num_points > 0);
    // Clear data
    PointTetras_.clear();
    R_.clear();
    if(num_points > 0) R_.reserve(num_points * 11);
    tetra_centers_.clear();
     if(num_points > 0) tetra_centers_.reserve(num_points * 11);
    // Voronoi Data
    del_.Clean();
    FacesInCell_.clear();
    PointsInFace_.clear();
    FaceNeighbors_.clear();
    CM_.clear();
    Face_CM_.clear();
    volume_.clear();
    area_.clear();
    Norg_ = num_points;
    duplicatedprocs_.clear();
    duplicated_points_.clear();
    Nghost_.clear();
}

/**
 * \author Maor Mizrachi
 * \brief Checks if a certain point is under my responsibility
*/
bool Voronoi3D::PointInMyDomain(const Vector3D &point) const
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(this->envAgent != nullptr);
    return (this->envAgent->getOwner(point) == rank);
}


/**
 * \author Maor Mizrachi
 * \brief Gets a point, its radius, a box and the normals to the box's faces, and returns the faces indices that the sphere (around `point`, in the given `radius`) intersects
*/
std::vector<size_t> Voronoi3D::CheckToMirror(const Vector3D &point, double radius, const std::vector<Face> &box, const std::vector<Vector3D> &normals)
{
    std::vector<size_t> facesItCuts;
    // std::cout << "point = " << point << ", radius = " << radius << std::endl;
    for(size_t i = 0; i < box.size(); i++)
    {
        // check for intersecting the sphere with radius `radius` around `point`, with the `i`th face of `box`
        Sphere sphere(point, radius);
        if(FaceSphereIntersections(box[i], sphere, normals[i]))
        {
            // intersects! mirror the point
            facesItCuts.push_back(i);
        }
    }
    return facesItCuts;
}

/**
 * \author Maor Mizrachi
 * \brief Creates a batch for a cycle (iteration) in the ghost points bringing loop
*/
std::queue<RangeQueryData> Voronoi3D::CreateBatches(boost::container::flat_set<size_t> &smallPoints, boost::container::flat_set<size_t> &largePoints, std::vector<double> &currentRadiuses, int iterations)
{
    std::queue<RangeQueryData> queries;
    boost::container::flat_set<size_t> newSmallPoints, newLargePoints;
    boost::container::flat_set<size_t> tetraToCancel;

    if(iterations == 1)
    {
        // at first iteration, run an initial query
        for(const size_t &pointIdx : smallPoints)
        {
            newSmallPoints.insert(pointIdx);
            const Vector3D &point = this->del_.points_[pointIdx];
            RangeQueryData query = {pointIdx, {point.x, point.y, point.z}, {point.x, point.y, point.z}, currentRadiuses[pointIdx], NO_MAX_POINTS, ASK_ALL};
            queries.push(query);
        }
    }
    else
    {
        for(const size_t &pointIdx : smallPoints)
        {
            // let us first check if the point is really small, or should be considered as large
            double biggestRadius = std::numeric_limits<double>::min(), smallestRadius = std::numeric_limits<double>::max();
            for(const size_t &tetraIdx : this->PointTetras_[pointIdx])
            {
                double tetraRadius = this->GetRadius(tetraIdx);
                biggestRadius = std::max<double>(biggestRadius, tetraRadius);
                smallestRadius = std::min<double>(smallestRadius, tetraRadius);
            }

            double relation = biggestRadius / smallestRadius;
            if(relation > 3) // todo: magic number
            {
                // point is now considered large!
                newLargePoints.insert(pointIdx);
            }
            else
            {
                newSmallPoints.insert(pointIdx);
            }
        }
        for(const size_t &pointIdx : largePoints)
        {
            newLargePoints.insert(pointIdx);
        }

        // treat large points
        for(const size_t &pointIdx : newLargePoints)
        {
            const Vector3D &point = this->del_.points_[pointIdx];
            for(const size_t &tetraIdx : this->PointTetras_[pointIdx])
            {
                if(!this->del_.tetras_[tetraIdx].newTetra)
                {
                    continue; // tetra does not need to be checked
                }
                const Vector3D &center = this->tetra_centers_[tetraIdx];
                double radius = this->GetRadius(tetraIdx);
                // from each big tetrahedron, ask each one of the intersecting ranks to give us the closest point it has to our point
                RangeQueryData query;
                // for a large point queries, if the iteration number is 2, we ask only the near ranks to give their closest point.
                // From the 3rd iteration, we ask all the intersecting ranks to give their closest point.
                if(iterations == 2)
                {
                    query = {pointIdx, {center.x, center.y, center.z}, {point.x, point.y, point.z}, radius, MAX_POINTS_IN_BIG_TETRA_QUERY, ASK_ONLY_CLOSE};
                }
                else
                {
                    query = {pointIdx, {center.x, center.y, center.z}, {point.x, point.y, point.z}, radius, MAX_POINTS_IN_BIG_TETRA_QUERY, ASK_ALL};
                }
                // add the tetra to the list of tetrahedra to clear (mark as 'not new')
                tetraToCancel.insert(tetraIdx);
                queries.push(query);
            }
        }

        // treat small points
        for(const size_t &pointIdx : newSmallPoints)
        {
            // submit one query which is a union of the others
            const Vector3D &point = this->del_.points_[pointIdx];
            double radius = currentRadiuses[pointIdx] *= RADIUSES_GROWING_FACTOR; // increase radius by 'RADIUSES_GROWING_FACTOR'
            // from each big tetrahedron, ask each one of the intersecting ranks to give us the closest point it has to our point
            RangeQueryData query = {pointIdx, {point.x, point.y, point.z}, {point.x, point.y, point.z}, radius, NO_MAX_POINTS, ASK_ALL};
            queries.push(query);
        }
    }

    if(iterations != 2)
    {
        for(const size_t &tetraIdx : tetraToCancel)
        {
            this->del_.tetras_[tetraIdx].newTetra = false;
        }
    }
    smallPoints = std::move(newSmallPoints);
    largePoints = std::move(newLargePoints);

    // std::cout << "has " << smallPoints.size() << " small points, " << largePoints.size() << " large points" << std::endl;
    return queries;
}

/**
 * \author Maor Mizrachi
 * \brief Gets a list of query, and tests for creating mirror points. In the end of this procedure, `mirroredPoints` contains pairs of <faceIdx, pointIdx>, of points that should be mirrored, in relative to which faces
*/
std::vector<std::pair<size_t, size_t>> Voronoi3D::MirrorPoints(std::queue<RangeQueryData> &queries, const std::vector<Face> &box, const std::vector<Vector3D> &normals)
{
    std::vector<std::pair<size_t, size_t>> mirroredPoints;
    std::queue<RangeQueryData> queriesBackup;
    while(!queries.empty())
    {
        RangeQueryData query = queries.front();
        queries.pop();
        queriesBackup.push(query);

        // check for mirroring:
        const Vector3D point(query.center.x, query.center.y, query.center.z);
        double radius = query.radius;
        size_t pointIdx = query.pointIndex;

        std::vector<size_t> facesItCuts = this->CheckToMirror(point, radius, box, normals);
        for(const size_t &faceIdx : facesItCuts)
        {
            mirroredPoints.push_back(std::make_pair(faceIdx, pointIdx));
        }
    }
    queries = std::move(queriesBackup);
    return mirroredPoints;
}

/**
 * \author Maor Mizrachi
 * \brief Updates the duplicated points array
*/
void Voronoi3D::UpdateDuplicatedPoints(const std::vector<int> &sentProc, const std::vector<std::vector<size_t>> &sentPoints)
{
    for(size_t i = 0; i < sentProc.size(); i++)
    {
      int _rank = sentProc[i];
      size_t rankIdx = std::find(this->duplicatedprocs_.begin(), this->duplicatedprocs_.end(), _rank) - this->duplicatedprocs_.begin();
      if(rankIdx == this->duplicatedprocs_.size())
      {
        // TODO: necessary? If `rankIdx` didn't appear in `this->duplicatedprocs_`, we will delete it in the next part
        // new rank in this->duplicatedprocs_, initialize it
        this->duplicatedprocs_.push_back(_rank);
        this->duplicated_points_.emplace_back(std::vector<size_t>());
        this->Nghost_.emplace_back(std::vector<size_t>());
      }
      for(const size_t &pointIdx : sentPoints[i])
      {
        this->duplicated_points_[rankIdx].push_back(pointIdx);
      }
    }
}

/**
 * \author Maor Mizrachi
 * \brief Ensures that the duplicated and ghost arrays contain only the points from/to ranks which are intersecting (sent iff received)
*/
void Voronoi3D::EnsureSymmetry(const std::vector<int> &sentProc, const std::vector<int> &recvProc)
{
    for(size_t i = 0; i < this->duplicatedprocs_.size(); i++)
    {
        int _rank =  this->duplicatedprocs_[i];
        if((std::find(sentProc.begin(), sentProc.end(), _rank) == sentProc.end()) or (std::find(recvProc.begin(), recvProc.end(), _rank) == recvProc.end()))
        {
            // not in the intersection, remove the rank
            this->duplicatedprocs_.erase(this->duplicatedprocs_.begin() + i);
            this->duplicated_points_.erase(this->duplicated_points_.begin() + i);
            this->Nghost_.erase(this->Nghost_.begin() + i);
            i--;
        }
    }
}

void Voronoi3D::InitialExchange(const std::vector<Vector3D> &points, std::vector<int> &sentProc, std::vector<std::vector<size_t>> &sentPoints)
{
    DistributedOctEnvironmentAgent *octEnvAgent = dynamic_cast<DistributedOctEnvironmentAgent*>(this->envAgent);
    if(octEnvAgent == nullptr)
    {
        return;
    }
    const DistributedOctTree<Vector3D> *octTree = octEnvAgent->getOctTree();
    if(octTree == nullptr)
    {
        return;
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t counter = 0;

    for(size_t pointIdx = 0; pointIdx < points.size(); pointIdx++)
    {
        bool isBorderPoint = false;
        for(const size_t &tetraIdx : this->PointTetras_[pointIdx])
        {
            const Tetrahedron &tet = this->del_.tetras_[tetraIdx];
            isBorderPoint = (tet.points[0] >= this->Norg_) or (tet.points[1] >= this->Norg_) or (tet.points[2] >= this->Norg_) or (tet.points[3] >= this->Norg_);
            if(isBorderPoint)
            {
                break;
            }
        }
        if(!isBorderPoint)
        {
            continue;
        }
        int closestRank = std::numeric_limits<int>::max();
        double closestDistance = std::numeric_limits<double>::max();
        auto distances = octTree->getClosestFurthestPointsByRanks(points[pointIdx]);
        for(int _rank = 0; _rank < size; _rank++)
        {
            if(_rank == rank)
            {
                continue;
            }
            if(distances[_rank].first < closestDistance)
            {
                closestDistance = distances[_rank].first;
                closestRank = _rank;
            }
        }
        size_t rankIdx = std::distance(sentProc.begin(), std::find(sentProc.begin(), sentProc.end(), closestRank));
        if(rankIdx == sentProc.size())
        {
            sentProc.push_back(closestRank);
            sentPoints.emplace_back(std::vector<size_t>());
        }
        sentPoints[rankIdx].push_back(pointIdx);
        counter++;
    }

    std::vector<size_t> sendLengths(size, 0);
    std::vector<std::vector<_3DPoint>> toSend;
    toSend.resize(sentProc.size());
    
    std::vector<MPI_Request> requests;
    requests.reserve(4 * sentProc.size()); // heuristic

    std::vector<size_t> recvLengths(size, 0);

    for(size_t i = 0; i < sentProc.size(); i++)
    {
        int _rank = sentProc[i];
        sendLengths[_rank] = sentPoints[i].size();
        toSend[i].reserve(sentPoints[i].size());
        for(size_t &pointIdx : sentPoints[i])
        {
            toSend[i].emplace_back(_3DPoint(points[pointIdx].x, points[pointIdx].y, points[pointIdx].z));
        }
    }

    MPI_Alltoall(&sendLengths[0], sizeof(size_t), MPI_BYTE, &recvLengths[0], sizeof(size_t), MPI_BYTE, MPI_COMM_WORLD);

    size_t totalLength = 0; 
    for(int _rank = 0; _rank < size; _rank++)
    {
        if(recvLengths[_rank] > 0)
        {
            totalLength += recvLengths[_rank];
            size_t rankIdx = std::distance(sentProc.begin(), std::find(sentProc.begin(), sentProc.end(), _rank));
            if(rankIdx != sentProc.size())
            {
                // rank has already been found
                continue;
            }
            sentProc.push_back(_rank);
            sentPoints.emplace_back(std::vector<size_t>());
        }
    }

    std::vector<_3DPoint> almostExtraPoints;
    almostExtraPoints.resize(totalLength);
    size_t insertedSoFar = 0; 
    for(const int &_rank : sentProc)
    {
        if(recvLengths[_rank] > 0)
        {
            // std::cout << "rank " << rank << " is receiving " << recvLengths[_rank] << " from rank " << _rank << ", insertedSoFar is " << insertedSoFar << "(total length: " << totalLength << ")" << std::endl;
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Irecv(&almostExtraPoints[insertedSoFar], sizeof(_3DPoint) * recvLengths[_rank], MPI_BYTE, _rank, INITIAL_SENDRECV_TAG, MPI_COMM_WORLD, &requests[requests.size() - 1]);
            size_t rankIdx = std::distance(sentProc.begin(), std::find(sentProc.begin(), sentProc.end(), _rank));
            
            size_t dupRankIdx = std::find(this->duplicatedprocs_.begin(), this->duplicatedprocs_.end(), _rank) - this->duplicatedprocs_.begin();
            if(dupRankIdx == this->duplicatedprocs_.size())
            {
                // new rank in this->duplicatedprocs_, initialize it
                this->duplicatedprocs_.push_back(_rank);
                this->duplicated_points_.emplace_back(std::vector<size_t>());
                this->Nghost_.emplace_back(std::vector<size_t>());
            }
            for(size_t i = 0; i < recvLengths[_rank]; i++)
            {
                // batchInfo.pointsFromRanks[_rank][i] holds an index of point, but this point will be added to my delaunay, so
                // its index there will be this->del_.points_.size() + batchInfo.pointsFromRanks[_rank][i]
                this->Nghost_[dupRankIdx].push_back(this->del_.points_.size() + insertedSoFar + i);
            }
            insertedSoFar += recvLengths[_rank];
        }
    }

    for(size_t i = 0; i < toSend.size(); i++)
    {
        int _rank = sentProc[i];
        size_t dupRankIdx = std::find(this->duplicatedprocs_.begin(), this->duplicatedprocs_.end(), _rank) - this->duplicatedprocs_.begin();        
        // std::cout << "rank " << rank << " is sending " << toSend[i].size() << " to rank " << _rank << std::endl;
        requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend(&toSend[i][0], sizeof(_3DPoint) * toSend[i].size(), MPI_BYTE, _rank, INITIAL_SENDRECV_TAG, MPI_COMM_WORLD, &requests[requests.size() - 1]);
    }

    if(!requests.empty())
    {
        MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
    }   

    std::vector<Vector3D> extraPoints;
    for(const _3DPoint &_point : almostExtraPoints)
    {
        extraPoints.emplace_back(Vector3D(_point.x, _point.y, _point.z));
    }

    this->del_.BuildExtra(extraPoints);

    this->R_.resize(this->del_.tetras_.size());
    std::fill(this->R_.begin(), this->R_.end(), -1);
    this->tetra_centers_.resize(this->R_.size());
    this->bigtet_ = SetPointTetras(this->PointTetras_, this->Norg_, this->del_.tetras_, this->del_.empty_tetras_);
}

/**
 * \author Maor Mizrachi
 * \brief The algorithm follows arepro paper (https://www.mpa-garching.mpg.de/~volker/arepo/arepo_paper.pdf), section 2.4.
*/
void Voronoi3D::BringGhostPointsToBuild(const std::vector<Vector3D> &points)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<Face> box;
    std::vector<Vector3D> normals;
    this->InitialBoxBuild(box, normals);

    bool sent_finished = false; // if I sent a finished message
    int finished = 0; // the number of finished ranks
    boost::container::flat_set<size_t> smallPoints; // indices of 'small' points
    boost::container::flat_set<size_t> largePoints; // indices of 'large' points
    // initialize `smallPoints`, as all the points (indices)
    for(size_t i = 0; i < points.size(); i++)
    {
        smallPoints.insert(i);
    }

    //BruteForceFinder rangeFinder(this->del_.points_.begin(), this->del_.points_.begin() + this->Norg_);
    //RangeTreeFinder rangeFinder(this->del_.points_.begin(), this->del_.points_.begin() + this->Norg_);
    OctTreeFinder rangeFinder(this->del_.points_.begin(), this->del_.points_.begin() + this->Norg_, this->ll_, this->ur_);
    //HashBruteForceFinder rangeFinder(this->envAgent, this->del_.points_.begin(), this->del_.points_.begin() + this->Norg_);
    //SmartBruteForceFinder rangeFinder(this->envAgent, this->del_.points_.begin(), this->del_.points_.begin() + this->Norg_);
    //KDTreeFinder rangeFinder(this->del_.points_.begin(), this->del_.points_.begin() + this->Norg_, this->ll_, this->ur_);
    //GroupRangeTreeFinder<256> rangeFinder(this->del_.points_.begin(), this->del_.points_.begin() + this->Norg_);
    
    std::vector<int> sentProc_;
    std::vector<std::vector<size_t>> sentPoints_;

    // this->InitialExchange(points, sentProc_, sentPoints_);
    // std::cout << "rank " << rank << " finished initial exchange" << std::endl;
    RangeAgent rangeAgent(this->envAgent, &rangeFinder, sentProc_, sentPoints_);

    std::vector<std::pair<size_t, size_t>> allMirrored;
    int iterations = 0;

    std::vector<double> currentRadiuses = this->radiuses;
    for(size_t i = 0; i < currentRadiuses.size(); i++)
    {
        double radius = currentRadiuses[i];
    }

    MPI_Request finishedReq;

    int I_finished = 0;
    int numFinished;

    while(true) // loop is not really infinite (has 'break')
    {
        iterations++;
        if(rank == 0) std::cout << "iteration " << iterations << std::endl;

        std::queue<RangeQueryData> queries = this->CreateBatches(smallPoints, largePoints, currentRadiuses, iterations);

        std::vector<std::pair<size_t, size_t>> mirroredPoints = this->MirrorPoints(queries, box, normals);

        I_finished = queries.empty()? 1 : 0;
        
        MPI_Iallreduce(&I_finished, &numFinished, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &finishedReq);

        QueryBatchInfo<RangeQueryData, _3DPoint> batchInfo = rangeAgent.runBatch(queries);

        boost::container::flat_set<size_t> newSmallPoints, newLargePoints;
        for(const QueryInfo<RangeQueryData, _3DPoint> &ans : batchInfo.queriesAnswers)
        {
            const size_t &pointIdx = ans.data.pointIndex;
            if(ans.data.maxPointsToGet == NO_MAX_POINTS)
            {
                // small query
                double maxRadius = this->GetMaxRadius(pointIdx);
                if(currentRadiuses[pointIdx] < 2 * maxRadius)
                {
                    // point is not yet done!
                    newSmallPoints.insert(pointIdx);
                }
                this->radiuses[pointIdx] = RADIUSES_GROWING_FACTOR * (2 * maxRadius);
            }
            else
            {
                // query is large, check if it returned non empty. If yes, we are not yet done
                if(!ans.finalResults.empty() or iterations == 2)
                {
                    newLargePoints.insert(pointIdx);
                }
            }
        }

        for(const size_t &pointIdx : largePoints)
        {
            if(newLargePoints.find(pointIdx) == newLargePoints.end())
            {
                // point was large, and is finished
                this->radiuses[pointIdx] = RADIUSES_GROWING_FACTOR * (2 * this->GetMinRadius(pointIdx));
            }
        }
        smallPoints = std::move(newSmallPoints);
        largePoints = std::move(newLargePoints);
        
        std::vector<_3DPoint> &_newPoints = batchInfo.result;
        std::vector<Vector3D> newPoints;
        newPoints.reserve(_newPoints.size());
        for(const _3DPoint &_point : _newPoints)
        {
           // std::cout << "rank " << rank << " in iteration " << iterations << ", received point " << Vector3D(_point.x, _point.y, _point.z) << std::endl;
            newPoints.push_back(Vector3D(_point.x, _point.y, _point.z));
        }

        const std::vector<int> &recvProc = rangeAgent.getRecvProc();
        const std::vector<std::vector<size_t>> &recvPoints = rangeAgent.getRecvPoints();

        for(size_t i = 0; i < recvProc.size(); i++)
        {
            int _rank = recvProc[i];
            const std::vector<size_t> &receivedFromRank = recvPoints[i];
            size_t rankIdx = std::find(this->duplicatedprocs_.begin(), this->duplicatedprocs_.end(), _rank) - this->duplicatedprocs_.begin();
            if(rankIdx == this->duplicatedprocs_.size())
            {
                // new rank in this->duplicatedprocs_, initialize it
                this->duplicatedprocs_.push_back(_rank);
                this->duplicated_points_.emplace_back(std::vector<size_t>());
                this->Nghost_.emplace_back(std::vector<size_t>());
            }
            for(const size_t &RelativePointIdx : receivedFromRank)
            {
                // batchInfo.pointsFromRanks[_rank][i] holds an index of point, but this point will be added to my delaunay, so
                // its index there will be this->del_.points_.size() + batchInfo.pointsFromRanks[_rank][i]
                this->Nghost_[rankIdx].push_back(this->del_.points_.size() + RelativePointIdx);
            }
        }
        // mirror points:
        allMirrored.reserve(allMirrored.size() + mirroredPoints.size());
        newPoints.reserve(newPoints.size() + mirroredPoints.size());
        for(const std::pair<size_t, size_t> &pairFacePoint : mirroredPoints)
        {
            // check if we have already mirrored this point with this face
            if(std::find(allMirrored.begin(), allMirrored.end(), pairFacePoint) == allMirrored.end())
            {
                allMirrored.push_back(pairFacePoint); // remember we mirrored this point with this face
                newPoints.push_back(MirrorPoint(box[pairFacePoint.first], this->del_.points_[pairFacePoint.second]));
            }
        }

        // performs internal tesselation:
        this->del_.BuildExtra(newPoints);

        this->R_.resize(this->del_.tetras_.size(), -1);
        for(size_t tetraIdx = 0; tetraIdx < this->R_.size(); tetraIdx++)
        {
            if(this->del_.tetras_[tetraIdx].newTetra)
            {
                this->R_[tetraIdx] = -1;
            }
        }
        // std::fill(this->R_.begin(), this->R_.end(), -1);
        this->tetra_centers_.resize(this->R_.size());
        this->bigtet_ = SetPointTetras(this->PointTetras_, this->Norg_, this->del_.tetras_, this->del_.empty_tetras_);

        // std::cout << "del_.points_.size() for rank " << rank << " is " << del_.points_.size() << std::endl;
        MPI_Wait(&finishedReq, MPI_STATUS_IGNORE);
        if(numFinished == size)
        {
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const std::vector<std::vector<size_t>> &sentPoints = rangeAgent.getSentPoints();
    const std::vector<int> &sentProc = rangeAgent.getSentProc();
    const std::vector<int> &recvProc = rangeAgent.getRecvProc();

    // calculate this->duplicated_points_
    this->UpdateDuplicatedPoints(sentProc, sentPoints);
    // remove whomever that does not appear both in my sent vector and receive vector (because if one appears in only one, it means that we either sent it a point, or received one, but has no used of it at all (otherwise it would require a symetric call))
    this->EnsureSymmetry(sentProc, recvProc);    
}

/**
 * \author Maor Mizrachi
 * \brief Calculates the initial radius for the circles in the AREPRO algorithm
*/
void Voronoi3D::CalculateInitialRadius(size_t pointsSize)
{
    this->radiuses.resize(pointsSize);
    if(std::abs(this->initialRadius) <= EPSILON)
    {
        // initial radius is zero, so we need to determine it
      double volume = (this->ur_[0] - this->ll_[0]) * (this->ur_[1] - this->ll_[1]) * (this->ur_[2] - this->ll_[2]);
      size_t N;
      MPI_Allreduce(&pointsSize, &N, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
      this->initialRadius = 2 * std::pow(volume / N, 0.333333f); // heuristic
    } 
    std::fill(this->radiuses.begin(), this->radiuses.end(), this->initialRadius);
}

/**
 * \author Maor Mizrachi
 * \brief Makes load rebalancing if needed, if needed, and initializing the environment agent (the object which is responsible for dividing the space to ranks)
*/
std::vector<Vector3D> Voronoi3D::PrepareToBuildHilbert(const std::vector<Vector3D> &points)
{
    if(this->firstCall == true)
    {
        // first call
        this->CalculateInitialRadius(points.size());
    }

    if(this->radiuses.size() < points.size())
    {
        // actually, should not reach here
        this->radiuses.resize(points.size(), this->initialRadius);
    }

    PointsExchangeResult exchangeResult;

    if(pointsManager.checkForRebalance(points) or (this->firstCall == true) or (this->envAgent == nullptr))
    {
        // calculate the first and initial order, and set it to the deepest hilbert order we have
        OctTree<Vector3D> tree(this->ll_, this->ur_, points);
        int depth = tree.getDepth(); // my own depth
        MPI_Allreduce(&depth, &this->hilbertOrder, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // calculates maximal depth
        this->hilbertOrder = std::min<size_t>(MAX_ALLOWED_HILBERT_ORDER, this->hilbertOrder);
        this->responsibilityRange = this->pointsManager.redetermineBorders(points, this->hilbertOrder); // recalculates borders accoridng to the deepest order
        exchangeResult = this->pointsManager.pointsExchange(this->responsibilityRange, this->hilbertOrder, points, this->radiuses); // exchange
        if(this->envAgent != nullptr)
        {
            this->envAgent->updateBorders(this->responsibilityRange, this->hilbertOrder);
        }
    }
    else
    {
        // perform points exchange, according to the environment agent
        exchangeResult = this->pointsManager.pointsExchangeByEnvAgent(this->envAgent, points, this->radiuses);
    }
    
    std::vector<Vector3D> new_points = std::move(exchangeResult.newPoints);
    this->radiuses = std::move(exchangeResult.newRadiuses);
    this->sentprocs_ = std::move(exchangeResult.sentProcessors);
    this->sentpoints_ = std::move(exchangeResult.sentIndicesToProcessors);
    this->self_index_ = std::move(exchangeResult.indicesToSelf);
    this->BuildInitialize(new_points.size());

    if(this->firstCall or (this->envAgent == nullptr))
    {
        // create new environment agent
        this->envAgent = new DistributedOctEnvironmentAgent(this->ll_, this->ur_, new_points, this->responsibilityRange, this->hilbertOrder);
        this->firstCall = false;
    }
    else
    {
        // update the existing environment agent
        assert(this->envAgent != nullptr);
        this->envAgent->update(new_points);
    }
    return new_points;
}

/**
 * \author Maor Mizrachi
 * \brief Build the voronoi, after rebalancing the points using a proper hilbert curve. 
*/
void Voronoi3D::BuildHilbert(const std::vector<Vector3D> &points)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bool first_call = this->firstCall;
    std::vector<Vector3D> new_points = this->PrepareToBuildHilbert(points);
    // std::cout << "points.size() was " << points.size() << " and now is " << new_points.size() << std::endl;
    
    std::vector<size_t> order;

    // build delaunay
    if(!new_points.empty())
    {
        std::pair<Vector3D, Vector3D> bounding_box = std::make_pair(new_points[0], new_points[0]);
        for(const Vector3D &point : new_points)
        {
            bounding_box.first.x = std::min(bounding_box.first.x, point.x);
            bounding_box.second.x = std::max(bounding_box.second.x, point.x);
            bounding_box.first.y = std::min(bounding_box.first.y, point.y);
            bounding_box.second.y = std::max(bounding_box.second.y, point.y);
            bounding_box.first.z = std::min(bounding_box.first.z, point.z);
            bounding_box.second.z = std::max(bounding_box.second.z, point.z);
        }

        // performs internal tesselation:
        // std::cout << "checking duplications..." << std::endl;
        // reportDuplications(new_points);
        order = HilbertOrder3D(new_points);
        
        // initial build for the points
        this->del_.Build(new_points, bounding_box.second, bounding_box.first, order);

        // updates the radiuses array of the tetrahedra, as well as the lists for each point what tetras it belongs to
        this->R_.resize(this->del_.tetras_.size());
        std::fill(this->R_.begin(), this->R_.end(), -1);
        this->tetra_centers_.resize(this->R_.size());
        this->bigtet_ = SetPointTetras(this->PointTetras_, this->Norg_, this->del_.tetras_, this->del_.empty_tetras_);

        if(first_call)
        {
            OctTree<Vector3D> myOctTree(this->ll_, this->ur_, new_points);
            for(size_t pointIdx = 0; pointIdx < new_points.size(); pointIdx++)
            {
                // todo second closest
                this->radiuses[pointIdx] = RADIUSES_GROWING_FACTOR * (100 * myOctTree.closestPointDistance(this->del_.points_[pointIdx]));
            }
        }
    }

    if(this->radiuses.size() != new_points.size())
    {
        throw UniversalError("Rank " + std::to_string(rank) + ", wrong size of radiuses (in voronoi build) (given this->radiuses.size()=" + std::to_string(this->radiuses.size()) + " while should be new_points.size()=" + std::to_string(new_points.size()) + ")");
    }

    this->BringGhostPointsToBuild(new_points);

    CM_.resize(del_.points_.size());
    volume_.resize(Norg_);

    if(new_points.size() != 0)
    {
        // Create Voronoi
        BuildVoronoi(order);
    }
    std::vector<double>().swap(R_);
    std::vector<tetra_vec>().swap(PointTetras_);

    CalcAllCM();
    for (std::size_t i = 0; i < FaceNeighbors_.size(); ++i)
        if (BoundaryFace(i))
            CalcRigidCM(i);

    // communicate the ghost CM
    
    // vector<vector<Vector3D>> incoming = MPI_Exchange_serializable(CM_, duplicatedprocs_, duplicated_points_);
    vector<vector<Vector3D>> incoming = MPI_exchange_data(duplicatedprocs_, duplicated_points_, CM_);
    // Add the recieved CM
    for (size_t i = 0; i < incoming.size(); ++i)
    {
        for (size_t j = 0; j < incoming.at(i).size(); ++j)
        {
            CM_[Nghost_.at(i).at(j)] = incoming[i][j];
        }
    }   
}

#endif // RICH_MPI

vector<vector<std::size_t>> &Voronoi3D::GetGhostIndeces(void)
{
    return Nghost_;
}

void Voronoi3D::CalcAllCM(void)
{
    std::array<Vector3D, 4> tetra;
    size_t Nfaces = FaceNeighbors_.size();
    assert(Nfaces == 0 or Nfaces >= 4);
    Vector3D vtemp;
    std::vector<Vector3D> vectemp;
    double vol;
    for (size_t i = 0; i < Nfaces; ++i)
    {
        size_t N0 = FaceNeighbors_[i].first;
        size_t N1 = FaceNeighbors_[i].second;
        size_t Npoints = PointsInFace_[i].size();
        vectemp.resize(Npoints);
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
        for (size_t j = 0; j < Npoints; ++j)
            vectemp[j] = tetra_centers_[PointsInFace_[i][j]];
        Npoints -= 2;
        tetra[0] = vectemp[0];

        for (std::size_t j = 0; j < Npoints; ++j)
        {
            tetra[1] = vectemp[j + 1];
            tetra[2] = vectemp[j + 2];
            if (N1 < Norg_)
            {
                tetra[3] = del_.points_[N1];
                vol = std::abs(GetTetraVolume(tetra));
                GetTetraCM(tetra, vtemp);
                volume_[N1] += vol;
                vtemp *= vol;
                CM_[N1] += vtemp;
            }
            tetra[3] = del_.points_[N0];
            vol = std::abs(GetTetraVolume(tetra));
            GetTetraCM(tetra, vtemp);
            volume_[N0] += vol;
            vtemp *= vol;
            CM_[N0] += vtemp;
        }
    }
#ifdef __INTEL_COMPILER
    //#pragma vector aligned
#pragma omp simd
#endif
    for (size_t i = 0; i < Norg_; ++i)
        CM_[i] *= (1.0 / volume_[i]);
    // Recalc points with high aspect ratio
    for (size_t i = 0; i < Norg_; ++i)
    {
        if (fastabs(CM_[i] - del_.points_[i]) > 0.4 * GetWidth(i))
        {
            tetra[3] = CM_[i];
            CM_[i] = Vector3D();
            volume_[i] = 0;
            Nfaces = FacesInCell_[i].size();
            for (size_t k = 0; k < Nfaces; ++k)
            {
                size_t Face = FacesInCell_[i][k];
                size_t Npoints = PointsInFace_[Face].size();
                tetra[0] = tetra_centers_[PointsInFace_[Face][0]];
                for (std::size_t j = 0; j < Npoints - 2; ++j)
                {
                    tetra[1] = tetra_centers_[PointsInFace_[Face][j + 1]];
                    tetra[2] = tetra_centers_[PointsInFace_[Face][j + 2]];
                    double vol2 = std::abs(GetTetraVolume(tetra));
                    volume_[i] += vol2;
                    GetTetraCM(tetra, vtemp);
                    CM_[i] += vol2 * vtemp;
                }
            }
            CM_[i] *= (1.0 / volume_[i]);
        }
    }
}

std::pair<Vector3D, Vector3D> Voronoi3D::GetBoxCoordinates(void) const
{
    return std::pair<Vector3D, Vector3D>(ll_, ur_);
}

void Voronoi3D::BuildNoBox(vector<Vector3D> const &points, vector<vector<Vector3D>> const &ghosts, vector<size_t> toduplicate)
{
    assert(points.size() > 0);
    // Clear data
    PointTetras_.clear();
    R_.clear();
    R_.reserve(points.size());
    tetra_centers_.clear();
    tetra_centers_.reserve(points.size() * 7);
    del_.Clean();
    // Voronoi Data
    FacesInCell_.clear();
    PointsInFace_.clear();
    FaceNeighbors_.clear();
    CM_.clear();
    Face_CM_.clear();
    volume_.clear();
    area_.clear();
    Norg_ = points.size();
    duplicatedprocs_.clear();
    duplicated_points_.clear();
    Nghost_.clear();
    std::vector<size_t> order = HilbertOrder3D(points);
    del_.Build(points, ur_, ll_, order);
    for (size_t i = 0; i < ghosts.size(); ++i)
    {
        del_.BuildExtra(ghosts[i]);
    }
    vector<std::pair<size_t, size_t>> duplicate(6);
    for (size_t j = 0; j < toduplicate.size(); ++j)
    {
#ifdef __INTEL_COMPILER
#pragma omp simd
#endif
        for (size_t i = 0; i < 6; ++i)
            duplicate[i] = std::pair<size_t, size_t>(i, toduplicate[j]);
        vector<vector<size_t>> past_duplicates;
        vector<Vector3D> extra_points = CreateBoundaryPoints(duplicate, past_duplicates);
        del_.BuildExtra(extra_points);
    }

    R_.resize(del_.tetras_.size());
    std::fill(R_.begin(), R_.end(), -1);
    tetra_centers_.resize(R_.size());
    bigtet_ = SetPointTetras(PointTetras_, Norg_, del_.tetras_, del_.empty_tetras_);

    CM_.resize(Norg_);
    volume_.resize(Norg_, 0);
    // Create Voronoi
    BuildVoronoi(order);

    CalcAllCM();
    CM_.resize(del_.points_.size());
    for (std::size_t i = 0; i < FaceNeighbors_.size(); ++i)
        if (BoundaryFace(i))
            CalcRigidCM(i);
}

void Voronoi3D::BuildDebug(int rank)
{
    std::vector<size_t> order = read_vecst("order_" + int2str(rank) + ".bin");
    std::vector<Vector3D> points = read_vec3d("points0_" + int2str(rank) + ".bin");
    Norg_ = points.size();
    std::vector<Vector3D> bb = read_vec3d("bb_" + int2str(rank) + ".bin");
    del_.Build(points, bb[1], bb[0], order);
    points = read_vec3d("points1_" + int2str(rank) + ".bin");
    del_.BuildExtra(points);
    points = read_vec3d("points2_" + int2str(rank) + ".bin");
    del_.BuildExtra(points);
    points = read_vec3d("points3_" + int2str(rank) + ".bin");
    del_.BuildExtra(points);
    points = read_vec3d("points4_" + int2str(rank) + ".bin");
    del_.BuildExtra(points);

    bigtet_ = SetPointTetras(PointTetras_, Norg_, del_.tetras_, del_.empty_tetras_);

    R_.resize(del_.tetras_.size());
    std::fill(R_.begin(), R_.end(), -1);
    tetra_centers_.resize(R_.size());

    CM_.resize(del_.points_.size());
    volume_.resize(Norg_, 0);
    // Create Voronoi
    BuildVoronoi(order);

    std::vector<double>().swap(R_);
    std::vector<tetra_vec>().swap(PointTetras_);
    std::vector<Tetrahedron>().swap(del_.tetras_);

    CalcAllCM();
    for (std::size_t i = 0; i < FaceNeighbors_.size(); ++i)
        if (BoundaryFace(i))
            CalcRigidCM(i);
}

void Voronoi3D::Build(vector<Vector3D> const &points)
{
    assert(points.size() > 0);
    // Clear data
    PointTetras_.clear();
    R_.clear();
    R_.reserve(points.size() * 11);
    tetra_centers_.clear();
    tetra_centers_.reserve(points.size() * 11);
    // Voronoi Data
    FacesInCell_.clear();
    PointsInFace_.clear();
    FaceNeighbors_.clear();
    CM_.clear();
    Face_CM_.clear();
    volume_.clear();
    area_.clear();
    Norg_ = points.size();
    duplicatedprocs_.clear();
    duplicated_points_.clear();
    Nghost_.clear();
    std::vector<size_t> order = HilbertOrder3D(points);
    del_.Build(points, ur_, ll_, order);

    R_.resize(del_.tetras_.size());
    std::fill(R_.begin(), R_.end(), -1);
    tetra_centers_.resize(R_.size());
    bigtet_ = SetPointTetras(PointTetras_, Norg_, del_.tetras_, del_.empty_tetras_);

    vector<std::pair<std::size_t, std::size_t>> ghost_index = SerialFirstIntersections();
    vector<vector<size_t>> past_duplicates;
    vector<Vector3D> extra_points = CreateBoundaryPoints(ghost_index, past_duplicates);

    del_.BuildExtra(extra_points);

    R_.resize(del_.tetras_.size());
    std::fill(R_.begin(), R_.end(), -1);
    tetra_centers_.resize(R_.size());
    bigtet_ = SetPointTetras(PointTetras_, Norg_, del_.tetras_, del_.empty_tetras_);
    ghost_index = SerialFindIntersections(true);
    extra_points = CreateBoundaryPoints(ghost_index, past_duplicates);
    del_.BuildExtra(extra_points);

    R_.resize(del_.tetras_.size());
    std::fill(R_.begin(), R_.end(), -1);
    tetra_centers_.resize(R_.size());
    bigtet_ = SetPointTetras(PointTetras_, Norg_, del_.tetras_, del_.empty_tetras_);
    ghost_index = SerialFindIntersections(false);
    extra_points = CreateBoundaryPoints(ghost_index, past_duplicates);
    del_.BuildExtra(extra_points);
    bigtet_ = SetPointTetras(PointTetras_, Norg_, del_.tetras_, del_.empty_tetras_);

    std::vector<std::pair<size_t, size_t>>().swap(ghost_index);
    std::vector<std::vector<size_t>>().swap(past_duplicates);
    std::vector<Vector3D>().swap(extra_points);

    R_.resize(del_.tetras_.size());
    std::fill(R_.begin(), R_.end(), -1);
    tetra_centers_.resize(R_.size());

    CM_.resize(del_.points_.size());
    volume_.resize(Norg_, 0);
    // Create Voronoi
    BuildVoronoi(order);

    std::vector<double>().swap(R_);
    std::vector<tetra_vec>().swap(PointTetras_);
    std::vector<Tetrahedron>().swap(del_.tetras_);

    CalcAllCM();
    for (std::size_t i = 0; i < FaceNeighbors_.size(); ++i)
        if (BoundaryFace(i))
            CalcRigidCM(i);
}

void Voronoi3D::BuildVoronoi(std::vector<size_t> const &order)
{
    FacesInCell_.resize(Norg_);
    area_.resize(Norg_ * 10);
    Face_CM_.resize(Norg_ * 10);
    FaceNeighbors_.resize(Norg_ * 10);
    PointsInFace_.resize(Norg_ * 10);

    std::array<size_t, 128> temp, temp3;
    // Build all voronoi points
    std::size_t Ntetra = del_.tetras_.size();
    for (size_t i = 0; i < Ntetra; ++i)
        if (ShouldCalcTetraRadius(del_.tetras_[i], Norg_))
            CalcTetraRadiusCenter(i);
    // Organize the faces and assign them to cells
    std::array<double, 128> diffs, Atempvec;

    size_t FaceCounter = 0;
    boost::container::flat_set<size_t> neigh_set;
    point_vec *temp_points_in_face;
    std::array<Vector3D, 128> clean_vec;
    std::array<double, 128> area_vec_temp;

    //std::vector<Vector3D, boost::alignment::aligned_allocator<Vector3D, 32> > clean_vec;
    for (size_t i = 0; i < Norg_; ++i)
    {
        neigh_set.clear();
        neigh_set.reserve(20);
        size_t point = order[i];
        size_t ntet = PointTetras_[point].size();
        // for each point loop over its tetras
        for (size_t j = 0; j < ntet; ++j)
        {
            const size_t tetcheck = PointTetras_[point][j];
            for (size_t k = 0; k < 4; ++k)
            {
                size_t point_other = del_.tetras_[tetcheck].points[k];
                if (point_other != point && point_other > point)
                {
                    // Did we already build this face?
                    if (neigh_set.find(point_other) == neigh_set.end())
                    {
                        size_t temp_size = 0;
                        // Find all tetras for face
                        temp[0] = tetcheck;
                        ++temp_size;
                        size_t next_check = NextLoopTetra(del_.tetras_[tetcheck], tetcheck, point, point_other);
                        size_t cur_check = next_check;
                        size_t last_check = tetcheck;
                        while (next_check != tetcheck)
                        {
                            Tetrahedron const &tet_check = del_.tetras_[cur_check];
                            temp[temp_size] = cur_check;
                            ++temp_size;
                            next_check = NextLoopTetra(tet_check, last_check, point, point_other);
                            last_check = cur_check;
                            cur_check = next_check;
                        }
                        // Is face too small?
                        if (temp_size < 3)
                            continue;
                        temp_points_in_face = &PointsInFace_[FaceCounter];
                        //temp_points_in_face->reserve(8);
                        double Asize = CleanDuplicates(temp, tetra_centers_, *temp_points_in_face, ScalarProd(del_.points_[point] - del_.points_[point_other], del_.points_[point] - del_.points_[point_other]), diffs, clean_vec, temp_size);
                        if (temp_points_in_face->size() < 3)
                            continue;
                        CalcFaceAreaCM(*temp_points_in_face, tetra_centers_, clean_vec, area_[FaceCounter],
                                                     Face_CM_[FaceCounter], Atempvec);
                        if (area_[FaceCounter] < (Asize * (IsPointOutsideBox(point_other) ? 1e-14 : 1e-15)))
                            continue;
                        if (point_other >= Norg_ && point_other < (Norg_ + 4))
                        {
                            UniversalError eo("Neighboring big tet point");
                            throw eo;
                        }
                        // Make faces right handed
                        MakeRightHandFace(*temp_points_in_face, del_.points_[point], tetra_centers_, temp3, area_[FaceCounter]);
                        CleanSameLine(*temp_points_in_face, tetra_centers_, area_vec_temp);
                        FaceNeighbors_[FaceCounter].first = point;
                        FaceNeighbors_[FaceCounter].second = point_other;

                        FacesInCell_[point].push_back(FaceCounter);
                        if (point_other < Norg_)
                        {
                            FacesInCell_[point_other].push_back(FaceCounter);
                        }
                        neigh_set.insert(point_other);
                        ++FaceCounter;
                        // realloc memory if needed
                        if (FaceCounter == FaceNeighbors_.size())
                        {
                            area_.resize(static_cast<size_t>(static_cast<double>(area_.size()) * 1.25));
                            Face_CM_.resize(static_cast<size_t>(static_cast<double>(Face_CM_.size()) * 1.25));
                            FaceNeighbors_.resize(static_cast<size_t>(static_cast<double>(FaceNeighbors_.size()) * 1.25));
                            PointsInFace_.resize(static_cast<size_t>(static_cast<double>(PointsInFace_.size()) * 1.25));
                        }
                    }
                }
            }
        }
    }

    // Fix Face CM (this prevents large face velocities for close by points)
    size_t Nfaces = FaceCounter;
    Vector3D mid, norm;
    for (size_t i = 0; i < Nfaces; ++i)
    {
        mid = del_.points_[FaceNeighbors_[i].first];
        mid += del_.points_[FaceNeighbors_[i].second];
        mid *= 0.5;
        norm = del_.points_[FaceNeighbors_[i].second];
        norm -= del_.points_[FaceNeighbors_[i].first];
        Face_CM_[i] -= ScalarProd(Face_CM_[i] - mid, norm) * norm / ScalarProd(norm, norm);
    }

    area_.resize(FaceCounter);
    area_.shrink_to_fit();
    Face_CM_.resize(FaceCounter);
    Face_CM_.shrink_to_fit();
    FaceNeighbors_.resize(FaceCounter);
    FaceNeighbors_.shrink_to_fit();
    PointsInFace_.resize(FaceCounter);
    PointsInFace_.shrink_to_fit();
    for (size_t i = 0; i < Norg_; ++i)
        FacesInCell_[i].shrink_to_fit();
}

inline double Voronoi3D::GetRadius(std::size_t index)
{
    return (R_[index] = (R_[index] < 0)? CalcTetraRadiusCenter(index) : R_[index]);
}

void Voronoi3D::FindIntersectionsSingle(vector<Face> const &box, std::size_t point, Sphere &sphere,
                                                                                vector<size_t> &intersecting_faces, std::vector<double> &Rtemp, std::vector<Vector3D> &vtemp)
{
    intersecting_faces.clear();
    std::size_t N = PointTetras_[point].size();
    Rtemp.resize(N);
    vtemp.resize(N);
    for (std::size_t i = 0; i < N; ++i)
    {
        Rtemp[i] = GetRadius(PointTetras_[point][i]);
        vtemp[i] = tetra_centers_[PointTetras_[point][i]];
    }
    size_t bsize = box.size();
    for (std::size_t j = 0; j < bsize; ++j)
    {
        Vector3D normal = CrossProduct(box[j].vertices[1] - box[j].vertices[0], box[j].vertices[2] - box[j].vertices[0]);
        normal *= (1.0 / fastsqrt(ScalarProd(normal, normal)));
        for (std::size_t i = 0; i < N; ++i)
        {
            sphere.radius = Rtemp[i];
            sphere.center = vtemp[i];
            if (FaceSphereIntersections(box[j], sphere, normal))
            {
                intersecting_faces.push_back(j);
                break;
            }
        }
    }
}

void Voronoi3D::GetPointToCheck(std::size_t point, vector<unsigned char> const &checked, vector<std::size_t> &res)
{
    res.clear();
    std::size_t ntetra = PointTetras_[point].size();
    for (std::size_t i = 0; i < ntetra; ++i)
    {
        size_t tetra = PointTetras_[point][i];
        for (std::size_t j = 0; j < 4; ++j)
            if (del_.tetras_[tetra].points[j] < Norg_ && checked[del_.tetras_[tetra].points[j]] == 0)
                res.push_back(del_.tetras_[tetra].points[j]);
    }
    std::sort(res.begin(), res.end());
    res = unique(res);
}

std::size_t Voronoi3D::GetFirstPointToCheck(void) const
{
    std::size_t i;
    Tetrahedron const &tet = del_.tetras_[bigtet_];
    for (i = 0; i < 4; ++i)
        if (tet.points[i] < Norg_)
            break;
    if (i < 4)
        return tet.points[i];
    else
        throw UniversalError("Can't find first point to start boundary search");
}

vector<std::pair<std::size_t, std::size_t>> Voronoi3D::SerialFirstIntersections(void)
{
    vector<Face> box;
    vector<Vector3D> normals;
    this->InitialBoxBuild(box, normals);
    size_t Nfaces = box.size();

    //    vector<std::size_t> point_neigh;
    vector<std::pair<std::size_t, std::size_t>> res;
    Sphere sphere;
    vector<unsigned char> will_check(Norg_, 0);
    std::size_t cur_loc;
    std::stack<std::size_t> check_stack;
    FirstCheckList(check_stack, will_check, Norg_, del_, PointTetras_);
    std::vector<double> vdist(Nfaces);
    std::vector<Vector3D> vtemp(Nfaces);
    while (!check_stack.empty())
    {
        cur_loc = check_stack.top();
        check_stack.pop();
        double inv_max = 0;
        size_t max_loc = 0;
        size_t j = 0;
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
        for (; j < Nfaces; ++j)
        {
            vtemp[j] = del_.points_[cur_loc];
            vtemp[j] -= box[j].vertices[0];
            double sprod = vtemp[j].x * normals[j].x + vtemp[j].y * normals[j].y + vtemp[j].z * normals[j].z;
            vdist[j] = 1.0 / std::abs(sprod);
        }
        j = 0;
        for (; j < Nfaces; ++j)
        {
            if (vdist[j] > inv_max)
            {
                inv_max = vdist[j];
                max_loc = j;
            }
        }
        res.push_back(std::pair<std::size_t, std::size_t>(max_loc, cur_loc));
    }
    return res;
}

vector<std::pair<std::size_t, std::size_t>> Voronoi3D::SerialFindIntersections(bool first_run)
{
    if (Norg_ < 50)
    {
        vector<std::pair<std::size_t, std::size_t>> res;
        size_t const Nfaces = box_faces_.empty() ? 6 : box_faces_.size();
        res.reserve(Norg_ * Nfaces);
        for (size_t i = 0; i < Norg_; ++i)
            for (size_t j = 0; j < Nfaces; ++j)
                res.push_back(std::pair<std::size_t, std::size_t>(j, i));
        return res;
    }
    std::stack<std::size_t> check_stack;
    vector<Face> box = box_faces_.empty() ? BuildBox(ll_, ur_) : box_faces_;
    vector<std::size_t> point_neigh;
    vector<std::pair<std::size_t, std::size_t>> res;
    Sphere sphere;
    vector<unsigned char> checked(Norg_, 0), will_check(Norg_, 0);
    std::size_t cur_loc;
    if (first_run)
    {
        FirstCheckList(check_stack, will_check, Norg_, del_, PointTetras_);
        //cur_loc = check_stack.top();
        check_stack.pop();
    }
    else
    {
        cur_loc = GetFirstPointToCheck();
        check_stack.push(cur_loc);
        will_check[cur_loc] = true;
    }
    vector<size_t> intersecting_faces;
    std::vector<double> Rtemp;
    std::vector<Vector3D> vtemp;
    while (!check_stack.empty())
    {
        cur_loc = check_stack.top();
        check_stack.pop();
        checked[cur_loc] = true;
        // Does sphere have any intersections?
        bool added = false;
        FindIntersectionsSingle(box, cur_loc, sphere, intersecting_faces, Rtemp, vtemp);
        if (!intersecting_faces.empty())
        {
            added = true;
            for (std::size_t j = 0; j < intersecting_faces.size(); ++j)
                res.push_back(std::pair<std::size_t, std::size_t>(intersecting_faces[j], cur_loc));
        }
        if (added && !first_run)
        {
            GetPointToCheck(cur_loc, checked, point_neigh);
            std::size_t Nneigh = point_neigh.size();
            for (std::size_t j = 0; j < Nneigh; ++j)
                if (point_neigh[j] < Norg_ && !will_check[point_neigh[j]])
                {
                    check_stack.push(point_neigh[j]);
                    will_check[point_neigh[j]] = true;
                }
        }
    }
    return res;
}

double Voronoi3D::CalcTetraRadiusCenter(std::size_t index)
{
    Vector3D v2(del_.points_[del_.tetras_[index].points[1]]);
    v2 -= del_.points_[del_.tetras_[index].points[0]];
    Vector3D v3(del_.points_[del_.tetras_[index].points[2]]);
    v3 -= del_.points_[del_.tetras_[index].points[0]];
    Vector3D v4(del_.points_[del_.tetras_[index].points[3]]);
    v4 -= del_.points_[del_.tetras_[index].points[0]];

    Mat33<double> m_a(v2.x, v2.y, v2.z,
                                        v3.x, v3.y, v3.z,
                                        v4.x, v4.y, v4.z);
    double a = m_a.determinant();
    if(std::abs(a) < 100 * std::numeric_limits<double>::min())
        return CalcTetraRadiusCenterHiPrecision(index);
    Mat33<double> m_Dx(ScalarProd(v2, v2), v2.y, v2.z,
                                         ScalarProd(v3, v3), v3.y, v3.z,
                                         ScalarProd(v4, v4), v4.y, v4.z);
    double DDx = m_Dx.determinant();

    Mat33<double> m_Dy(ScalarProd(v2, v2), v2.x, v2.z,
                                         ScalarProd(v3, v3), v3.x, v3.z,
                                         ScalarProd(v4, v4), v4.x, v4.z);
    double DDy = -m_Dy.determinant();

    Mat33<double> m_Dz(ScalarProd(v2, v2), v2.x, v2.y,
                                         ScalarProd(v3, v3), v3.x, v3.y,
                                         ScalarProd(v4, v4), v4.x, v4.y);
    double DDz = m_Dz.determinant();
    Vector3D center = Vector3D(DDx / (2 * a), DDy / (2 * a), DDz / (2 * a)) + del_.points_[del_.tetras_[index].points[0]];
    tetra_centers_[index] = center;
    double Rres = 0.5 * std::sqrt(DDx * DDx + DDy * DDy + DDz * DDz) / std::abs(a);
    // Sanity check
    /*double Rcheck0 = fastabs(del_.points_[del_.tetras_[index].points[0]] - center);
        double Rcheck1 = fastabs(del_.points_[del_.tetras_[index].points[1]] - center);
        double Rcheck2 = fastabs(del_.points_[del_.tetras_[index].points[2]] - center);
        double Rcheck3 = fastabs(del_.points_[del_.tetras_[index].points[3]] - center);*/
    Vector3D v1(del_.points_[del_.tetras_[index].points[0]]);
    double Rcheck0 = fastabs(v1 - center);
    v2 += v1;
    v2 -= center;
    double Rcheck1 = fastabs(v2);
    v3 += v1;
    v3 -= center;
    double Rcheck2 = fastabs(v3);
    v4 += v1;
    v4 -= center;
    double Rcheck3 = fastabs(v4);
    double tol = 1 + 1e-6;
    if (((Rcheck0 + Rcheck1 + Rcheck2 + Rcheck3) * tol < (4 * Rcheck0)) || ((Rcheck0 + Rcheck1 + Rcheck2 + Rcheck3) > (tol * 4 * Rcheck0)))
        return CalcTetraRadiusCenterHiPrecision(index);
    if (Rcheck0 > tol * Rres || Rcheck0 * tol < Rres)
        return CalcTetraRadiusCenterHiPrecision(index);
    double const a_tol = 1e-8;
    if (std::abs(a) < Rres * Rres * Rres * a_tol)
        return CalcTetraRadiusCenterHiPrecision(index);
    return Rres;
}

double Voronoi3D::CalcTetraRadiusCenterHiPrecision(std::size_t index)
{
    std::array<boost::multiprecision::cpp_dec_float_50, 3> V0;
    V0[0] = del_.points_[del_.tetras_[index].points[0]].x;
    V0[1] = del_.points_[del_.tetras_[index].points[0]].y;
    V0[2] = del_.points_[del_.tetras_[index].points[0]].z;
    std::array<boost::multiprecision::cpp_dec_float_50, 3> V2;
    V2[0] = del_.points_[del_.tetras_[index].points[1]].x;
    V2[1] = del_.points_[del_.tetras_[index].points[1]].y;
    V2[2] = del_.points_[del_.tetras_[index].points[1]].z;
    std::array<boost::multiprecision::cpp_dec_float_50, 3> V3;
    V3[0] = del_.points_[del_.tetras_[index].points[2]].x;
    V3[1] = del_.points_[del_.tetras_[index].points[2]].y;
    V3[2] = del_.points_[del_.tetras_[index].points[2]].z;
    std::array<boost::multiprecision::cpp_dec_float_50, 3> V4;
    V4[0] = del_.points_[del_.tetras_[index].points[3]].x;
    V4[1] = del_.points_[del_.tetras_[index].points[3]].y;
    V4[2] = del_.points_[del_.tetras_[index].points[3]].z;
    V2[0] -= V0[0];
    V2[1] -= V0[1];
    V2[2] -= V0[2];
    V3[0] -= V0[0];
    V3[1] -= V0[1];
    V3[2] -= V0[2];
    V4[0] -= V0[0];
    V4[1] -= V0[1];
    V4[2] -= V0[2];
    std::array<boost::multiprecision::cpp_dec_float_50, 9> mat;
    mat[0] = V2[0];
    mat[1] = V2[1];
    mat[2] = V2[2];
    mat[3] = V3[0];
    mat[4] = V3[1];
    mat[5] = V3[2];
    mat[6] = V4[0];
    mat[7] = V4[1];
    mat[8] = V4[2];
    boost::multiprecision::cpp_dec_float_50 ba = Calc33Det(mat);
    mat[0] = V2[0] * V2[0] + V2[1] * V2[1] + V2[2] * V2[2];
    mat[1] = V2[1];
    mat[2] = V2[2];
    mat[3] = V3[0] * V3[0] + V3[1] * V3[1] + V3[2] * V3[2];
    mat[4] = V3[1];
    mat[5] = V3[2];
    mat[6] = V4[0] * V4[0] + V4[1] * V4[1] + V4[2] * V4[2];
    mat[7] = V4[1];
    mat[8] = V4[2];
    boost::multiprecision::cpp_dec_float_50 bDx = Calc33Det(mat);
    mat[0] = V2[0] * V2[0] + V2[1] * V2[1] + V2[2] * V2[2];
    mat[1] = V2[0];
    mat[2] = V2[2];
    mat[3] = V3[0] * V3[0] + V3[1] * V3[1] + V3[2] * V3[2];
    mat[4] = V3[0];
    mat[5] = V3[2];
    mat[6] = V4[0] * V4[0] + V4[1] * V4[1] + V4[2] * V4[2];
    mat[7] = V4[0];
    mat[8] = V4[2];
    boost::multiprecision::cpp_dec_float_50 bDy = -Calc33Det(mat);
    mat[0] = V2[0] * V2[0] + V2[1] * V2[1] + V2[2] * V2[2];
    mat[1] = V2[0];
    mat[2] = V2[1];
    mat[3] = V3[0] * V3[0] + V3[1] * V3[1] + V3[2] * V3[2];
    mat[4] = V3[0];
    mat[5] = V3[1];
    mat[6] = V4[0] * V4[0] + V4[1] * V4[1] + V4[2] * V4[2];
    mat[7] = V4[0];
    mat[8] = V4[1];
    boost::multiprecision::cpp_dec_float_50 bDz = Calc33Det(mat);
    boost::multiprecision::cpp_dec_float_50 temp = (bDx / (2 * ba) + V0[0]);
    tetra_centers_[index].x = temp.convert_to<double>();
    temp = (bDy / (2 * ba) + V0[1]);
    tetra_centers_[index].y = temp.convert_to<double>();
    temp = (bDz / (2 * ba) + V0[2]);
    tetra_centers_[index].z = temp.convert_to<double>();
    temp = (boost::multiprecision::sqrt(bDx * bDx + bDy * bDy + bDz * bDz) / ba);
    return 0.5 * temp.convert_to<double>();
}

void Voronoi3D::GetTetraCM(std::array<Vector3D, 4> const &points, Vector3D &CM) const
{
    double x = 0, y = 0, z = 0;
    //CM.Set(0, 0, 0);
#ifdef __INTEL_COMPILER
#pragma omp simd reduction(+ \
                                                     : x, y, z)
#endif
    for (std::size_t i = 0; i < 4; i++)
    {
        x += points[i].x;
        y += points[i].y;
        z += points[i].z;
    }
    CM.Set(x, y, z);
    CM *= 0.25;
}

double Voronoi3D::GetTetraVolume(std::array<Vector3D, 4> const &points) const
{
    return std::abs(orient3d(points)) / 6.0;
}

/*
void Voronoi3D::CalcCellCMVolume(std::size_t index)
{
    volume_[index] = 0;
    CM_[index] = Vector3D();
    std::size_t Nfaces = FacesInCell_[index].size();
    std::array<Vector3D, 4> tetra;
    tetra[3] = del_.points_[index];
    Vector3D vtemp;
    for (std::size_t i = 0; i < Nfaces; ++i)
        {
            std::size_t face = FacesInCell_[index][i];
            std::size_t Npoints = PointsInFace_[face].size();
            tetra[0] = tetra_centers_[PointsInFace_[face][0]];
            double fvol = 0;
            for (std::size_t j = 0; j < Npoints - 2; ++j)
	{
	    tetra[1] = tetra_centers_[PointsInFace_[face][j + 1]];
	    tetra[2] = tetra_centers_[PointsInFace_[face][j + 2]];
	    double vol = GetTetraVolume(tetra);
	    fvol += std::abs(vol);
	    GetTetraCM(tetra, vtemp);
	    CM_[index] += std::abs(vol)*vtemp;
	}
            volume_[index] += fvol;
        }
    CM_[index] = CM_[index] / volume_[index];
}
*/

void Voronoi3D::output(std::string const &filename) const
{

    std::ofstream file_handle(filename.c_str(), std::ios::out | std::ios::binary);
    assert(file_handle.is_open());
    binary_write_single_int(static_cast<int>(Norg_), file_handle);

    // Points
    for (std::size_t i = 0; i < Norg_; ++i)
    {
        binary_write_single_double(del_.points_[i].x, file_handle);
        binary_write_single_double(del_.points_[i].y, file_handle);
        binary_write_single_double(del_.points_[i].z, file_handle);
    }

    binary_write_single_int(static_cast<int>(tetra_centers_.size()), file_handle);
    // Face Points
    for (std::size_t i = 0; i < tetra_centers_.size(); ++i)
    {
        binary_write_single_double(tetra_centers_[i].x, file_handle);
        binary_write_single_double(tetra_centers_[i].y, file_handle);
        binary_write_single_double(tetra_centers_[i].z, file_handle);
    }

    // Faces in cell
    for (std::size_t i = 0; i < Norg_; ++i)
    {
        binary_write_single_int(static_cast<int>(FacesInCell_[i].size()), file_handle);
        for (std::size_t j = 0; j < FacesInCell_[i].size(); ++j)
            binary_write_single_int(static_cast<int>(FacesInCell_[i][j]), file_handle);
    }

    // Points in Face
    binary_write_single_int(static_cast<int>(PointsInFace_.size()), file_handle);
    for (std::size_t i = 0; i < PointsInFace_.size(); ++i)
    {
        binary_write_single_int(static_cast<int>(PointsInFace_[i].size()), file_handle);
        for (std::size_t j = 0; j < PointsInFace_[i].size(); ++j)
            binary_write_single_int(static_cast<int>(PointsInFace_[i][j]), file_handle);
    }

    file_handle.close();
}

#ifdef RICH_MPI
void Voronoi3D::output_buildextra(std::string const &filename) const
{
    std::ofstream file_handle(filename.c_str(), std::ios::out | std::ios::binary);
    assert(file_handle.is_open());
    size_t stemp = Norg_;
    binary_write_single_int(static_cast<int>(stemp), file_handle);
    stemp = del_.points_.size();
    binary_write_single_int(static_cast<int>(stemp), file_handle);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Points
    for (std::size_t i = 0; i < stemp; ++i)
    {
        binary_write_single_double(del_.points_[i].x, file_handle);
        binary_write_single_double(del_.points_[i].y, file_handle);
        binary_write_single_double(del_.points_[i].z, file_handle);
    }

    binary_write_single_int(static_cast<int>(duplicatedprocs_.size()), file_handle);
    // Procs
    assert(duplicatedprocs_.size() == Nghost_.size());
    for (size_t i = 0; i < duplicatedprocs_.size(); ++i)
    {
        binary_write_single_int(static_cast<int>(duplicatedprocs_[i]), file_handle);
        binary_write_single_int(static_cast<int>(Nghost_[i].size()), file_handle);
        for (size_t j = 0; j < Nghost_[i].size(); ++j)
            binary_write_single_int(static_cast<int>(Nghost_[i][j]), file_handle);
    }
    file_handle.close();
}
#endif

std::size_t Voronoi3D::GetPointNo(void) const
{
    return Norg_;
}

Vector3D Voronoi3D::GetMeshPoint(std::size_t index) const
{
    return del_.points_[index];
}

double Voronoi3D::GetArea(std::size_t index) const
{
    return area_[index];
}

Vector3D const &Voronoi3D::GetCellCM(std::size_t index) const
{
    return CM_[index];
}

std::size_t Voronoi3D::GetTotalFacesNumber(void) const
{
    return FaceNeighbors_.size();
}

double Voronoi3D::GetWidth(std::size_t index) const
{
    return std::pow(3 * volume_[index] * 0.25 / M_PI, 0.3333333333);
}

double Voronoi3D::GetVolume(std::size_t index) const
{
    return volume_[index];
}

face_vec const &Voronoi3D::GetCellFaces(std::size_t index) const
{
    return FacesInCell_[index];
}

vector<Vector3D> &Voronoi3D::accessMeshPoints(void)
{
    return del_.points_;
}

const vector<Vector3D> &Voronoi3D::getMeshPoints(void) const
{
    return del_.points_;
}

vector<std::size_t> Voronoi3D::GetNeighbors(std::size_t index) const
{
    const size_t N = FacesInCell_[index].size();
    vector<size_t> res(N);
#ifdef __INTEL_COMPILER
#pragma omp simd
#endif
    for (size_t i = 0; i < N; ++i)
    {
        size_t face = FacesInCell_[index][i];
        res[i] = FaceNeighbors_[face].first == index ? FaceNeighbors_[face].second : FaceNeighbors_[face].first;
    }
    return res;
}

void Voronoi3D::GetNeighbors(size_t index, vector<size_t> &res) const
{
    std::size_t N = FacesInCell_[index].size();
    res.resize(N);
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for (std::size_t i = 0; i < N; ++i)
    {
        std::size_t face = FacesInCell_[index][i];
        res[i] = FaceNeighbors_[face].first == index ? FaceNeighbors_[face].second : FaceNeighbors_[face].first;
    }
}

Tessellation3D *Voronoi3D::clone(void) const
{
    return new Voronoi3D(*this);
}

Voronoi3D::Voronoi3D(Voronoi3D const &other) : ll_(other.ll_), ur_(other.ur_), Norg_(other.Norg_), bigtet_(other.bigtet_),
                                                set_temp_(other.set_temp_), stack_temp_(other.stack_temp_), del_(other.del_), PointTetras_(other.PointTetras_), R_(other.R_),
                                                tetra_centers_(other.tetra_centers_), FacesInCell_(other.FacesInCell_), PointsInFace_(other.PointsInFace_),
                                                FaceNeighbors_(other.FaceNeighbors_), CM_(other.CM_), Face_CM_(other.Face_CM_), volume_(other.volume_), area_(other.area_),
                                                duplicated_points_(other.duplicated_points_), sentprocs_(other.sentprocs_), duplicatedprocs_(other.duplicatedprocs_), sentpoints_(other.sentpoints_),
                                                Nghost_(other.Nghost_), self_index_(other.self_index_), temp_points_(std::array<Vector3D, 4>()), temp_points2_(std::array<Vector3D, 5>()), box_faces_(other.box_faces_)
                                                #ifdef RICH_MPI
                                                , envAgent(other.envAgent), initialRadius(other.initialRadius), firstCall(true), pointsManager(PointsManager(this->ll_, this->ur_)), hilbertOrder(NULL_ORDER)
                                                #endif // RICH_MPI
                                                {}

bool Voronoi3D::NearBoundary(std::size_t index) const
{
    std::size_t N = FacesInCell_[index].size();
    for (std::size_t i = 0; i < N; ++i)
    {
        if (BoundaryFace(FacesInCell_[index][i]))
            return true;
    }
    return false;
}

bool Voronoi3D::IsPointOutsideBox(size_t index) const
{
    if(box_faces_.empty())
        return !PointInDomain(ll_, ur_, del_.points_[index]);
    else
        return !PointInPoly(box_faces_, del_.points_[index]);
}

bool Voronoi3D::BoundaryFace(std::size_t index) const
{
    if (FaceNeighbors_[index].first >= Norg_ || FaceNeighbors_[index].second >= Norg_)
    {
#ifdef RICH_MPI
        if(box_faces_.empty())
        {
            if (PointInDomain(ll_, ur_, del_.points_[std::max(FaceNeighbors_[index].first, FaceNeighbors_[index].second)]))
                return false;
            else
                return true;
        }
        else
            if(PointInPoly(box_faces_, del_.points_[std::max(FaceNeighbors_[index].first, FaceNeighbors_[index].second)]))
                return false;
            else
#endif
            return true;
    }
    else
        return false;
}

vector<vector<std::size_t>> &Voronoi3D::GetDuplicatedPoints(void)
{
    return duplicated_points_;
}

vector<vector<std::size_t>> const &Voronoi3D::GetDuplicatedPoints(void) const
{
    return duplicated_points_;
}

std::size_t Voronoi3D::GetTotalPointNumber(void) const
{
    return del_.points_.size();
}

vector<Vector3D> &Voronoi3D::GetAllCM(void)
{
    return CM_;
}

vector<Vector3D> Voronoi3D::GetAllCM(void) const
{
    return CM_;
}

void Voronoi3D::GetNeighborNeighbors(vector<std::size_t> &result, std::size_t point) const
{
    result.clear();
    result.reserve(70);
    vector<std::size_t> neigh = GetNeighbors(point);
    result = neigh;
    std::size_t N = neigh.size();
    std::sort(neigh.begin(), neigh.end());
    vector<std::size_t> temp;
    for (std::size_t i = 0; i < N; ++i)
    {
        if (neigh[i] < Norg_)
        {
            temp = GetNeighbors(neigh[i]);
            result.insert(result.end(), temp.begin(), temp.end());
        }
    }
    std::sort(result.begin(), result.end());
    result = unique(result);
    result = RemoveList(result, neigh);
    RemoveVal(result, point);
}

vector<boost::container::small_vector<size_t, 8>> &Voronoi3D::GetAllPointsInFace(void)
{
    return PointsInFace_;
}

vector<boost::container::small_vector<size_t, 8>> const& Voronoi3D::GetAllPointsInFace(void)const
{
    return PointsInFace_;
}

size_t &Voronoi3D::GetPointNo(void)
{
    return Norg_;
}

std::vector<std::pair<size_t, size_t>> &Voronoi3D::GetAllFaceNeighbors(void)
{
    return FaceNeighbors_;
}

vector<double> &Voronoi3D::GetAllVolumes(void)
{
    return volume_;
}

vector<double> Voronoi3D::GetAllVolumes(void) const
{
    return volume_;
}

Vector3D Voronoi3D::Normal(std::size_t faceindex) const
{
    return del_.points_[FaceNeighbors_[faceindex].second] - del_.points_[FaceNeighbors_[faceindex].first];
}

bool Voronoi3D::IsGhostPoint(std::size_t index) const
{
    return index >= Norg_;
}

Vector3D Voronoi3D::FaceCM(std::size_t index) const
{
    return Face_CM_[index];
}

Vector3D Voronoi3D::CalcFaceVelocity(std::size_t index, Vector3D const &v0, Vector3D const &v1) const
{
    std::size_t p0 = FaceNeighbors_[index].first;
    std::size_t p1 = FaceNeighbors_[index].second;
    Vector3D r0 = GetMeshPoint(p0);
    Vector3D r1 = GetMeshPoint(p1);
    Vector3D r_diff = r1 - r0;
    double abs_r_diff = ScalarProd(r_diff, r_diff);

    Vector3D f = FaceCM(index);
    r1 += r0;
    r1 *= 0.5;
    f -= r1;
    Vector3D delta_w = ScalarProd((v0 - v1), f) * r_diff / abs_r_diff;
#ifdef RICH_DEBUG
    double dw_abs = fastabs(delta_w);
#endif // RICH_DEBUG
    Vector3D w = (v0 + v1) * 0.5;
#ifdef RICH_DEBUG
    //    double w_abs = std::max(fastabs(v0),fastabs(v1));
#endif // RICH_DEBUG
    //if (dw_abs > w_abs)
    //	delta_w *= (1 + (std::atan(dw_abs / w_abs) - 0.25 * M_PI)*2) * (w_abs / dw_abs);
#ifdef RICH_DEBUG
    if (!std::isfinite(dw_abs))
    {
        r0 = GetMeshPoint(p0);
        r1 = GetMeshPoint(p1);
        f = FaceCM(index);
        UniversalError eo("Bad Face velocity");
        eo.addEntry("Face index", index);
        eo.addEntry("Neigh 0", p0);
        eo.addEntry("Neigh 1", p1);
        eo.addEntry("Neigh 0 x", r0.x);
        eo.addEntry("Neigh 0 y", r0.y);
        eo.addEntry("Neigh 0 z", r0.z);
        eo.addEntry("Neigh 0 CMx", CM_[p0].x);
        eo.addEntry("Neigh 0 CMy", CM_[p0].y);
        eo.addEntry("Neigh 0 CMz", CM_[p0].z);
        eo.addEntry("Neigh 1 x", r1.x);
        eo.addEntry("Neigh 1 y", r1.y);
        eo.addEntry("Neigh 1 z", r1.z);
        eo.addEntry("Neigh 1 CMx", CM_[p1].x);
        eo.addEntry("Neigh 1 CMy", CM_[p1].y);
        eo.addEntry("Neigh 1 CMz", CM_[p1].z);
        eo.addEntry("Face CMx", f.x);
        eo.addEntry("Face CMy", f.y);
        eo.addEntry("Face CMz", f.z);
        eo.addEntry("V0x", v0.x);
        eo.addEntry("V0y", v0.y);
        eo.addEntry("V0z", v0.z);
        eo.addEntry("V1x", v1.x);
        eo.addEntry("V1y", v1.y);
        eo.addEntry("V1z", v1.z);
        throw eo;
    }
#endif
    w += delta_w;
    return w;
}

vector<double> &Voronoi3D::GetAllArea(void)
{
    return area_;
}

vector<Vector3D> &Voronoi3D::GetAllFaceCM(void)
{
    return Face_CM_;
}

vector<face_vec> &Voronoi3D::GetAllCellFaces(void)
{
    return FacesInCell_;
}

vector<face_vec> const& Voronoi3D::GetAllCellFaces(void)const
{
    return FacesInCell_;
}

vector<Vector3D> &Voronoi3D::GetFacePoints(void)
{
    return tetra_centers_;
}

vector<Vector3D> const &Voronoi3D::GetFacePoints(void) const
{
    return tetra_centers_;
}

point_vec const &Voronoi3D::GetPointsInFace(std::size_t index) const
{
    return PointsInFace_[index];
}

std::pair<std::size_t, std::size_t> Voronoi3D::GetFaceNeighbors(std::size_t face_index) const
{
    return std::pair<std::size_t, std::size_t>(FaceNeighbors_[face_index]);
}

vector<int> Voronoi3D::GetDuplicatedProcs(void) const
{
    return duplicatedprocs_;
}

vector<int> Voronoi3D::GetSentProcs(void) const
{
    return sentprocs_;
}

vector<vector<std::size_t>> const &Voronoi3D::GetSentPoints(void) const
{
    return sentpoints_;
}

vector<std::size_t> const &Voronoi3D::GetSelfIndex(void) const
{
    return self_index_;
}

vector<int> &Voronoi3D::GetSentProcs(void)
{
    return sentprocs_;
}

vector<vector<std::size_t>> &Voronoi3D::GetSentPoints(void)
{
    return sentpoints_;
}

vector<std::size_t> &Voronoi3D::GetSelfIndex(void)
{
    return self_index_;
}

void Voronoi3D::SetBox(Vector3D const &ll, Vector3D const &ur)
{
    ll_ = ll;
    ur_ = ur;
    #ifdef RICH_MPI
        this->pointsManager = PointsManager(this->ll_, this->ur_);
        this->radiuses.clear();
        this->firstCall = false;
        delete this->envAgent;
        this->envAgent = nullptr;
    #endif // RICH_MPI
}
