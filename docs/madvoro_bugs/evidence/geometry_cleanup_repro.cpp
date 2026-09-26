// Test-only verbatim extraction of CleanSameLine and a one-line candidate fix. Production source is unchanged.
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <iomanip>
#include "examples/Vector3D.hpp"
#include "types.hpp"
#include "exception/MadVoroException.hpp"
#include "elementary/PointOps.hpp"
using namespace MadVoro::fallback;
using MadVoro::point_vec;
using std::vector;
template <typename T> using face_scratch=boost::container::small_vector<T,128>;
    template <typename PointT>
    bool CleanSameLine(boost::container::small_vector<size_t, 8> &indeces, vector<PointT> const& face_points, face_scratch<double> &area_vec_temp)
    {
        point_vec old;
        size_t const N = indeces.size();
        area_vec_temp.resize(N);
        double const small_fraction = 1e-14;
        // double const medium_fraction = 3e-1;
        // Find correct normal
        PointT good_normal;
        for(size_t i = 0; i < N; ++i)
        {
            area_vec_temp[i] = fastabs(CrossProduct(face_points[indeces[i]] - face_points[indeces[(N + i - 1) % N]], face_points[indeces[(i + 1) % N]] 
            - face_points[indeces[(N + i - 1) % N]]));
            old.push_back(indeces[i]);
        }

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

        double const area_scale = area_vec_temp[max_index];
        good_normal = CrossProduct(face_points[indeces[max_index]] - face_points[indeces[(N + max_index - 1) % N]], face_points[indeces[(max_index + 1) % N]] - face_points[indeces[(N + max_index - 1) % N]]);
        good_normal *= 1.0 / fastabs(good_normal);

        size_t Nindeces = indeces.size();
        for(size_t i = 0; i < Nindeces; ++i)
        {
            PointT normal_temp = CrossProduct(face_points[indeces[i]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]], face_points[indeces[(i + 1) % Nindeces]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]]);
            double const area = fastabs(normal_temp);
            normal_temp *= 1.0 / (100 * std::numeric_limits<double>::min() + area);
            if((area < area_scale * small_fraction) || (ScalarProd(normal_temp, good_normal) < 0.9999))
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
                PointT normal_temp = CrossProduct(face_points[indeces[i]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]], face_points[indeces[(i + 1) % Nindeces]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]]);
                double const area = fastabs(normal_temp);
                normal_temp *= 1.0 / area;
                if((area < area_scale * small_fraction) || ScalarProd(normal_temp, good_normal) < 0.9999)
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
            Nindeces = N;
            indeces = old;
            MadVoro::Exception::MadVoroException eo("Bad CleanSameLine");
            eo.addEntry("N", N);
            eo.addEntry("good normal x", good_normal.x);
            eo.addEntry("good normal y", good_normal.y);
            eo.addEntry("good normal z", good_normal.z);
            for(size_t i = 0; i < N; ++i)
            {
                eo.addEntry("index", old[i]);
                eo.addEntry("area_vec_temp", area_vec_temp[i]);
                eo.addEntry("point " + std::to_string(i) + " x", face_points[indeces[i]].x);
                eo.addEntry("point " + std::to_string(i) + " y", face_points[indeces[i]].y);
                eo.addEntry("point " + std::to_string(i) + " z", face_points[indeces[i]].z);
                PointT normal_temp = CrossProduct(face_points[indeces[i]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]], face_points[indeces[(i + 1) % Nindeces]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]]);
                normal_temp *= (1.0 / abs(normal_temp));
                eo.addEntry("normal " + std::to_string(i) + " x", normal_temp.x);
                eo.addEntry("normal " + std::to_string(i) + " y", normal_temp.y);
                eo.addEntry("normal " + std::to_string(i) + " z", normal_temp.z);
                eo.addEntry("dot", ScalarProd(good_normal, normal_temp));
            }
            throw eo;
        }

        return true;
    }

    template <typename PointT>
    bool CleanSameLineFixed(boost::container::small_vector<size_t, 8> &indeces, vector<PointT> const& face_points, face_scratch<double> &area_vec_temp)
    {
        point_vec old;
        size_t const N = indeces.size();
        area_vec_temp.resize(N);
        double const small_fraction = 1e-14;
        // double const medium_fraction = 3e-1;
        // Find correct normal
        PointT good_normal;
        for(size_t i = 0; i < N; ++i)
        {
            area_vec_temp[i] = fastabs(CrossProduct(face_points[indeces[i]] - face_points[indeces[(N + i - 1) % N]], face_points[indeces[(i + 1) % N]] 
            - face_points[indeces[(N + i - 1) % N]]));
            old.push_back(indeces[i]);
        }

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

        double const area_scale = area_vec_temp[max_index];
        good_normal = CrossProduct(face_points[indeces[max_index]] - face_points[indeces[(N + max_index - 1) % N]], face_points[indeces[(max_index + 1) % N]] - face_points[indeces[(N + max_index - 1) % N]]);
        good_normal *= 1.0 / fastabs(good_normal);

        size_t Nindeces = indeces.size();
        for(size_t i = 0; i < Nindeces; ++i)
        {
            PointT normal_temp = CrossProduct(face_points[indeces[i]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]], face_points[indeces[(i + 1) % Nindeces]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]]);
            double const area = fastabs(normal_temp);
            normal_temp *= 1.0 / (100 * std::numeric_limits<double>::min() + area);
            if((area < area_scale * small_fraction) || (ScalarProd(normal_temp, good_normal) < 0.9999))
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
            Nindeces = indeces.size();
            max_index = second_max_index;
            good_normal = CrossProduct(face_points[indeces[max_index]] - face_points[indeces[(N + max_index - 1) % N]], face_points[indeces[(max_index + 1) % N]] - face_points[indeces[(N + max_index - 1) % N]]);
            good_normal *= 1.0 / fastabs(good_normal);

            for(size_t i = 0; i < Nindeces; i++)
            {
                PointT normal_temp = CrossProduct(face_points[indeces[i]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]], face_points[indeces[(i + 1) % Nindeces]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]]);
                double const area = fastabs(normal_temp);
                normal_temp *= 1.0 / area;
                if((area < area_scale * small_fraction) || ScalarProd(normal_temp, good_normal) < 0.9999)
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
            Nindeces = N;
            indeces = old;
            MadVoro::Exception::MadVoroException eo("Bad CleanSameLine");
            eo.addEntry("N", N);
            eo.addEntry("good normal x", good_normal.x);
            eo.addEntry("good normal y", good_normal.y);
            eo.addEntry("good normal z", good_normal.z);
            for(size_t i = 0; i < N; ++i)
            {
                eo.addEntry("index", old[i]);
                eo.addEntry("area_vec_temp", area_vec_temp[i]);
                eo.addEntry("point " + std::to_string(i) + " x", face_points[indeces[i]].x);
                eo.addEntry("point " + std::to_string(i) + " y", face_points[indeces[i]].y);
                eo.addEntry("point " + std::to_string(i) + " z", face_points[indeces[i]].z);
                PointT normal_temp = CrossProduct(face_points[indeces[i]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]], face_points[indeces[(i + 1) % Nindeces]] - face_points[indeces[(Nindeces + i - 1) % Nindeces]]);
                normal_temp *= (1.0 / abs(normal_temp));
                eo.addEntry("normal " + std::to_string(i) + " x", normal_temp.x);
                eo.addEntry("normal " + std::to_string(i) + " y", normal_temp.y);
                eo.addEntry("normal " + std::to_string(i) + " z", normal_temp.z);
                eo.addEntry("dot", ScalarProd(good_normal, normal_temp));
            }
            throw eo;
        }

        return true;
    }


int main(){std::mt19937_64 rng(7788); std::uniform_real_distribution<double> dist(0.02,1.0); for(size_t trial=0;trial<100000;trial++){std::vector<Vector3D> pts; for(size_t i=0;i<6;i++){double a=6.283185307179586*i/6;double r=dist(rng);pts.emplace_back(r*cos(a),r*sin(a),0);}
point_vec x,y;for(size_t i=0;i<pts.size();i++){x.push_back(i);y.push_back(i);} face_scratch<double>s1,s2;bool ok1=true,ok2=true;try{CleanSameLine(x,pts,s1);}catch(...){ok1=false;}try{CleanSameLineFixed(y,pts,s2);}catch(...){ok2=false;}
if(!ok1&&ok2){std::cout<<"trial="<<trial<<" original=throw fixed_vertices="<<y.size()<<"\n"<<std::setprecision(17);for(auto&p:pts)std::cout<<p.x<<","<<p.y<<","<<p.z<<"\n";return 0;}}
std::cout<<"No distinguishing polygon found\n";return 1;}
