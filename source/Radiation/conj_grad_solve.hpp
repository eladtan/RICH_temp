#ifndef CG_HPP
#define CG_HPP 1

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#ifdef RICH_MPI
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#endif
#include <boost/container/small_vector.hpp>
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "misc/utils.hpp"

namespace CG
{
    using vec = boost::container::small_vector<double, 20>;         // vector
    using vec_size_t = boost::container::small_vector<size_t, 20>;         // vector
    using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)
    using size_t_mat = std::vector<vec_size_t>;
    double constexpr speed_of_light = 2.99792458e10;
    size_t constexpr max_size_t = std::numeric_limits<size_t>::max();
    double constexpr stefan_boltzman = 5.670374e-5;
    double constexpr radiation_constant = 4 * stefan_boltzman / speed_of_light;
    double constexpr boltzmann_constant = 1.380649e-16;
    double constexpr electron_mass = 9.1093837015e-28;
    double constexpr max_coupling_strength = 1e2;
    double constexpr compton_optical_depth_turn_off = 10;

    struct BiCGSTABWorkspace
    {
        mat A;
        size_t_mat A_indeces;
        std::vector<double> b;
        std::vector<double> sub_x;
        std::vector<size_t> A_row_ptr;
        std::vector<size_t> A_col_idx;
        std::vector<double> A_values;
        std::vector<double> M;
        std::vector<double> r_old;
        std::vector<double> sub_a_times_p;
        std::vector<double> sub_r;
        std::vector<double> sub_p;
        std::vector<double> sub_r0;
        std::vector<double> y;
        std::vector<double> z;
        std::vector<double> v;
        std::vector<double> h;
        std::vector<double> s;
        std::vector<double> t;
        std::vector<double> scratch_rescale1;
        std::vector<double> scratch_rescale2;
        std::vector<double> old_x;
    };

    //! \brief Class that build the data for the solution of the linear system A*x=b
    class MatrixBuilder
    {
         public:
            MatrixBuilder(std::vector<std::string> const zero_cells = std::vector<std::string> ()) : zero_cells_(zero_cells){}
        /*!
            \brief Builds the initial conditions for the CG method to solve A*x=b
            \param tess The tessellation
            \param A The A matrix to build
            \param A_indeces The indeces of the values in A, this is needed since A is sparse
            \param cells The computational cells
            \param dt The time step
            \param b The b vector to calculate
            \param x0 The initial solution guess
            \param current_time The time
        */

        virtual ~MatrixBuilder() = default;

        /**
         * @brief Virtual function to get the length scale used in the CG method.
         * 
         * This function returns the length scale used in the CG method. The default implementation returns 1.0.
         * Derived classes may override this function to provide their own length scale.
         * 
         * @return A double representing the length scale.
         */
        virtual double GetLengthScale() const {return 1.0;}

        virtual void BuildMatrix(Tessellation3D const& tess, mat& A, size_t_mat& A_indeces, std::vector<ComputationalCell3D> const& cells,
            double const dt, std::vector<double>& b, std::vector<double>& x0, double const current_time) const = 0;
        /*!
        \brief This method does post processing after the CG has finished (e.g. update the thermal energy)
        \param tess The tesselation
        \param extensives The extensives
        \param dt The time step
        \param cells The primitive variables
        \param CG_result The result from the CG
        */
        virtual void PostCG(Tessellation3D const& tess, std::vector<Conserved3D>& extensives, double const dt, std::vector<ComputationalCell3D>& cells,
            std::vector<double>const& CG_result, std::vector<double> const&  full_CG_result)const = 0;

        virtual void PrintDebugData(size_t const index) const {;}
        
        std::vector<std::string> const zero_cells_;
    };

    //! The fastest implementation of conjugate gradient algorithm, using data-based parallelism only
    std::vector<double> conj_grad_solver(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution);
/**
 * @brief Performs the BiCGSTAB (Biconjugate Gradient Stabilized) method to solve the linear system A*x=b.
 * 
 * @param tolerance The tolerance for the solution. The method stops when the relative residual norm is less than or equal to this value.
 * @param total_iters Reference to an integer that will store the total number of iterations performed by the method.
 * @param tess The 3D tessellation used to discretize the problem.
 * @param cells The computational cells associated with the tessellation.
 * @param dt The time step for the problem.
 * @param matrix_builder A reference to an object that implements the MatrixBuilder interface, used to build the A matrix and b vector.
 * @param time The current time of the simulation.
 * @param sub_x_solution Reference to a vector that will store the solution x.
 * @param good_end Reference to a boolean that will be set to true if the method successfully converged to the desired tolerance.
 * 
 * @return A vector containing the solution x.
 */
    std::vector<double> BiCGSTAB(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution,
        bool &good_end);

    std::vector<double> &BiCGSTAB(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution,
        bool &good_end, BiCGSTABWorkspace &workspace);

    double mpi_dot_product(const std::vector<double> &sub_u, const std::vector<double> &sub_v);

    double mpi_dot_product2(const std::vector<double> &sub_u, const std::vector<double> &sub_v);
}

#endif
