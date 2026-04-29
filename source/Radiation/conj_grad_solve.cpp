#include "conj_grad_solve.hpp"
#include <limits>
#include <vectorclass.h>
#include "boost/math/special_functions/pow.hpp"
#include "misc/memory_profile.hpp"

using boost::math::pow;

namespace CG
{
    static constexpr bool use_crs_matvec = true;

    // Matrix times vector
    void mat_times_vec(const mat &sub_A_values, const size_t_mat &sub_A_indices, const std::vector<double> &v, 
        std::vector<double> &result)
    {

        // NOTE: when using MPI with > 1 proc, A will be only a sub-matrix (a subset of rows) of the full matrix
        // since we are 1D decomposing the matrix by rows

        size_t const sub_num_rows = sub_A_values.size();
        if(sub_num_rows == 0) return;

        size_t const sub_num_cols = sub_A_values[0].size();
        result.resize(sub_num_rows, 0);
        double dot_prod;
        for (size_t i = 0; i < sub_num_rows; i++) {
            dot_prod = 0;  // rezero the dot_prod buffer. we need this buffer so we can make it private to the thread to avoid race conditions.
            for (size_t j = 0; j < sub_num_cols; j++) {
                if(sub_A_indices[i][j] == max_size_t)
                    break;
                dot_prod += sub_A_values[i][j] * v[sub_A_indices[i][j]]; 
            }
            result[i] = dot_prod;
        }
    }

    void build_crs(const mat &sub_A_values, const size_t_mat &sub_A_indices,
        std::vector<size_t> &row_ptr, std::vector<size_t> &col_idx, std::vector<double> &values)
    {
        size_t const sub_num_rows = sub_A_values.size();
        row_ptr.assign(sub_num_rows + 1, 0);
        col_idx.clear();
        values.clear();
        col_idx.reserve(sub_num_rows);
        values.reserve(sub_num_rows);
        for (size_t i = 0; i < sub_num_rows; ++i) {
            for (size_t j = 0; j < sub_A_values[i].size(); ++j) {
                if (sub_A_indices[i][j] == max_size_t)
                    break;
                col_idx.push_back(sub_A_indices[i][j]);
                values.push_back(sub_A_values[i][j]);
            }
            row_ptr[i + 1] = col_idx.size();
        }
    }

    void mat_times_vec_crs(const std::vector<size_t> &row_ptr, const std::vector<size_t> &col_idx,
        const std::vector<double> &values, const std::vector<double> &v, std::vector<double> &result)
    {
        size_t const sub_num_rows = row_ptr.size() > 0 ? row_ptr.size() - 1 : 0;
        if(sub_num_rows == 0) return;
        result.resize(sub_num_rows, 0);
        const double* __restrict__ val_ptr = values.data();
        const size_t* __restrict__ col_ptr = col_idx.data();
        const double* __restrict__ v_ptr = v.data();
        double* __restrict__ res_ptr = result.data();
        const size_t* __restrict__ rp = row_ptr.data();
        for (size_t i = 0; i < sub_num_rows; ++i) {
            double dot_prod = 0.0;
            for (size_t k = rp[i]; k < rp[i + 1]; ++k)
                dot_prod += val_ptr[k] * v_ptr[col_ptr[k]];
            res_ptr[i] = dot_prod;
        }
    }

    std::vector<double> vector_rescale(std::vector<double> const& a, std::vector<double> const& b)
    {
        size_t const N = a.size(); 
        if(a.size() != b.size())
            throw UniversalError("Sizes do not match in vector_rescale");
        std::vector<double> res(N);
        for(size_t i = 0; i < N; ++i)
            res[i] = a[i] * b[i];
        return res;
    }

    void vector_rescale(std::vector<double> const& a, std::vector<double> const& b,
        std::vector<double> &result)
    {
        size_t const N = a.size(); 
        if(a.size() != b.size())
            throw UniversalError("Sizes do not match in vector_rescale");
        result.resize(N);
        const double* a_ptr = a.data();
        const double* __restrict__ b_ptr = b.data();
        double* res_ptr = result.data();
        for(size_t i = 0; i < N; ++i)
            res_ptr[i] = a_ptr[i] * b_ptr[i];
    }

    // Linear combination of vectors; safe when result aliases u or v
    void vec_lin_combo(double a, const std::vector<double> &u, double b, const std::vector<double> &v, 
        std::vector<double> &result)
    {
        if(u.size() != v.size())
            throw UniversalError("Unequal vector sizes in vec_lin_combo");
        size_t n = u.size();
        result.resize(n);
        for (size_t j = 0; j < n; j++)
            result[j] = a * u[j] + b * v[j];
    }

    double mpi_dot_product(const std::vector<double> &sub_u, const std::vector<double> &sub_v) // need to pass it the buffer where to keep the result
    {
        if(sub_u.size() != sub_v.size())
            throw UniversalError("Unequal vector sizes in vec_lin_combo");
        size_t length = sub_u.size();

        double sub_prod = 0;
        size_t length4 = length / 4;
        size_t loop_number = length4 * 4;
        for(size_t i = 0; i < loop_number; i += 4)
        {
            Vec4d _u(sub_u[i], sub_u[i+1], sub_u[i+2], sub_u[i+3]);
            Vec4d _v(sub_v[i], sub_v[i+1], sub_v[i+2], sub_v[i+3]);
            Vec4d _uv = _u * _v;
            sub_prod += ((_uv[0] + _uv[1]) + (_uv[2] + _uv[3]));
        }
        for(size_t i = loop_number; i < length; i++)
            sub_prod += sub_u[i] * sub_v[i];
#ifdef RICH_MPI
        // do a reduction over sub_prod to get the total dot product
        MPI_Allreduce(MPI_IN_PLACE, &sub_prod, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        return sub_prod;
    }


    // performs a reduction over the sub-vectors which are passed to it... All_Reduce broadcasts the value to all procs
    double mpi_dot_product2(const std::vector<double> &sub_u, const std::vector<double> &sub_v) // need to pass it the buffer where to keep the result
    {
        if(sub_u.size() != sub_v.size())
            throw UniversalError("Unequal vector sizes in vec_lin_combo");
        size_t length = sub_u.size();
        const double* __restrict__ u_ptr = sub_u.data();
        const double* __restrict__ v_ptr = sub_v.data();

        double sub_prod = 0.0;
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER) || defined(__GNUC__)
#pragma omp simd reduction(+:sub_prod)
#endif
        for (size_t i = 0; i < length; i++) {
            sub_prod += u_ptr[i] * v_ptr[i];
        }
#ifdef RICH_MPI
        // do a reduction over sub_prod to get the total dot product
        MPI_Allreduce(MPI_IN_PLACE, &sub_prod, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        return sub_prod;
    }
    
    void finalize_conjugate_gradient(size_t slice, size_t max_loc0, size_t max_loc1, int rank, size_t i,
        double error, double max_data_0_val, double max_data_1_val, double max_data_2_val,
        int max_data_0_id, int max_data_1_id, std::vector<double>& sub_x, std::vector<double>& sub_x_solution,
        std::vector<ComputationalCell3D> const& cells, Tessellation3D const& tess, int &total_iters,
        const std::vector<size_t>& A_row_ptr, const std::vector<size_t>& A_col_idx, const std::vector<double>& A_values,
        std::vector<double> const& b, size_t Nlocal, double lengthscale,
        std::vector<double>& sub_a_times_p, std::vector<double>& sub_r)
    {
        max_loc0 /= slice;
        max_loc1 /= slice;
        if (cells.size() > max_loc0 and cells.size() > max_loc1){
            if(rank == 0)
                std:: cout << "Converged at iter = " << i <<" error "<<error<<" negative value "<<max_data_2_val<<std::endl;
            if(rank == max_data_0_id)
                std::cout<<"Max0 "<<max_data_0_val<<" cell ID "<<cells[max_loc0].ID<<" density "<<cells[max_loc0].density<<" temperature "<<cells[max_loc0].temperature<<" Er "<<cells[max_loc0].Erad * cells[max_loc0].density
                <<" X "<<tess.GetMeshPoint(max_loc0)<<" sub_x "<<sub_x[max_loc0]<<std::endl; 
            if(rank == max_data_1_id)
                std::cout<<"Max1 "<<max_data_1_val<<" cell ID "<<cells[max_loc1].ID<<" density "<<cells[max_loc1].density<<" temperature "<<cells[max_loc1].temperature<<" Er "<<cells[max_loc1].Erad * cells[max_loc1].density
                <<" X "<<tess.GetMeshPoint(max_loc1)<<" sub_x "<<sub_x[max_loc1]<<std::endl; 
        }
        total_iters = i;
#ifdef RICH_MPI
        MPI_exchange_data(tess, sub_x, true, slice);
#endif
        mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, sub_x, sub_a_times_p);
        sub_x.resize(Nlocal);
        vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);
        sub_x_solution.resize(Nlocal);
        double sum_dx = 0, sum_dx_sign = 0, negative_x = 0;
        std::vector<double> dx_vec(slice, 0);
        for(size_t k = 0; k < Nlocal; ++k)
        {
            // std::cout<<"subx["<<k<<"] "<<sub_x[k]<<std::endl;
            // sub_x_solution[k] = sub_x[k];
            double const cell_volume = tess.GetVolume(k / slice) * pow<3>(lengthscale);
            sub_x_solution[k] = sub_x[k] + sub_r[k] / cell_volume;
            if(sub_x_solution[k] < 0)
            {
                negative_x -= cell_volume * sub_x_solution[k];
                sum_dx += cell_volume * (std::abs(sub_x[k]) * 0.1 - sub_x_solution[k]);
                sum_dx_sign += cell_volume * (std::abs(sub_x[k]) * 0.1 - sub_x_solution[k]);
                dx_vec[k%slice] += cell_volume * (std::abs(sub_x[k]) * 0.1 - sub_x_solution[k]);
                // if(cell_volume * (std::abs(sub_x[k]) * 0.1 - sub_x_solution[k])>1e37)
                //     std::cout<<"Large Erad change sub_x[k] "<<sub_x[k]<<" "<<cells[k/slice]<<std::endl;
                sub_x_solution[k] = std::abs(sub_x[k]) * 0.1;
            }
            else
            {
                if(sub_x_solution[k] < sub_x[k] * 0.5 && sub_x_solution[k] >= 0 && sub_x[k] > 0)
                {
                    sum_dx += cell_volume * (std::abs(sub_x[k]) * 0.5 - sub_x_solution[k]);
                    sum_dx_sign += cell_volume * (std::abs(sub_x[k]) * 0.5 - sub_x_solution[k]);
                    dx_vec[k%slice] += cell_volume * (std::abs(sub_x[k]) * 0.5 - sub_x_solution[k]);
                    sub_x_solution[k] = std::abs(sub_x[k]) * 0.5;
                }
                if(sub_x_solution[k] > std::abs(sub_x[k]) * 2 && sub_x[k] > 0)
                {
                    sum_dx += cell_volume * std::abs(sub_x[k] * 2 - sub_x_solution[k]);
                    sum_dx_sign -= cell_volume * std::abs(sub_x[k] * 2 - sub_x_solution[k]);
                    dx_vec[k%slice] -= cell_volume * std::abs(sub_x[k] * 2 - sub_x_solution[k]);
                    sub_x_solution[k] = sub_x[k] * 2;
                }
            }
        }
        dx_vec.push_back(sum_dx);
        dx_vec.push_back(sum_dx_sign);
        dx_vec.push_back(negative_x);
#ifdef RICH_MPI
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : dx_vec.data(), dx_vec.data(), dx_vec.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
        if(rank == 0)
        {
            std::cout << "Energy change due to sub_x_solution " << dx_vec[slice] <<" with sign "<<dx_vec[slice + 1]<<" negative_x "<<dx_vec.back()<<std::endl;
            // for(size_t i = 0; i < slice; ++i) std::cout << "energy change group " << i << " = " << dx_vec[i] << std::endl;
        }
    }
    
    void build_M(const mat &sub_A_values, const size_t_mat &sub_A_indices, std::vector<double> &M)
    {
        size_t const sub_num_rows = sub_A_values.size();
        if(sub_num_rows == 0) return;
        int rank = 0;

        size_t const sub_num_cols = sub_A_values[0].size();
        M.resize(sub_num_rows);
        for (size_t i = 0; i < sub_num_rows; i++) {
            for (size_t j = 0; j < sub_num_cols; j++) {
                if(sub_A_indices[i][j] == max_size_t)
                    break;
                if(sub_A_indices[i][j] == i)
                {
                    M[i] = 1.0 / sub_A_values[i][j];
                    // M[i] = 1.0;
                    break;
                }
            }
        }
    }

    void build_M_crs(const std::vector<size_t> &row_ptr, const std::vector<size_t> &col_idx,
        const std::vector<double> &values, std::vector<double> &M)
    {
        size_t const sub_num_rows = row_ptr.size() > 0 ? row_ptr.size() - 1 : 0;
        if(sub_num_rows == 0) return;
        M.resize(sub_num_rows);
        for (size_t i = 0; i < sub_num_rows; ++i) {
            for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                if (col_idx[k] == i) {
                    M[i] = 1.0 / values[k];
                    break;
                }
            }
        }
    }

    static bool abs_compare(double a, double b)
    {
        return (std::abs(a) < std::abs(b));
    }

std::vector<double> conj_grad_solver(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution)  //total_iters is to store # of iters in it
    {
        size_t Nlocal = tess.GetPointNo();
        
        // NOTE: when using MPI with > 1 proc, A will be only a sub-matrix (a subset of rows) of the full matrix
        // since we are 1D decomposing the matrix by rows
        // b will be the full vector

        int nprocs = 1, rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
        MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    #endif
        int const max_iter = 10000;

        mat A;
        size_t_mat A_indeces;
        std::vector<double> b;
        std::vector<double> sub_x; // this is for the initial guess
        matrix_builder.BuildMatrix(tess, A, A_indeces, cells, dt, b, sub_x, time);
        std::vector<size_t> A_row_ptr, A_col_idx;
        std::vector<double> A_values;
        build_crs(A, A_indeces, A_row_ptr, A_col_idx, A_values);
        std::vector<double> M; // The preconditioner
        if(use_crs_matvec)
            build_M_crs(A_row_ptr, A_col_idx, A_values, M);
        else
            build_M(A, A_indeces, M);
        std::vector<double> r_old, sub_a_times_p;
        std::vector<double> sub_r;
#ifdef RICH_MPI
        MPI_exchange_data(tess, sub_x, true);
#endif
        if(use_crs_matvec)
            mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, sub_x, sub_a_times_p);
        else
            mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
        // Find maximum value of A, this is used for normalization of the error
        double maxA[2] = {0, 0};
        size_t const Na = A.size();
        for(size_t i = 0; i < Na ; ++i)
        {
            maxA[0] = std::max(maxA[0], std::abs(sub_x[i]));
            maxA[1] = std::max(maxA[1], std::abs(b[i]));
        }       
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &maxA, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);    
        std::vector<double> sub_p(sub_r);
        sub_p.resize(Nlocal);
        sub_x.resize(Nlocal);
        vector_rescale(sub_p, M, sub_p);
        std::vector<double> result2(Nlocal, 0), result3, p(sub_p), old_result2(Nlocal, 0);
        std::vector<double> old_x = sub_x;
        size_t Ntotal = Nlocal;
#ifdef RICH_MPI
        MPI_exchange_data(tess, p, true);
        MPI_Allreduce(&Nlocal, &Ntotal, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        double sub_r_sqrd = mpi_dot_product(sub_r, sub_p);
        double const delta_init = sub_r_sqrd;
        // double sub_r_sqrd_convergence = mpi_dot_product(sub_r, sub_r);
        // if(rank == 0)
        //     std::cout<<"CG init delta "<<delta_init<<std::endl;
        double sub_r_sqrd_old = 0, sub_p_by_ap = 0, alpha = 0, beta = 0;
        bool good_end = false;
        struct
        {
            double val;
            int mpi_id;
        }max_data[3];
        max_data[0].mpi_id = rank;
        max_data[1].mpi_id = rank;
        max_data[2].mpi_id = rank;
        
        size_t max_loc0 = 0, max_loc1 = 0, max_loc2 = 0;
        // Main Conjugate Gradient loop
        // this loop must be serial b/c CG is an iterative method
        for (int i = 0; i < max_iter; i++) {
            // note: make sure matrix is big enough for the number of processors you are using!
            max_data[2].mpi_id = rank;
            max_data[0].mpi_id = rank;
            r_old = sub_r;                 // Store previous residual
            sub_r_sqrd_old = sub_r_sqrd;  // save a recalculation of r_old^2 later

            if(use_crs_matvec)
                mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, p, sub_a_times_p);  //split up with MPI and then finer parallelize with openmp
            else
                mat_times_vec(A, A_indeces, p, sub_a_times_p);

            sub_p_by_ap = mpi_dot_product(sub_p, sub_a_times_p);

            alpha = sub_r_sqrd / (sub_p_by_ap + std::numeric_limits<double>::min() * 100);         

            // Next estimate of solution
            vec_lin_combo(1.0, sub_x, alpha, sub_p, sub_x);
            if(i > 1 && i % 50 == 0)
            {
#ifdef RICH_MPI
                MPI_exchange_data(tess, sub_x, true);
#endif
                if(use_crs_matvec)
                    mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, sub_x, sub_a_times_p);
                else
                    mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
                sub_x.resize(Nlocal);
                vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);    
            }
            else
                vec_lin_combo(1.0, sub_r, -alpha, sub_a_times_p, sub_r);

            max_data[0].val = 0;
            max_data[1].val = 0;
            max_data[2].val = 0;
            for(size_t j = 0; j < Nlocal; ++j)
            {
                double const local_scale = std::abs(b[j]);
                if(std::abs(sub_r[j]) > max_data[1].val * (std::abs(A[j][0] * (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + maxA[0] * 1e-9))))
                {
                    max_data[1].val = std::abs(sub_r[j]) / (std::abs(A[j][0] * (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + maxA[0] * 1e-9)));
                    max_loc1 = j;
                }
                if(std::abs(sub_x[j] - old_x[j]) > max_data[0].val * (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + maxA[0] * 1e-9))
                {
                     max_data[0].val = std::abs(sub_x[j] - old_x[j]) / (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + maxA[0] * 1e-9);
                     max_loc0 = j;
                }
                if(sub_x[j] < 0)
                {
                    max_loc2 = j;
                    max_data[2].val = 1;
                }
            }

#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, max_data, 3, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
#endif
            old_result2 = result2;
            vector_rescale(sub_r, M, result2);
            sub_r_sqrd = mpi_dot_product(sub_r, result2);
            // recall that we can't have a 'break' within an openmp parallel region, so end it here then all threads are merged, and the convergence is checked
            // Convergence test
            if (sub_r_sqrd < delta_init * tolerance//std::sqrt(sub_r_sqrd_convergence / Ntotal) < tolerance * maxA [1]
                && max_data[1].val < 1e-5 && max_data[0].val < 1e-5 && (i > 250 || max_data[2].val < 0.5)) { // norm is just sqrt(dot product so don't need to use a separate norm fnc) // vector norm needs to use a all reduce!
                if(rank == 0)
                    std:: cout << "Converged at iter = " << i <<" delta "<<sub_r_sqrd<<" negative value "<<max_data[2].val<<std::endl;
                if(rank == max_data[0].mpi_id)
                    std::cout<<"Max0 "<<max_data[0].val<<" cell ID "<<cells[max_loc0].ID<<" density "<<cells[max_loc0].density<<" temperature "<<cells[max_loc0].temperature<<" Er "<<cells[max_loc0].Erad * cells[max_loc0].density
                    <<" X "<<tess.GetMeshPoint(max_loc0).x<<" Y "<<tess.GetMeshPoint(max_loc0).y<<" Z "<<tess.GetMeshPoint(max_loc0).z<<std::endl; 
                if(rank == max_data[1].mpi_id)
                    std::cout<<"Max1 "<<max_data[1].val<<" cell ID "<<cells[max_loc1].ID<<" density "<<cells[max_loc1].density<<" temperature "<<cells[max_loc1].temperature<<" Er "<<cells[max_loc1].Erad * cells[max_loc1].density
                    <<" X "<<tess.GetMeshPoint(max_loc1).x<<" Y "<<tess.GetMeshPoint(max_loc1).y<<" Z "<<tess.GetMeshPoint(max_loc1).z<<std::endl; 
                total_iters = i;
                good_end = true;
#ifdef RICH_MPI
                MPI_exchange_data(tess, sub_x, true);
#endif
                sub_x_solution = sub_x;
                if(use_crs_matvec)
                    mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, sub_x, sub_a_times_p);
                else
                    mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
                sub_x.resize(Nlocal);
                vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);
                break;
            }
            old_x = sub_x;
            vec_lin_combo(1.0, result2, -1.0, old_result2, result3);   
            double const  Polak_Ribiere = mpi_dot_product(sub_r, result3);
            beta = std::max(0.0, Polak_Ribiere / sub_r_sqrd_old);       
            
            vec_lin_combo(1.0, result2, beta, sub_p, sub_p);
            p = sub_p;
    #ifdef RICH_MPI
            MPI_exchange_data(tess, p, true);
    #endif
        }
        if(not good_end)
        {
            total_iters = max_iter;
            std::vector<double> volumes = tess.GetAllVolumes();
#ifdef RICH_MPI
			MPI_exchange_data(tess, volumes, true);
#endif
            if(rank == 0)
	      std:: cout <<"not good end, delta "<<sub_r_sqrd<<" maxdata2 "<<max_data[2].val<<std::endl;
            if(rank == max_data[0].mpi_id)
            {
                std::vector<size_t> neigh = tess.GetNeighbors(max_loc0);
                double min_volume = std::numeric_limits<double>::max(), max_volume = 0;
                for(size_t k = 0; k < neigh.size(); ++k)
                {
                    if(not tess.IsPointOutsideBox(neigh[k]))
                    {
                        min_volume = std::min(min_volume, volumes[neigh[k]]);
                        max_volume = std::max(max_volume, volumes[neigh[k]]);
                    }
                }
                std::cout<<"Max0 "<<max_data[0].val<<" cell ID "<<cells[max_loc0].ID<<" density "<<cells[max_loc0].density<<" temperature "<<cells[max_loc0].temperature<<" Er "<<cells[max_loc0].Erad * cells[max_loc0].density
                <<" X "<<tess.GetMeshPoint(max_loc0).x<<" Y "<<tess.GetMeshPoint(max_loc0).y<<" Z "<<tess.GetMeshPoint(max_loc0).z<<" volume "<<tess.GetVolume(max_loc0)<<" max volume "<<max_volume<<" min volume "<<min_volume<<std::endl; 
            }
            if(rank == max_data[1].mpi_id)
            {
                std::vector<size_t> neigh = tess.GetNeighbors(max_loc1);
                double min_volume = std::numeric_limits<double>::max(), max_volume = 0;
                for(size_t k = 0; k < neigh.size(); ++k)
                {
                    if(not tess.IsPointOutsideBox(neigh[k]))
                    {
                        min_volume = std::min(min_volume, volumes[neigh[k]]);
                        max_volume = std::max(max_volume, volumes[neigh[k]]);
                    }
                }
                std::cout<<"Max1 "<<max_data[1].val<<" cell ID "<<cells[max_loc1].ID<<" density "<<cells[max_loc1].density<<" temperature "<<cells[max_loc1].temperature<<" Er "<<cells[max_loc1].Erad * cells[max_loc1].density
                <<" X "<<tess.GetMeshPoint(max_loc1).x<<" Y "<<tess.GetMeshPoint(max_loc1).y<<" Z "<<tess.GetMeshPoint(max_loc1).z<<" volume "<<tess.GetVolume(max_loc1)<<" max volume "<<max_volume<<" min volume "<<min_volume<<std::endl; 
            }
	    if(rank == max_data[2].mpi_id)
	      std::cout<<"rank "<<rank<<" i "<<max_loc2<<" sub_x "<<sub_x[max_loc2]<<std::endl;
        std::fill_n(sub_x.begin(), sub_x.size(), -1.0);
	    // throw UniversalError("CG did not converge");
        }
#ifdef RICH_MPI
        MPI_exchange_data(tess, sub_x, true);
#endif
        return sub_x;
    }
    
    std::vector<double> BiCGSTAB(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution, bool &good_end)  //total_iters is to store # of iters in it
    {
        BiCGSTABWorkspace workspace;
        return BiCGSTAB(tolerance, total_iters, tess, cells, dt, matrix_builder, time, sub_x_solution, good_end, workspace);
    }

    std::vector<double> &BiCGSTAB(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution, bool &good_end, BiCGSTABWorkspace &workspace)  //total_iters is to store # of iters in it
    {
        MEMORY_PROFILE_SCOPE("diffusion BiCGSTAB");
        size_t slice = 1;
#ifdef ENERGY_GROUPS_NUM
        slice = ENERGY_GROUPS_NUM;
#endif
        size_t Nlocal = tess.GetPointNo() * slice;
        
        // NOTE: when using MPI with > 1 proc, A will be only a sub-matrix (a subset of rows) of the full matrix
        // since we are 1D decomposing the matrix by rows
        // b will be the full vector

        int nprocs = 1, rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
        MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    #endif
        int const max_iter = 10000;

        mat &A = workspace.A;
        size_t_mat &A_indeces = workspace.A_indeces;
        std::vector<double> &b = workspace.b;
        std::vector<double> &sub_x = workspace.sub_x; // this is for the initial guess
        A.clear();
        A_indeces.clear();
        b.clear();
        sub_x.clear();
        {
            MEMORY_PROFILE_SCOPE("diffusion matrix build");
            matrix_builder.BuildMatrix(tess, A, A_indeces, cells, dt, b, sub_x, time);
        }
        std::vector<size_t> &A_row_ptr = workspace.A_row_ptr;
        std::vector<size_t> &A_col_idx = workspace.A_col_idx;
        std::vector<double> &A_values = workspace.A_values;
        A_row_ptr.clear();
        A_col_idx.clear();
        A_values.clear();
        build_crs(A, A_indeces, A_row_ptr, A_col_idx, A_values);
        std::vector<double> &M = workspace.M; // The preconditioner
        M.clear();
        if(use_crs_matvec)
            build_M_crs(A_row_ptr, A_col_idx, A_values, M);
        else
            build_M(A, A_indeces, M);
        auto matvec = [&](const std::vector<double> &in, std::vector<double> &out)
        {
            if(use_crs_matvec)
                mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, in, out);
            else
                mat_times_vec(A, A_indeces, in, out);
        };
        std::vector<double> &r_old = workspace.r_old;
        std::vector<double> &sub_a_times_p = workspace.sub_a_times_p;
        std::vector<double> &sub_r = workspace.sub_r;
        r_old.clear();
        sub_a_times_p.clear();
        sub_r.clear();
#ifdef RICH_MPI
        MPI_exchange_data(tess, sub_x, true, slice);
#endif
        matvec(sub_x, sub_a_times_p);
        // Find maximum value of A, this is used for normalization of the error
        double maxA[2] = {0, 0};
        for(size_t i = 0; i < A.size(); ++i)
        {
            maxA[0] = std::max(maxA[0], std::abs(sub_x[i]));
            maxA[1] = std::max(maxA[1], std::abs(b[i]));
        }       
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &maxA, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);  
        double rho0 = mpi_dot_product(sub_r, sub_r);  
        std::vector<double> &sub_p = workspace.sub_p;
        std::vector<double> &sub_r0 = workspace.sub_r0;
        sub_p = sub_r;
        sub_r0 = sub_r;
        std::vector<bool> fixed_negative_x(Nlocal, false);
        sub_p.resize(Nlocal);
        sub_x.resize(Nlocal);
        sub_r0.resize(Nlocal);
        vector_rescale(sub_p, M, sub_p);
        std::vector<double> &y = workspace.y;
        std::vector<double> &z = workspace.z;
        std::vector<double> &v = workspace.v;
        std::vector<double> &h = workspace.h;
        std::vector<double> &s = workspace.s;
        std::vector<double> &t = workspace.t;
        std::vector<double> &scratch_rescale1 = workspace.scratch_rescale1;
        std::vector<double> &scratch_rescale2 = workspace.scratch_rescale2;
        std::vector<double> &old_x = workspace.old_x;
        y.assign(Nlocal, 0);
        z.assign(Nlocal, 0);
        v.assign(Nlocal, 0);
        h.assign(Nlocal, 0);
        s.assign(Nlocal, 0);
        t.assign(Nlocal, 0);
        scratch_rescale1.assign(Nlocal, 0);
        scratch_rescale2.assign(Nlocal, 0);
        old_x = sub_x; 
        std::vector<double> org_x = sub_x;
        size_t Ntotal = Nlocal;
        double sub_r_sqrd = mpi_dot_product(sub_r, sub_p);
        double const delta_init = sub_r_sqrd;
        vector_rescale(b, M, scratch_rescale1);
        double const scale_b = mpi_dot_product(scratch_rescale1, b);
        double sub_r_sqrd_old = 0, sub_p_by_ap = 0, alpha = 0, beta = 0;
        struct
        {
            double val;
            int mpi_id;
        } max_data[3];
        max_data[0].mpi_id = rank;
        max_data[1].mpi_id = rank;
        max_data[2].mpi_id = rank;
        
        size_t max_loc0 = 0, max_loc1 = 0, max_loc2 = 0;
        // Main Conjugate Gradient loop
        // this loop must be serial b/c CG is an iterative method

        sub_p = sub_r;
        double max_sub_x = 0;
        double error = std::numeric_limits<double>::max();
        bool print = false;//rank == 0 && time >  136.97915 && time <  136.98;
        for (int i = 0; i < max_iter; i++) {
            total_iters = i;
            // note: make sure matrix is big enough for the number of processors you are using!
            if(i > 1 && i % 50 == 0)
            {
#ifdef RICH_MPI
                MPI_exchange_data(tess, sub_x, true, slice);
#endif
                matvec(sub_x, sub_a_times_p);
                sub_x.resize(Nlocal);
                vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);
                sub_r.resize(Nlocal);
                sub_r0 = sub_r;
                sub_p = sub_r;
                rho0 = mpi_dot_product(sub_r0, sub_r);
                vector_rescale(sub_r, M, scratch_rescale1);
                sub_r_sqrd = mpi_dot_product(scratch_rescale1, sub_r);
            }
            if(std::abs(rho0) < std::numeric_limits<double>::min()*1e100){
                if(rank == 0)
                    std::cout<<"Exited BiCGSTAB: rho0 is very small: "<<rho0<<std::endl;
                finalize_conjugate_gradient(slice, max_loc0, max_loc1, rank, i, error, max_data[0].val, 
                    max_data[1].val, max_data[2].val, max_data[0].mpi_id, max_data[1].mpi_id, sub_x, sub_x_solution,
                    cells, tess, total_iters, A_row_ptr, A_col_idx, A_values, b, Nlocal, matrix_builder.GetLengthScale(),
                    sub_a_times_p, sub_r);
                good_end = true;
                break;
            }
            if(print)
                std::cout<<"iter "<<i<<" rho0 "<<rho0<<" sub_r_sqrd "<<sub_r_sqrd<<std::endl;
            // r_old = sub_r;                 // Store previous residual
            sub_r_sqrd_old = sub_r_sqrd;  // save a recalculation of r_old^2 later
            if(std::min(sub_r_sqrd_old, std::abs(rho0)) < std::numeric_limits<double>::min()*1e100){
                if(rank == 0)
                    std::cout<<"Exited BiCGSTAB: sub_r_sqrd_old is very small: "<<sub_r_sqrd_old<<std::endl;
                finalize_conjugate_gradient(slice, max_loc0, max_loc1, rank, i, error, max_data[0].val, 
                    max_data[1].val, max_data[2].val, max_data[0].mpi_id, max_data[1].mpi_id, sub_x, sub_x_solution,
                    cells, tess, total_iters, A_row_ptr, A_col_idx, A_values, b, Nlocal, matrix_builder.GetLengthScale(),
                    sub_a_times_p, sub_r);
                break;
            }
            max_data[2].mpi_id = rank;
            max_data[0].mpi_id = rank;
            vector_rescale(sub_p, M, y);
#ifdef RICH_MPI
            MPI_exchange_data(tess, y, true, slice);
#endif
            matvec(y, v);
            y.resize(Nlocal);
            double const sub_r0_v = mpi_dot_product(sub_r0, v);
            double const alpha = std::abs(sub_r0_v) < std::numeric_limits<double>::min()*1e100 ? 0.0 : rho0 / sub_r0_v;
            if(print)
                std::cout<<"alpha "<<alpha<<" sub_r0_v "<<sub_r0_v<<std::endl;
            vec_lin_combo(1.0, sub_x, alpha, y, h);
            vec_lin_combo(1.0, sub_r, -alpha, v, s);
            vector_rescale(s, M, z);
#ifdef RICH_MPI
            MPI_exchange_data(tess, z, true, slice);
#endif
            matvec(z, t);
            z.resize(Nlocal);
            vector_rescale(t, M, scratch_rescale1);
            vector_rescale(s, M, scratch_rescale2);
            double const up = mpi_dot_product(scratch_rescale1, scratch_rescale2);
            double const down = mpi_dot_product(scratch_rescale1, scratch_rescale1);
            double w = std::abs(down) < std::numeric_limits<double>::min()*1e100 ? 0.0 : up / down;
            if(print)
                std::cout<<"w "<<w<<" up "<<up<<" down "<<down<<std::endl;
            if(std::abs(alpha) < std::numeric_limits<double>::min()*1e100 && std::abs(w) < std::numeric_limits<double>::min()*1e100)
            {
                if(rank == 0)
                    std::cout<<"Exited BiCGSTAB: alpha and w very small: alpha "<<alpha<<" w "<<w<<" rho0 "<<rho0<<std::endl;
                finalize_conjugate_gradient(slice, max_loc0, max_loc1, rank, i, error, max_data[0].val, 
                    max_data[1].val, max_data[2].val, max_data[0].mpi_id, max_data[1].mpi_id, sub_x, sub_x_solution,
                    cells, tess, total_iters, A_row_ptr, A_col_idx, A_values, b, Nlocal, matrix_builder.GetLengthScale(),
                    sub_a_times_p, sub_r);
                good_end = true;
                break;
            }
            max_sub_x = 0;
            old_x = sub_x;
            vec_lin_combo(1.0, h, w, z, sub_x);
            vec_lin_combo(1.0, s, -w, t, sub_r);
            vector_rescale(sub_r, M, scratch_rescale1);
            sub_r_sqrd = mpi_dot_product(scratch_rescale1, sub_r);
            
            error = sub_r_sqrd / scale_b;
            if(print)
                std::cout<<"iter "<<i<<" error "<<error<<std::endl;
            max_data[0].val = 0;
            max_data[1].val = 0;
            max_data[2].val = 0;
            max_data[0].mpi_id = rank;
            max_data[1].mpi_id = rank;
            max_data[2].mpi_id = rank;
            max_sub_x = 0;
            for(std::size_t j = 0; j < Nlocal; ++j)
                max_sub_x = std::max(max_sub_x, std::abs(sub_x[j]));
#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, &max_sub_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
            for(size_t j = 0; j < Nlocal; ++j)
            {
                if(std::abs(sub_r[j]) > max_data[1].val * (std::abs(A[j][0] * (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + max_sub_x * 2e-5))))
                {
                    max_data[1].val = std::abs(sub_r[j]) / (std::abs(A[j][0] * (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + max_sub_x * 2e-5)));
                    max_loc1 = j;
                }
                if(sub_x[j] < -max_sub_x * 1e-10)
                {
                    if(i > 10 && not fixed_negative_x[j])
                    {
                        fixed_negative_x[j] = true;
                        sub_x[j] = 0.5 * org_x[j];
                    }
                    else
                    {
                        max_loc2 = j;
                        max_data[2].val = 1;
                    }
                }
                if(std::abs(sub_x[j] - old_x[j]) > max_data[0].val * (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + max_sub_x * 2e-5))
                {
                    max_data[0].val = std::abs(sub_x[j] - old_x[j]) / (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + max_sub_x * 2e-5);
                    max_loc0 = j;
                }
            }
#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, max_data, 3, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
#endif
            if(i >= 10 && error < 1e-2 && max_data[1].val < 1e-6 && max_data[0].val < 1e-6)
            {
                if(rank == 0)
                    std::cout << "Exited BiCGSTAB: max0 and max1 below 1e-6, max0 "
                              << max_data[0].val << " max1 " << max_data[1].val << std::endl;
                finalize_conjugate_gradient(slice, max_loc0, max_loc1, rank, i, error, max_data[0].val,
                    max_data[1].val, max_data[2].val, max_data[0].mpi_id, max_data[1].mpi_id, sub_x, sub_x_solution,
                    cells, tess, total_iters, A_row_ptr, A_col_idx, A_values, b, Nlocal, matrix_builder.GetLengthScale(),
                    sub_a_times_p, sub_r);
                good_end = true;
                break;
            }
            // Convergence test
            if (error < tolerance)
            {
                if((max_data[1].val < 1e-5 && max_data[0].val < 1e-5 && ((i > 25 && max_data[1].val < 1e-10) || max_data[2].val < 0.5)) || error < 1e-100) 
                { // norm is just sqrt(dot product so don't need to use a separate norm fnc) // vector norm needs to use a all reduce!
                    finalize_conjugate_gradient(slice, max_loc0, max_loc1, rank, i, error, max_data[0].val, 
                        max_data[1].val, max_data[2].val, max_data[0].mpi_id, max_data[1].mpi_id, sub_x, sub_x_solution,
                        cells, tess, total_iters, A_row_ptr, A_col_idx, A_values, b, Nlocal, matrix_builder.GetLengthScale(),
                        sub_a_times_p, sub_r);
                    good_end = true;
                    break;
                }
            }
            double const rho0_new = mpi_dot_product(sub_r0, sub_r);
            double const beta = rho0_new * alpha / (w * rho0);
            if(print)
                std::cout<<"beta "<<beta<<" rho0_new "<<rho0_new<<" rho0 "<<rho0<<" w "<<w<<" alpha "<<alpha<<std::endl;
            vec_lin_combo(1.0, sub_p, -w, v, t);
            vec_lin_combo(1.0, sub_r, beta, t, sub_p); 
            rho0 = rho0_new;
        }
        if(not good_end)
        {
            max_loc0 /= slice;
            max_loc1 /= slice;
            if(rank == 0)
	            std:: cout <<"not good end, delta "<<sub_r_sqrd<<" maxdata2 "<<max_data[2].val<<" iterations "<<total_iters<<" scale_b "<<scale_b<<std::endl;
            if(rank == max_data[0].mpi_id)
                std::cout<<"Max0 "<<max_data[0].val<<" cell ID "<<cells[max_loc0].ID<<" density "<<cells[max_loc0].density<<" temperature "<<cells[max_loc0].temperature<<" Er "<<cells[max_loc0].Erad * cells[max_loc0].density
                <<" X "<<tess.GetMeshPoint(max_loc0).x<<" Y "<<tess.GetMeshPoint(max_loc0).y<<" Z "<<tess.GetMeshPoint(max_loc0).z<<" sub_x "<<sub_x[max_loc0]<<std::endl; 
            if(rank == max_data[1].mpi_id)
                std::cout<<"Max1 "<<max_data[1].val<<" cell ID "<<cells[max_loc1].ID<<" density "<<cells[max_loc1].density<<" temperature "<<cells[max_loc1].temperature<<" Er "<<cells[max_loc1].Erad * cells[max_loc1].density
                <<" X "<<tess.GetMeshPoint(max_loc1).x<<" Y "<<tess.GetMeshPoint(max_loc1).y<<" Z "<<tess.GetMeshPoint(max_loc1).z<<" sub_x "<<sub_x[max_loc1]<<std::endl; 
            if(rank == max_data[2].mpi_id)
                std::cout<<"rank "<<rank<<" i "<<max_loc2 <<" sub_x "<<sub_x[max_loc2]<<std::endl;
            std::fill_n(sub_x.begin(), sub_x.size(), -1.0);
            // throw UniversalError("CG did not converge");
        }
#ifdef RICH_MPI
        MPI_exchange_data(tess, sub_x, true, slice);
        MPI_exchange_data(tess, sub_x_solution, true, slice);
#endif
        return sub_x;
    }
}
