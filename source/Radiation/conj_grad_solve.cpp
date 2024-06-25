#include "conj_grad_solve.hpp"
#include <limits>

namespace CG
{
    // Matrix times vector
    void mat_times_vec(const mat &sub_A_values, const size_t_mat &sub_A_indices, const std::vector<double> &v, 
        std::vector<double> &result)
    {

        // NOTE: when using MPI with > 1 proc, A will be only a sub-matrix (a subset of rows) of the full matrix
        // since we are 1D decomposing the matrix by rows

        size_t const sub_num_rows = sub_A_values.size();
        size_t const sub_num_cols = sub_A_values[0].size();
        if(sub_num_rows == 0)
            return;
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

    // Linear combination of vectors
    void vec_lin_combo(double a, const std::vector<double> &u, double b, const std::vector<double> &v, 
        std::vector<double> &result)
    {
        if(u.size() != v.size())
            throw UniversalError("Unequal vector sizes in vec_lin_combo");
        size_t n = u.size();
        result.resize(n, 0);
        for (size_t j = 0; j < n; j++)
            result[j] = a * u[j] + b * v[j];
    }


    // performs a reduction over the sub-vectors which are passed to it... All_Reduce broadcasts the value to all procs
    double mpi_dot_product(const std::vector<double> &sub_u, const std::vector<double> &sub_v) // need to pass it the buffer where to keep the result
    {
        if(sub_u.size() != sub_v.size())
            throw UniversalError("Unequal vector sizes in vec_lin_combo");
        size_t length = sub_u.size();

        double sub_prod = 0.0;
        for (size_t i = 0; i < length; i++) {
            sub_prod += sub_u[i] * sub_v[i];
        }
#ifdef RICH_MPI
        // do a reduction over sub_prod to get the total dot product
        MPI_Allreduce(MPI_IN_PLACE, &sub_prod, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        return sub_prod;
    }
    
    void build_M(const mat &sub_A_values, const size_t_mat &sub_A_indices, std::vector<double> &M)
    {
        size_t const sub_num_rows = sub_A_values.size();
        size_t const sub_num_cols = sub_A_values[0].size();
        if(sub_num_rows == 0)
            return;
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
        int const max_iter = 25000;

        mat A;
        size_t_mat A_indeces;
        std::vector<double> b;
        std::vector<double> sub_x; // this is for the initial guess
        matrix_builder.BuildMatrix(tess, A, A_indeces, cells, dt, b, sub_x, time);
        std::vector<double> M; // The preconditioner
        build_M(A, A_indeces, M);
        std::vector<double> r_old, sub_a_times_p;
        std::vector<double> sub_r;
#ifdef RICH_MPI
        MPI_exchange_data2(tess, sub_x, true);
#endif
        mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
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
        std::vector<double> sub_p(sub_r);
        sub_p.resize(Nlocal);
        sub_x.resize(Nlocal);
        sub_p = vector_rescale(sub_p, M);
        std::vector<double> result1, result2(Nlocal, 0), result3, p(sub_p), old_result2(Nlocal, 0);
        std::vector<double> old_x = sub_x;
        size_t Ntotal = Nlocal;
#ifdef RICH_MPI
        MPI_exchange_data2(tess, p, true);
        MPI_Allreduce(&Nlocal, &Ntotal, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        double sub_r_sqrd = mpi_dot_product(sub_r, sub_p);
        double const delta_init = sub_r_sqrd;
        // double sub_r_sqrd_convergence = mpi_dot_product(sub_r, sub_r);
        if(rank == 0)
            std::cout<<"CG init delta "<<delta_init<<std::endl;
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

            mat_times_vec(A, A_indeces, p, sub_a_times_p);  //split up with MPI and then finer parallelize with openmp

            sub_p_by_ap = mpi_dot_product(sub_p, sub_a_times_p);

            alpha = sub_r_sqrd / (sub_p_by_ap + std::numeric_limits<double>::min() * 100);         

            // Next estimate of solution
            vec_lin_combo(1.0, sub_x, alpha, sub_p, result1);
            sub_x = result1;
            if(i > 1 && i % 50 == 0)
            {
#ifdef RICH_MPI
                MPI_exchange_data2(tess, sub_x, true);
#endif
                mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
                sub_x.resize(Nlocal);
                vec_lin_combo(1.0, b, -1.0, sub_a_times_p, result1);    
            }
            else
                vec_lin_combo(1.0, sub_r, -alpha, sub_a_times_p, result1);
            sub_r = result1;

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
            result2 = vector_rescale(sub_r, M);
            sub_r_sqrd = mpi_dot_product(sub_r, result2);
            // recall that we can't have a 'break' within an openmp parallel region, so end it here then all threads are merged, and the convergence is checked
            // Convergence test
            if (sub_r_sqrd < delta_init * tolerance//std::sqrt(sub_r_sqrd_convergence / Ntotal) < tolerance * maxA [1]
                && max_data[1].val < 1e-6 && max_data[0].val < 1e-6 && (i > 250 || max_data[2].val < 0.5)) { // norm is just sqrt(dot product so don't need to use a separate norm fnc) // vector norm needs to use a all reduce!
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
                MPI_exchange_data2(tess, sub_x, true);
#endif
                sub_x_solution = sub_x;
                mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
                sub_x.resize(Nlocal);
                vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);
                break;
            }
            old_x = sub_x;
            vec_lin_combo(1.0, result2, -1.0, old_result2, result3);   
            double const  Polak_Ribiere = mpi_dot_product(sub_r, result3);
            beta = std::max(0.0, Polak_Ribiere / sub_r_sqrd_old);       
            
            vec_lin_combo(1.0, result2, beta, sub_p, result3);             // Next gradient
            sub_p = result3;
            p = sub_p;
    #ifdef RICH_MPI
            MPI_exchange_data2(tess, p, true);
    #endif
        }
        if(not good_end)
        {
            if(rank == 0)
	      std:: cout <<"not good end, delta "<<sub_r_sqrd<<" maxdata2 "<<max_data[2].val<<std::endl;
            if(rank == max_data[0].mpi_id)
                std::cout<<"Max0 "<<max_data[0].val<<" cell ID "<<cells[max_loc0].ID<<" density "<<cells[max_loc0].density<<" temperature "<<cells[max_loc0].temperature<<" Er "<<cells[max_loc0].Erad * cells[max_loc0].density
                <<" X "<<tess.GetMeshPoint(max_loc0).x<<" Y "<<tess.GetMeshPoint(max_loc0).y<<" Z "<<tess.GetMeshPoint(max_loc0).z<<std::endl; 
            if(rank == max_data[1].mpi_id)
                std::cout<<"Max1 "<<max_data[1].val<<" cell ID "<<cells[max_loc1].ID<<" density "<<cells[max_loc1].density<<" temperature "<<cells[max_loc1].temperature<<" Er "<<cells[max_loc1].Erad * cells[max_loc1].density
                <<" X "<<tess.GetMeshPoint(max_loc1).x<<" Y "<<tess.GetMeshPoint(max_loc1).y<<" Z "<<tess.GetMeshPoint(max_loc1).z<<std::endl; 
	    if(rank == max_data[2].mpi_id)
	      std::cout<<"rank "<<rank<<" i "<<max_loc2<<" sub_x "<<sub_x[max_loc2]<<std::endl;
        std::fill_n(sub_x.begin(), sub_x.size(), -1.0);
	    // throw UniversalError("CG did not converge");
        }
#ifdef RICH_MPI
        MPI_exchange_data2(tess, sub_x, true);
#endif
        return sub_x;
    }

    std::vector<double> BiCGSTAB(const double tolerance, int &total_iters,
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
        int const max_iter = 1000;

        mat A;
        size_t_mat A_indeces;
        std::vector<double> b;
        std::vector<double> sub_x; // this is for the initial guess
        matrix_builder.BuildMatrix(tess, A, A_indeces, cells, dt, b, sub_x, time);
        std::vector<double> M; // The preconditioner
        build_M(A, A_indeces, M);
        std::vector<double> r_old, sub_a_times_p;
        std::vector<double> sub_r;
#ifdef RICH_MPI
        MPI_exchange_data2(tess, sub_x, true);
#endif
        mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
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
        std::vector<double> sub_p(sub_r), sub_r0(sub_r);
        sub_p.resize(Nlocal);
        sub_x.resize(Nlocal);
        sub_r0.resize(Nlocal);
        sub_p = vector_rescale(sub_p, M);
        std::vector<double> y(Nlocal, 0), z(Nlocal, 0), v(Nlocal, 0), h(Nlocal, 0), s(Nlocal, 0), t(Nlocal, 0);
        std::vector<double> old_x = sub_x;
        size_t Ntotal = Nlocal;
        double sub_r_sqrd = mpi_dot_product(sub_r, sub_p);
        double const delta_init = sub_r_sqrd;
        // double sub_r_sqrd_convergence = mpi_dot_product(sub_r, sub_r);
        if(rank == 0)
            std::cout<<"CG init delta "<<delta_init<<std::endl;
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

        sub_p = sub_r;

        for (int i = 0; i < max_iter; i++) {
            // note: make sure matrix is big enough for the number of processors you are using!
            max_data[2].mpi_id = rank;
            max_data[0].mpi_id = rank;
            // r_old = sub_r;                 // Store previous residual
            sub_r_sqrd_old = sub_r_sqrd;  // save a recalculation of r_old^2 later

            y = vector_rescale(sub_p, M);
#ifdef RICH_MPI
            MPI_exchange_data2(tess, y, true);
#endif
            mat_times_vec(A, A_indeces, y, v);
            y.resize(Nlocal);
            double const alpha = rho0 / mpi_dot_product(sub_r0, v);
            vec_lin_combo(1.0, sub_x, alpha, y, h);
            vec_lin_combo(1.0, sub_r, -alpha, v, s);
            z = vector_rescale(s, M);
#ifdef RICH_MPI
            MPI_exchange_data2(tess, z, true);
#endif
            mat_times_vec(A, A_indeces, z, t);
            z.resize(Nlocal);
            double const up = mpi_dot_product(vector_rescale(t, M), vector_rescale(s, M));
            double const down = mpi_dot_product(vector_rescale(t, M), vector_rescale(t, M));
            double const w = up / down;
            old_x = sub_x;
            vec_lin_combo(1.0, h, w, z, sub_x);
            vec_lin_combo(1.0, s, -w, t, sub_r);
            sub_r_sqrd = mpi_dot_product(vector_rescale(sub_r, M), sub_r);
            
            // Convergence test
            if (sub_r_sqrd < delta_init * tolerance)
            {
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
                if(max_data[1].val < 1e-6 && max_data[0].val < 1e-6 && (i > 250 || max_data[2].val < 0.5)) { // norm is just sqrt(dot product so don't need to use a separate norm fnc) // vector norm needs to use a all reduce!
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
                    MPI_exchange_data2(tess, sub_x, true);
#endif
                    sub_x_solution = sub_x;
                    mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
                    sub_x.resize(Nlocal);
                    vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);
                    break;
                }
            }
            double const rho0_new = mpi_dot_product(sub_r0, sub_r);
            double const beta = rho0_new * alpha / (w * rho0);
            vec_lin_combo(1.0, sub_p, -w, v, t);
            vec_lin_combo(1.0, sub_r, beta, t, sub_p); 
            rho0 = rho0_new;
     
    
        }
        if(not good_end)
        {
            if(rank == 0)
	            std:: cout <<"not good end, delta "<<sub_r_sqrd<<" maxdata2 "<<max_data[2].val<<std::endl;
            if(rank == max_data[0].mpi_id)
                std::cout<<"Max0 "<<max_data[0].val<<" cell ID "<<cells[max_loc0].ID<<" density "<<cells[max_loc0].density<<" temperature "<<cells[max_loc0].temperature<<" Er "<<cells[max_loc0].Erad * cells[max_loc0].density
                <<" X "<<tess.GetMeshPoint(max_loc0).x<<" Y "<<tess.GetMeshPoint(max_loc0).y<<" Z "<<tess.GetMeshPoint(max_loc0).z<<std::endl; 
            if(rank == max_data[1].mpi_id)
                std::cout<<"Max1 "<<max_data[1].val<<" cell ID "<<cells[max_loc1].ID<<" density "<<cells[max_loc1].density<<" temperature "<<cells[max_loc1].temperature<<" Er "<<cells[max_loc1].Erad * cells[max_loc1].density
                <<" X "<<tess.GetMeshPoint(max_loc1).x<<" Y "<<tess.GetMeshPoint(max_loc1).y<<" Z "<<tess.GetMeshPoint(max_loc1).z<<std::endl; 
	    if(rank == max_data[2].mpi_id)
	      std::cout<<"rank "<<rank<<" i "<<max_loc2<<" sub_x "<<sub_x[max_loc2]<<std::endl;
        std::fill_n(sub_x.begin(), sub_x.size(), -1.0);
	    // throw UniversalError("CG did not converge");
        }
#ifdef RICH_MPI
        MPI_exchange_data2(tess, sub_x, true);
#endif
        return sub_x;
    }

    std::vector<double> QMRCGSTAB(const double tolerance, int &total_iters,
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
        int const max_iter = 1000;

        mat A;
        size_t_mat A_indeces;
        std::vector<double> b;
        std::vector<double> sub_x; // this is for the initial guess
        matrix_builder.BuildMatrix(tess, A, A_indeces, cells, dt, b, sub_x, time);
        std::vector<double> M; // The preconditioner
        build_M(A, A_indeces, M);
        std::vector<double> r_old, sub_a_times_p;
        std::vector<double> sub_r;
#ifdef RICH_MPI
        MPI_exchange_data2(tess, sub_x, true);
#endif
        mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
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
        std::vector<double> sub_r0(sub_r);
        sub_x.resize(Nlocal);
        sub_r0.resize(Nlocal);
        std::vector<double> sub_p(Nlocal, 0), y(Nlocal, 0), z(Nlocal, 0), v(Nlocal, 0), s(Nlocal, 0), t(Nlocal, 0), d(Nlocal, 0), d_temp(Nlocal, 0), x_temp(Nlocal, 0);
        std::vector<double> old_x = sub_x;
        size_t Ntotal = Nlocal;
        double sub_r_sqrd = mpi_dot_product(sub_r, vector_rescale(sub_r, M));
        double const delta_init = sub_r_sqrd;
        // double sub_r_sqrd_convergence = mpi_dot_product(sub_r, sub_r);
        if(rank == 0)
            std::cout<<"CG init delta "<<delta_init<<std::endl;
        double sub_r_sqrd_old = 0, sub_p_by_ap = 0, alpha = 1, beta = 0, w = 1;
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

        // sub_p = sub_r;
        double rho0 = 1;
        double tau = std::sqrt(mpi_dot_product(sub_r0, sub_r0));
        // w = tau;
        // rho0 = tau;
        double theta = 0;
        double eta = 0;
        sub_r = vector_rescale(sub_r0, M);
        for (int i = 0; i < max_iter; i++) {
            // note: make sure matrix is big enough for the number of processors you are using!
            max_data[2].mpi_id = rank;
            max_data[0].mpi_id = rank;
            // r_old = sub_r;                 // Store previous residual
            sub_r_sqrd_old = sub_r_sqrd;  // save a recalculation of r_old^2 later

            double rho0_new = mpi_dot_product(sub_r, sub_r0);
            beta = rho0_new * alpha / (rho0 * w);
            rho0 = rho0_new;
            vec_lin_combo(1.0, sub_p, -w, v, t);
            vec_lin_combo(1.0, sub_r, beta, t, sub_p);

            // y = vector_rescale(sub_p, M);
#ifdef RICH_MPI
            MPI_exchange_data2(tess, sub_p, true);
#endif
            mat_times_vec(A, A_indeces, sub_p, v);
            v = vector_rescale(v, M);
            sub_p.resize(Nlocal);
            double down = mpi_dot_product(sub_r0, v);
            alpha = rho0 / down;
            vec_lin_combo(1.0, sub_r, -alpha, v, s);


            double const s_2 = mpi_dot_product(s, s);
            double const theta_temp = std::sqrt(s_2) / tau;
            double c = 1.0 / std::sqrt(1 + theta_temp * theta_temp);
            double const tau_temp = tau * theta_temp * c;
            double const eta_temp = c * c * alpha;
            vec_lin_combo(1.0, sub_p,  theta * theta * eta / alpha, d, d_temp);
            vec_lin_combo(1.0, sub_x,  eta_temp, d_temp, x_temp);

#ifdef RICH_MPI
            MPI_exchange_data2(tess, s, true);
#endif
            mat_times_vec(A, A_indeces, s, t);
            t = vector_rescale(t, M);
            s.resize(Nlocal);
            double const up = mpi_dot_product(t, s);
            down = mpi_dot_product(t, t);
            w = up / down;
            vec_lin_combo(1.0, s, -w, t, sub_r);

            theta = std::sqrt(mpi_dot_product(sub_r, sub_r)) / tau_temp;
            c = 1.0 / std::sqrt(1 + theta * theta);
            tau = tau_temp * theta * c;
            eta = c * c * w;
            vec_lin_combo(1.0, s,  theta_temp * theta_temp * eta_temp / w, d_temp, d);
            old_x = sub_x;
            vec_lin_combo(1.0, x_temp,  eta, d, sub_x);
            if(i % 10 == 0)
            {
#ifdef RICH_MPI
                MPI_exchange_data2(tess, sub_x, true);
#endif
                mat_times_vec(A, A_indeces, sub_x, t);
                sub_x.resize(Nlocal);
                vec_lin_combo(1.0, b, -1, t, sub_r);
                sub_r= vector_rescale(sub_r, M);
            }
            sub_r_sqrd = mpi_dot_product(vector_rescale(sub_r, M), sub_r);
            // Convergence test
            if (sub_r_sqrd < delta_init * tolerance)
            {
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
                if(max_data[1].val < 1e-6 && max_data[0].val < 1e-6 && (i > 250 || max_data[2].val < 0.5)) { // norm is just sqrt(dot product so don't need to use a separate norm fnc) // vector norm needs to use a all reduce!
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
                    MPI_exchange_data2(tess, sub_x, true);
#endif
                    sub_x_solution = sub_x;
                    mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
                    sub_x.resize(Nlocal);
                    vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);
                    break;
                }
            }
        }
        if(not good_end)
        {
            if(rank == 0)
	            std:: cout <<"not good end, delta "<<sub_r_sqrd<<" maxdata2 "<<max_data[2].val<<std::endl;
            if(rank == max_data[0].mpi_id)
                std::cout<<"Max0 "<<max_data[0].val<<" cell ID "<<cells[max_loc0].ID<<" density "<<cells[max_loc0].density<<" temperature "<<cells[max_loc0].temperature<<" Er "<<cells[max_loc0].Erad * cells[max_loc0].density
                <<" X "<<tess.GetMeshPoint(max_loc0).x<<" Y "<<tess.GetMeshPoint(max_loc0).y<<" Z "<<tess.GetMeshPoint(max_loc0).z<<std::endl; 
            if(rank == max_data[1].mpi_id)
                std::cout<<"Max1 "<<max_data[1].val<<" cell ID "<<cells[max_loc1].ID<<" density "<<cells[max_loc1].density<<" temperature "<<cells[max_loc1].temperature<<" Er "<<cells[max_loc1].Erad * cells[max_loc1].density
                <<" X "<<tess.GetMeshPoint(max_loc1).x<<" Y "<<tess.GetMeshPoint(max_loc1).y<<" Z "<<tess.GetMeshPoint(max_loc1).z<<std::endl; 
	    if(rank == max_data[2].mpi_id)
	      std::cout<<"rank "<<rank<<" i "<<max_loc2<<" sub_x "<<sub_x[max_loc2]<<std::endl;
        std::fill_n(sub_x.begin(), sub_x.size(), -1.0);
	    // throw UniversalError("CG did not converge");
        }
#ifdef RICH_MPI
        MPI_exchange_data2(tess, sub_x, true);
#endif
        return sub_x;
    }

    std::vector<double> QMRCORSTAB(const double tolerance, int &total_iters,
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
        int const max_iter = 1000;

        mat A;
        size_t_mat A_indeces;
        std::vector<double> b;
        std::vector<double> sub_x; // this is for the initial guess
        matrix_builder.BuildMatrix(tess, A, A_indeces, cells, dt, b, sub_x, time);
        std::vector<double> M; // The preconditioner
        build_M(A, A_indeces, M);
        std::vector<double> r_old, sub_a_times_p;
        std::vector<double> sub_r;
#ifdef RICH_MPI
        MPI_exchange_data2(tess, sub_x, true);
#endif
        mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
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
        std::vector<double> sub_r0(sub_r);
#ifdef RICH_MPI
        MPI_exchange_data2(tess, sub_r, true);
#endif
        mat_times_vec(A, A_indeces, sub_r, sub_r0);
        sub_r.resize(Nlocal);
        sub_x.resize(Nlocal);
        sub_r0.resize(Nlocal);
        std::vector<double> sub_p(Nlocal, 0), zhat(Nlocal, 0), sub_rBO(Nlocal, 0), zBO(Nlocal, 0), zsub_p(Nlocal, 0), t(Nlocal, 0), q(Nlocal, 0), d_temp(Nlocal, 0), x_temp(Nlocal, 0), zq(Nlocal, 0), zqhat(Nlocal, 0), s(Nlocal, 0), zd(Nlocal, 0), zd_temp(Nlocal, 0), e(Nlocal, 0), e_temp(Nlocal, 0), sub_rtemp(Nlocal, 0), zs(Nlocal, 0), zt((Nlocal, 0));
        std::vector<double> old_x = sub_x;
        size_t Ntotal = Nlocal;
        double sub_r_sqrd = mpi_dot_product(sub_r, vector_rescale(sub_r, M));
        double const delta_init = sub_r_sqrd;
        // double sub_r_sqrd_convergence = mpi_dot_product(sub_r, sub_r);
        // if(rank == 0)
        //     std::cout<<"CG init delta "<<delta_init<<std::endl;
        double sub_r_sqrd_old = 0, sub_p_by_ap = 0, alpha = 1, beta = 0, w = 1;
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

        // sub_p = sub_r;
        double rho0 = 1;
        sub_rBO = sub_r;
        zBO = vector_rescale(M, sub_r);
        double tau = std::sqrt(mpi_dot_product(zBO, zBO));
        // w = tau;
        // rho0 = tau;
        double theta = 0;
        double eta = 0;
        // sub_r = vector_rescale(sub_r0, M);
        for (int i = 0; i < max_iter; i++) {
            // note: make sure matrix is big enough for the number of processors you are using!
            max_data[2].mpi_id = rank;
            max_data[0].mpi_id = rank;
            // r_old = sub_r;                 // Store previous residual
            sub_r_sqrd_old = sub_r_sqrd;  // save a recalculation of r_old^2 later

#ifdef RICH_MPI
            MPI_exchange_data2(tess, zBO, true);
#endif
            mat_times_vec(A, A_indeces, zBO, zhat);
            zBO.resize(Nlocal);
            double rho0_new = mpi_dot_product(sub_r0, zhat);
            if(i == 0)
            {
                sub_p = sub_rBO;
                zsub_p = vector_rescale(sub_p, M);
                q = zhat;
            }
            else
            {
                beta = rho0_new * alpha / (w * rho0);
                vec_lin_combo(1.0, zsub_p, -w, zq, t);
                vec_lin_combo(1.0, zBO, beta, t, zsub_p);
                vec_lin_combo(1.0, q, -w, zqhat, t);
                vec_lin_combo(1.0, zhat, beta, t, q);
            }
            zq = vector_rescale(q, M);
            rho0 = rho0_new;
#ifdef RICH_MPI
            MPI_exchange_data2(tess, zq, true);
#endif           
            mat_times_vec(A, A_indeces, zq, zqhat);
            zq.resize(Nlocal);
            alpha = rho0 / mpi_dot_product(sub_r0, zqhat);
            vec_lin_combo(1.0, sub_rBO, -alpha, q, s);
            vec_lin_combo(1.0, zBO, -alpha, zq, zs);
            double theta_temp = std::sqrt(mpi_dot_product(zs, zs)) / tau;
            double c = 1 / std::sqrt(1 + theta_temp * theta_temp);
            double tau_temp = tau * theta_temp * c;
            double eta_temp = c * c * alpha;
            vec_lin_combo(1.0, zsub_p, theta * theta * eta / alpha, zd, zd_temp);
            vec_lin_combo(1.0, sub_x, eta_temp, zd_temp, x_temp);
            vec_lin_combo(1.0, q, theta * theta * eta / alpha, e, e_temp);
            vec_lin_combo(1.0, sub_r, -eta_temp, e_temp, sub_rtemp);
            vec_lin_combo(1.0, zhat, -alpha, zqhat, t);
            w = mpi_dot_product(t, s) / mpi_dot_product(t, t);
            zt = vector_rescale(M, t);
            vec_lin_combo(1.0, s, -w, t, sub_rBO);
            vec_lin_combo(1.0, zs, -w, zt, zBO);
            theta = std::sqrt(mpi_dot_product(zBO, zBO)) / tau_temp;
            c = 1 / std::sqrt(1 + theta * theta);
            tau = theta * tau_temp * c;
            eta = c * c * w;
            vec_lin_combo(1.0, zs, theta_temp * theta_temp * eta_temp / w, zd_temp, zd);
            old_x = sub_x;
            vec_lin_combo(1.0, x_temp, eta, zd, sub_x);
            vec_lin_combo(1.0, t, theta_temp * theta_temp * eta_temp / w, e_temp, e);
            vec_lin_combo(1.0, sub_rtemp, -eta, e, sub_r);

            sub_r_sqrd = mpi_dot_product(vector_rescale(sub_r, M), sub_r);
            // Convergence test
            if (sub_r_sqrd < delta_init * tolerance)
            {
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
                if(max_data[1].val < 1e-6 && max_data[0].val < 1e-6 && (i > 250 || max_data[2].val < 0.5)) { // norm is just sqrt(dot product so don't need to use a separate norm fnc) // vector norm needs to use a all reduce!
                    // if(rank == 0)
                    //     std:: cout << "Converged at iter = " << i <<" delta "<<sub_r_sqrd<<" negative value "<<max_data[2].val<<std::endl;
                    // if(rank == max_data[0].mpi_id)
                    //     std::cout<<"Max0 "<<max_data[0].val<<" cell ID "<<cells[max_loc0].ID<<" density "<<cells[max_loc0].density<<" temperature "<<cells[max_loc0].temperature<<" Er "<<cells[max_loc0].Erad * cells[max_loc0].density
                    //     <<" X "<<tess.GetMeshPoint(max_loc0).x<<" Y "<<tess.GetMeshPoint(max_loc0).y<<" Z "<<tess.GetMeshPoint(max_loc0).z<<std::endl; 
                    // if(rank == max_data[1].mpi_id)
                    //     std::cout<<"Max1 "<<max_data[1].val<<" cell ID "<<cells[max_loc1].ID<<" density "<<cells[max_loc1].density<<" temperature "<<cells[max_loc1].temperature<<" Er "<<cells[max_loc1].Erad * cells[max_loc1].density
                    //     <<" X "<<tess.GetMeshPoint(max_loc1).x<<" Y "<<tess.GetMeshPoint(max_loc1).y<<" Z "<<tess.GetMeshPoint(max_loc1).z<<std::endl; 
                    total_iters = i;
                    good_end = true;
#ifdef RICH_MPI
                    MPI_exchange_data2(tess, sub_x, true);
#endif
                    sub_x_solution = sub_x;
                    mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
                    sub_x.resize(Nlocal);
                    vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);
                    break;
                }
            }
        }
        if(not good_end)
        {
            if(rank == 0)
	            std:: cout <<"not good end, delta "<<sub_r_sqrd<<" maxdata2 "<<max_data[2].val<<std::endl;
            if(rank == max_data[0].mpi_id)
                std::cout<<"Max0 "<<max_data[0].val<<" cell ID "<<cells[max_loc0].ID<<" density "<<cells[max_loc0].density<<" temperature "<<cells[max_loc0].temperature<<" Er "<<cells[max_loc0].Erad * cells[max_loc0].density
                <<" X "<<tess.GetMeshPoint(max_loc0).x<<" Y "<<tess.GetMeshPoint(max_loc0).y<<" Z "<<tess.GetMeshPoint(max_loc0).z<<std::endl; 
            if(rank == max_data[1].mpi_id)
                std::cout<<"Max1 "<<max_data[1].val<<" cell ID "<<cells[max_loc1].ID<<" density "<<cells[max_loc1].density<<" temperature "<<cells[max_loc1].temperature<<" Er "<<cells[max_loc1].Erad * cells[max_loc1].density
                <<" X "<<tess.GetMeshPoint(max_loc1).x<<" Y "<<tess.GetMeshPoint(max_loc1).y<<" Z "<<tess.GetMeshPoint(max_loc1).z<<std::endl; 
	    if(rank == max_data[2].mpi_id)
	      std::cout<<"rank "<<rank<<" i "<<max_loc2<<" sub_x "<<sub_x[max_loc2]<<std::endl;
        std::fill_n(sub_x.begin(), sub_x.size(), -1.0);
	    // throw UniversalError("CG did not converge");
        }
#ifdef RICH_MPI
        MPI_exchange_data2(tess, sub_x, true);
#endif
        return sub_x;
    }
}

