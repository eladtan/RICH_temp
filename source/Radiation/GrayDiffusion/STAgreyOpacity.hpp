#ifndef STA_OPACITY_HPP
#define STA_OPACITY_HPP

#include "Diffusion.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/misc/utils.hpp"
#include "source/misc/simple_io.hpp"

/**
 * @brief The STAgreyOpacity class calculates the diffusion, Planck, and scattering opacities based on the given temperature and density.
 */
class STAgreyOpacity: public DiffusionCoefficientCalculator
{
private:
    std::vector<double> rho_, T_;  // Vectors to store the density and temperature values
    std::vector<std::vector<double>> rossland_, planck_, scatter_;  // 2D vectors to store the opacity values

public:
    /**
     * @brief Constructor to initialize the STAgreyOpacity object by reading the opacity data from the given file directory.
     * @param file_directory The directory containing the opacity data files.
     */
    STAgreyOpacity(std::string file_directory);

    /**
     * @brief Calculates the diffusion coefficient based on the given computational cell properties.
     * @param cell The computational cell for which the diffusion coefficient is calculated.
     * @return The calculated diffusion coefficient.
     */
    double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const override;

    /**
     * @brief Calculates the Planck opacity based on the given computational cell properties.
     * @param cell The computational cell for which the Planck opacity is calculated.
     * @return The calculated Planck opacity.
     */
    double CalcPlanckOpacity(ComputationalCell3D const& cell) const override;

    /**
     * @brief Calculates the scattering opacity based on the given computational cell properties.
     * @param cell The computational cell for which the scattering opacity is calculated.
     * @return The calculated scattering opacity.
     */
    double CalcScatteringOpacity(ComputationalCell3D const& cell) const override;
};

#endif // STA_OPACITY_HPP