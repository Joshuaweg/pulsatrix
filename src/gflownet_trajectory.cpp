#include "pulsatrix/gflownet_trajectory.hpp"

#include <stdexcept>

namespace pulsatrix {

GFlowNetTrajectory sample_gflownet_trajectory(HyperGridEnv&, GFlowNetForwardPolicy&) {
    throw std::logic_error("sample_gflownet_trajectory not yet implemented");
}

}  // namespace pulsatrix
