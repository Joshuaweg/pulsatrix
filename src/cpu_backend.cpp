#include "exai/cpu_backend.hpp"

// Implementations deliberately not yet written -- see
// how/campaigns/campaign_exai_dl_library_phase0_tensor_autograd_core/missions/mission_devicebackend_cpu.md
// Objective 3. This file exists (rather than being created alongside the .hpp) so the build
// itself proves the RED step of Objective 2/3's Red-Green-Refactor cycle: cpu_backend_test.cpp
// fails to link against these not-yet-defined symbols until Objective 3 fills them in.
