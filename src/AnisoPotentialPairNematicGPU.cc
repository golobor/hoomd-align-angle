// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPairGPU.h"
#include "EvaluatorPairNematic.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_AnisoPotentialPairNematicGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairNematic>(m, "AnisoPotentialPairNematicGPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
