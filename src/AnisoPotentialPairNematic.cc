// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPair.h"
#include "EvaluatorPairNematic.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

template void
export_AnisoPotentialPair<EvaluatorPairNematic>(pybind11::module& m, const std::string& name);

void export_AnisoPotentialPairNematic(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairNematic>(m, "AnisoPotentialPairNematic");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
