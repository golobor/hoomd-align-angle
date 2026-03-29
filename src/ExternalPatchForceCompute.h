// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#pragma once

#include "hoomd/ForceCompute.h"
#include "hoomd/GPUArray.h"
#include "hoomd/Index1D.h"
#include "hoomd/VectorMath.h"
#include "hoomd/md/NeighborList.h"

#include <memory>
#include <vector>

/*! \file ExternalPatchForceCompute.h
    \brief Declares a class for computing patch interactions with externally
           defined patch directions.

    Each designated particle i has a virtual "patch" whose direction is the unit
    vector from i toward its partner j. When two patched particles i and k
    approach within r_cut, they interact via:

        U_ik = f_i * f_k * epsilon * (1 - r^2/rc^2)^2

    where f_i is a cubic Hermite (smoothstep) angular envelope:

        t = clamp((u - (1 - width)) / width, 0, 1)
        f = 3t^2 - 2t^3

    with u = p_hat_i . r_hat_ik and
    p_hat_i = (r_i - r_j) / |r_i - r_j| (outward from partner).

    This is a four-body interaction: forces are distributed to particles
    i, j, k, l (where j = partner(i), l = partner(k)) via the chain rule.

    No quaternion DOFs are required — the "torque" on the patch direction
    manifests as non-central translational forces on i and j (and on k and l).
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {

//! Computes external-patch interactions
/*! \ingroup computes
*/
class PYBIND11_EXPORT ExternalPatchForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    ExternalPatchForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<NeighborList> nlist);

    //! Destructor
    virtual ~ExternalPatchForceCompute();

    //! Set the partner assignments from Python
    void setPartners(pybind11::list pairs);

    //! Get the partner assignments as a Python list
    pybind11::list getPartners();

    //! Set interaction parameters from Python dict
    void setParams(pybind11::dict params);

    //! Get interaction parameters as Python dict
    pybind11::dict getParams();

    //! Set rcut and update the neighbor list
    void setRCut(Scalar r_cut);

    //! Get rcut
    Scalar getRCut();

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this force
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    /// Notify that we are being detached from the Simulation
    virtual void notifyDetach()
        {
        if (m_attached)
            {
            m_nlist->removeRCutMatrix(m_r_cut_nlist);
            }
        m_attached = false;
        }

    protected:
    std::shared_ptr<NeighborList> m_nlist; //!< The neighbor list
    std::shared_ptr<GPUArray<Scalar>> m_r_cut_nlist; //!< r_cut matrix for nlist

    //! Per-tag partner map: partner_tag[tag_i] = tag_j, or -1 if no partner
    GPUArray<int> m_partner_tags;
    unsigned int m_partner_tags_size; //!< Current allocated size

    //! Interaction parameters
    Scalar m_epsilon;    //!< Attraction strength
    Scalar m_rcutsq;     //!< Cutoff squared
    Scalar m_rcut;       //!< Cutoff radius
    Scalar m_width;      //!< Hermite transition width in cosine space

    //! Type-pair indexer
    Index2D m_typpair_idx;

    //! Track whether we are attached (for r_cut cleanup)
    bool m_attached = true;

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    private:
    //! Fill the r_cut nlist matrix uniformly with the current m_rcut
    void updateRCutMatrix();
    };

namespace detail
    {
void export_ExternalPatchForceCompute(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
