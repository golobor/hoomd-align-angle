// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "ExternalPatchForceCompute.h"
#include "MixedPrecisionCompat.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

/*! \file ExternalPatchForceCompute.cc
    \brief Contains code for the ExternalPatchForceCompute class.

    Physics
    -------
    For a pair of patched particles (i, k) with directors j = partner(i),
    l = partner(k):

        p_hat_i = normalize(r_j - r_i)    patch direction of i
        p_hat_k = normalize(r_l - r_k)    patch direction of k
        r_hat   = normalize(r_k - r_i)    inter-particle direction

    Sigmoid angular envelope (rescaled to [0,1]):
        f_i = sigma_bar(omega * (p_hat_i . r_hat - cos(alpha)))
        f_k = sigma_bar(omega * (p_hat_k . (-r_hat) - cos(alpha)))

    REPLACED BY cubic Hermite (smoothstep):
        u_lo = 1 - width
        t = clamp((u - u_lo) / width, 0, 1)
        f = 3t^2 - 2t^3
        df/du = 6t(1-t) / width

    Radial potential:
        V(r) = epsilon * (1 - r^2/rc^2)^2

    Total energy:
        U_ik = f_i * f_k * V(r)

    Forces on four particles:
        1) Radial + envelope-position: pair force on i, k (from dV/dr and df/dr_hat)
        2) Patch-direction: forces on (i,j) and (k,l) from df/dp_hat * dp_hat/dr
*/

namespace hoomd
    {
namespace md
    {

// ─── Constructor ─────────────────────────────────────────────────────────────

ExternalPatchForceCompute::ExternalPatchForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist)
    : ForceCompute(sysdef),
      m_nlist(nlist),
      m_partner_tags_size(0),
      m_epsilon(0),
      m_rcutsq(0),
      m_rcut(0),
      m_width(Scalar(0.5)),
      m_typpair_idx(m_pdata->getNTypes())
    {
    m_exec_conf->msg->notice(5) << "Constructing ExternalPatchForceCompute" << endl;

    // Allocate r_cut matrix for the neighbor list (N_types x N_types)
    m_r_cut_nlist
        = std::make_shared<GPUArray<Scalar>>(m_typpair_idx.getNumElements(), m_exec_conf);
    m_nlist->addRCutMatrix(m_r_cut_nlist);

    // Allocate partner tag array (initially empty)
    unsigned int max_tag = m_pdata->getMaximumTag() + 1;
    if (max_tag < 1)
        max_tag = 1;
    m_partner_tags_size = max_tag;
    m_partner_tags = GPUArray<int>(m_partner_tags_size, m_exec_conf);

    // Initialize all to -1 (no partner)
        {
        ArrayHandle<int> h_partners(m_partner_tags, access_location::host, access_mode::overwrite);
        for (unsigned int i = 0; i < m_partner_tags_size; i++)
            h_partners.data[i] = -1;
        }
    }

// ─── Destructor ──────────────────────────────────────────────────────────────

ExternalPatchForceCompute::~ExternalPatchForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ExternalPatchForceCompute" << endl;

    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        }
    }

// ─── Partner assignment ──────────────────────────────────────────────────────

void ExternalPatchForceCompute::setPartners(pybind11::list pairs)
    {
    // Resize if needed
    unsigned int max_tag = m_pdata->getMaximumTag() + 1;
    if (max_tag < 1)
        max_tag = 1;
    if (max_tag > m_partner_tags_size)
        {
        m_partner_tags_size = max_tag;
        m_partner_tags = GPUArray<int>(m_partner_tags_size, m_exec_conf);
        }

    ArrayHandle<int> h_partners(m_partner_tags, access_location::host, access_mode::overwrite);

    // Clear all
    for (unsigned int i = 0; i < m_partner_tags_size; i++)
        h_partners.data[i] = -1;

    // Set from list of (attractor_tag, director_tag) tuples
    for (size_t idx = 0; idx < pybind11::len(pairs); idx++)
        {
        pybind11::tuple pair = pairs[idx];
        unsigned int attractor = pair[0].cast<unsigned int>();
        unsigned int director = pair[1].cast<unsigned int>();

        if (attractor >= m_partner_tags_size)
            {
            throw runtime_error("ExternalPatch: attractor tag " + to_string(attractor)
                                + " exceeds maximum tag");
            }
        if (h_partners.data[attractor] >= 0)
            {
            throw runtime_error("ExternalPatch: attractor tag " + to_string(attractor)
                                + " appears more than once");
            }
        h_partners.data[attractor] = static_cast<int>(director);
        }
    }

pybind11::list ExternalPatchForceCompute::getPartners()
    {
    pybind11::list result;
    ArrayHandle<int> h_partners(m_partner_tags, access_location::host, access_mode::read);
    for (unsigned int i = 0; i < m_partner_tags_size; i++)
        {
        if (h_partners.data[i] >= 0)
            {
            result.append(pybind11::make_tuple(i, (unsigned int)h_partners.data[i]));
            }
        }
    return result;
    }

// ─── Parameters ──────────────────────────────────────────────────────────────

void ExternalPatchForceCompute::setParams(pybind11::dict params)
    {
    if (params.contains("epsilon"))
        m_epsilon = params["epsilon"].cast<Scalar>();
    if (params.contains("width"))
        m_width = params["width"].cast<Scalar>();
    if (params.contains("r_cut"))
        {
        setRCut(params["r_cut"].cast<Scalar>());
        }
    }

pybind11::dict ExternalPatchForceCompute::getParams()
    {
    pybind11::dict d;
    d["epsilon"] = m_epsilon;
    d["width"] = m_width;
    d["r_cut"] = m_rcut;
    return d;
    }

void ExternalPatchForceCompute::setRCut(Scalar r_cut)
    {
    m_rcut = r_cut;
    m_rcutsq = r_cut * r_cut;
    updateRCutMatrix();
    }

Scalar ExternalPatchForceCompute::getRCut()
    {
    return m_rcut;
    }

void ExternalPatchForceCompute::updateRCutMatrix()
    {
    // Fill every type-pair entry with the global r_cut
    ArrayHandle<Scalar> h_r_cut(*m_r_cut_nlist, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_typpair_idx.getNumElements(); i++)
        h_r_cut.data[i] = m_rcut;
    m_nlist->notifyRCutMatrixChange();
    }

// ─── Force computation ──────────────────────────────────────────────────────

void ExternalPatchForceCompute::computeForces(uint64_t timestep)
    {
    // Make sure neighbor list is up to date
    m_nlist->compute(timestep);

    // Access particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // Access neighbor list
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    // Access partner tags
    ArrayHandle<int> h_partners(m_partner_tags, access_location::host, access_mode::read);

    // Access force output (overwrite — we accumulate from scratch)
    ArrayHandle<ForceReal4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<ForceReal> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // Zero output arrays
    memset(h_force.data, 0, sizeof(ForceReal4) * m_force.getNumElements());
    memset(h_virial.data, 0, sizeof(ForceReal) * m_virial.getNumElements());

    const BoxDim& box = m_pdata->getGlobalBox();
    const unsigned int N_local = m_pdata->getN(); // local particles only
    const Scalar rcutsq = m_rcutsq;
    const Scalar epsilon = m_epsilon;
    const Scalar width = m_width;

    // Guard distance squared for degenerate director distances
    const Scalar guard_dsq = Scalar(1e-24);

    // Precompute Hermite envelope constants
    // Transition from u_lo = 1 - width to u_hi = 1
    const Scalar u_lo = Scalar(1) - width;
    const Scalar w_inv = Scalar(1) / width;

    // ── Main loop over local particles ──
    for (unsigned int idx_i = 0; idx_i < N_local; idx_i++)
        {
        unsigned int tag_i = h_tag.data[idx_i];

        // Check if particle i has a partner
        if (tag_i >= m_partner_tags_size)
            continue;
        int partner_j_tag = h_partners.data[tag_i];
        if (partner_j_tag < 0)
            continue;

        // Look up partner j's local index
        unsigned int idx_j = h_rtag.data[(unsigned int)partner_j_tag];
        if (idx_j == NOT_LOCAL)
            continue; // partner j not accessible (shouldn't happen if ghost width is sufficient)

        // Particle i position
        Scalar3 pos_i = make_scalar3(h_pos.data[idx_i].x,
                                      h_pos.data[idx_i].y,
                                      h_pos.data[idx_i].z);
        // Partner j position
        Scalar3 pos_j = make_scalar3(h_pos.data[idx_j].x,
                                      h_pos.data[idx_j].y,
                                      h_pos.data[idx_j].z);

        // Compute patch direction: p_hat_i = (r_i - r_j) / |r_i - r_j|  (outward)
        Scalar3 d_ij;
        d_ij.x = pos_i.x - pos_j.x;
        d_ij.y = pos_i.y - pos_j.y;
        d_ij.z = pos_i.z - pos_j.z;
        d_ij = box.minImage(d_ij);

        Scalar d_ij_sq = d_ij.x * d_ij.x + d_ij.y * d_ij.y + d_ij.z * d_ij.z;
        if (d_ij_sq < guard_dsq)
            continue; // degenerate — skip

        Scalar d_ij_inv = fast::rsqrt(d_ij_sq);
        vec3<Scalar> p_hat_i(d_ij.x * d_ij_inv, d_ij.y * d_ij_inv, d_ij.z * d_ij_inv);
        Scalar d_ij_mag = Scalar(1) / d_ij_inv;

        // ── Neighbor loop ──
        const unsigned int n_neigh_i = h_n_neigh.data[idx_i];
        const size_t head_i = h_head_list.data[idx_i];

        for (unsigned int neigh_idx = 0; neigh_idx < n_neigh_i; neigh_idx++)
            {
            unsigned int idx_k = h_nlist.data[head_i + neigh_idx];
            unsigned int tag_k = h_tag.data[idx_k];

            // Check if neighbor k has a partner
            if (tag_k >= m_partner_tags_size)
                continue;
            int partner_l_tag = h_partners.data[tag_k];
            if (partner_l_tag < 0)
                continue;

            // Look up partner l's local index
            unsigned int idx_l = h_rtag.data[(unsigned int)partner_l_tag];
            if (idx_l == NOT_LOCAL)
                continue;

            // Positions
            Scalar3 pos_k = make_scalar3(h_pos.data[idx_k].x,
                                          h_pos.data[idx_k].y,
                                          h_pos.data[idx_k].z);

            // Inter-particle displacement dr = r_k - r_i (with minimum image)
            Scalar3 dr;
            dr.x = pos_k.x - pos_i.x;
            dr.y = pos_k.y - pos_i.y;
            dr.z = pos_k.z - pos_i.z;
            dr = box.minImage(dr);

            Scalar rsq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
            if (rsq >= rcutsq || rsq < guard_dsq)
                continue;

            // Partner l position
            Scalar3 pos_l = make_scalar3(h_pos.data[idx_l].x,
                                          h_pos.data[idx_l].y,
                                          h_pos.data[idx_l].z);

            // Patch direction of k: p_hat_k = (r_k - r_l) / |r_k - r_l|  (outward)
            Scalar3 d_kl;
            d_kl.x = pos_k.x - pos_l.x;
            d_kl.y = pos_k.y - pos_l.y;
            d_kl.z = pos_k.z - pos_l.z;
            d_kl = box.minImage(d_kl);

            Scalar d_kl_sq = d_kl.x * d_kl.x + d_kl.y * d_kl.y + d_kl.z * d_kl.z;
            if (d_kl_sq < guard_dsq)
                continue;

            Scalar d_kl_inv = fast::rsqrt(d_kl_sq);
            vec3<Scalar> p_hat_k(d_kl.x * d_kl_inv, d_kl.y * d_kl_inv, d_kl.z * d_kl_inv);

            // ─── Compute interaction ─────────────────────────────────

            Scalar r_inv = fast::rsqrt(rsq);
            Scalar r_mag = Scalar(1) / r_inv;
            vec3<Scalar> r_hat(dr.x * r_inv, dr.y * r_inv, dr.z * r_inv);

            // --- Radial potential ---
            // V(r) = -epsilon * (1 - r^2/rc^2)^2   (attractive)
            Scalar x = Scalar(1) - rsq / rcutsq; // (1 - r^2/rc^2)
            Scalar Vr = -epsilon * x * x;

            // dV/dr = -epsilon * 2 * (1 - r^2/rc^2) * (-2r/rc^2)
            //       = 4 * epsilon * r * x / rc^2
            Scalar dVdr = Scalar(4) * epsilon * r_mag * x / rcutsq;

            // --- Angular envelope for particle i ---
            // u_i = p_hat_i . r_hat  (cosine of angle between patch and displacement)
            Scalar u_i = dot(p_hat_i, r_hat);

            // Cubic Hermite (smoothstep): t = clamp((u - u_lo) / width, 0, 1)
            Scalar t_i = (u_i - u_lo) * w_inv;
            t_i = (t_i < Scalar(0)) ? Scalar(0) : ((t_i > Scalar(1)) ? Scalar(1) : t_i);
            Scalar fi = t_i * t_i * (Scalar(3) - Scalar(2) * t_i);
            // df/du = 6*t*(1-t) / width
            Scalar dfi_du = Scalar(6) * t_i * (Scalar(1) - t_i) * w_inv;

            // --- Angular envelope for particle k ---
            // u_k = p_hat_k . (-r_hat)  (k's patch faces toward i)
            Scalar u_k = -dot(p_hat_k, r_hat);

            Scalar t_k = (u_k - u_lo) * w_inv;
            t_k = (t_k < Scalar(0)) ? Scalar(0) : ((t_k > Scalar(1)) ? Scalar(1) : t_k);
            Scalar fk = t_k * t_k * (Scalar(3) - Scalar(2) * t_k);
            Scalar dfk_du = Scalar(6) * t_k * (Scalar(1) - t_k) * w_inv;

            // --- Total energy ---
            Scalar pair_eng = fi * fk * Vr;

            // ─── Force computation ───────────────────────────────────

            // Channel 1: Radial force from V'(r)
            // F_i = fi * fk * dV/dr * r̂
            // (positive sign because ∂r/∂(dr) = r̂ and F_i = +∂U/∂(dr))
            vec3<Scalar> F_radial = fi * fk * dVdr * r_hat;

            // Channel 2: Envelope-position gradient (force from df/dr_hat)
            // Using the quotient rule for u = dot(dr, n) / |dr|:
            //   du/dr = (n - u * r_hat) / r
            // where n = p_hat for fi or n = -p_hat for fk.

            // For fi: dfi/dr = dfi/du_i * du_i/dr
            //   u_i = dot(p_hat_i, r_hat) = dot(p_hat_i, dr) / r
            //   du_i/dr = (p_hat_i - u_i * r_hat) / r
            vec3<Scalar> dui_dr = (p_hat_i - u_i * r_hat) * r_inv;

            // For fk: dfk/dr = dfk/du_k * du_k/dr
            //   u_k = -dot(p_hat_k, r_hat) = -dot(p_hat_k, dr) / r
            //   du_k/dr = -(p_hat_k + u_k * r_hat) / r
            //   Note: u_k = -p_hat_k.r_hat, so p_hat_k + u_k*r_hat means:
            //   du_k/dr = (-p_hat_k - u_k*r_hat) / r  ... let's be careful.
            //   u_k = (-p_hat_k . dr) / r
            //   du_k/dr = (-p_hat_k * r - (-p_hat_k . dr) * r_hat) / r^2
            //           = (-p_hat_k - u_k * r_hat) / r    [using u_k = (-p_hat_k.dr)/r]
            // Wait, let me redo this:
            //   u_k = -dot(p_hat_k, dr) / r
            //   let hi = -dot(p_hat_k, dr), lo = r
            //   du_k/dr = (lo * dhi/dr - hi * dlo/dr) / lo^2
            //   dhi/dr = -p_hat_k, dlo/dr = r_hat
            //   du_k/dr = (r * (-p_hat_k) - (-dot(p_hat_k,dr)) * r_hat) / r^2
            //           = (-p_hat_k + dot(p_hat_k, r_hat) * r_hat) / r
            //           = (-p_hat_k - u_k * r_hat) / r   [since u_k = -dot(p_hat_k,r_hat)]
            // Hmm wait: u_k = -dot(p_hat_k, r_hat), so -u_k = dot(p_hat_k, r_hat)
            //   du_k/dr = (-p_hat_k + (-u_k) * r_hat) / r  ✓
            vec3<Scalar> duk_dr = (-p_hat_k - u_k * r_hat) * r_inv;

            // Envelope force on i (from dependence of fi, fk on dr):
            //   F_env_on_i = Vr * (dfi/du * du_i/d(dr) * fk + dfk/du * du_k/d(dr) * fi)
            // (positive sign: F_i = +∂U/∂(dr) since ∂(dr)/∂r_i = -I)
            vec3<Scalar> F_env = Vr * (dfi_du * dui_dr * fk + dfk_du * duk_dr * fi);

            // Total pair force on i from channels 1 + 2:
            vec3<Scalar> F_pair_on_i = F_radial + F_env;

            // Channel 3: Patch-direction gradient (force on partner j)
            //
            //   F_j_patch = +fk * V(r) * dfi/du_i * (r_hat - u_i * p_hat_i) / d_ij_mag
            //
            // Forces on k and l are NOT computed here -- they are computed
            // when k runs as the central particle in its own loop iteration.

            vec3<Scalar> perp_i = r_hat - u_i * p_hat_i;
            vec3<Scalar> F_j_patch = fk * Vr * dfi_du * perp_i * (Scalar(1) / d_ij_mag);
            vec3<Scalar> F_i_patch = -F_j_patch;

            // ─── Accumulate (full-weight forces, half-weight energy) ───
            //
            // Each iteration computes forces only on central particle i
            // and its partner j. The reverse iteration (k as central)
            // handles k's and l's forces identically.
            // Energy uses factor 0.5 (standard full-nlist convention).

            vec3<Scalar> F_i_total = F_pair_on_i + F_i_patch;
            h_force.data[idx_i].x += F_i_total.x;
            h_force.data[idx_i].y += F_i_total.y;
            h_force.data[idx_i].z += F_i_total.z;
            h_force.data[idx_i].w += Scalar(0.5) * pair_eng;

            // Force on partner j (may be ghost — only accumulate if local)
            if (idx_j < N_local)
                {
                h_force.data[idx_j].x += F_j_patch.x;
                h_force.data[idx_j].y += F_j_patch.y;
                h_force.data[idx_j].z += F_j_patch.z;
                }

            // Note: virial = 0 for now (NVT/NVE only), NPT support is a follow-up
            } // end neighbor loop
        } // end particle loop
    }

// ─── pybind11 export ─────────────────────────────────────────────────────────

namespace detail
    {
void export_ExternalPatchForceCompute(pybind11::module& m)
    {
    pybind11::class_<ExternalPatchForceCompute,
                     ForceCompute,
                     std::shared_ptr<ExternalPatchForceCompute>>(m,
                                                                  "ExternalPatchForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>())
        .def("setPartners", &ExternalPatchForceCompute::setPartners)
        .def("getPartners", &ExternalPatchForceCompute::getPartners)
        .def("setParams", &ExternalPatchForceCompute::setParams)
        .def("getParams", &ExternalPatchForceCompute::getParams)
        .def("setRCut", &ExternalPatchForceCompute::setRCut)
        .def("getRCut", &ExternalPatchForceCompute::getRCut);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
