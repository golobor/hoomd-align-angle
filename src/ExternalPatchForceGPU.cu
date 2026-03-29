// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"

#include "ExternalPatchForceGPU.cuh"
#include "hoomd/VectorMath.h"

#include <assert.h>

/*! \file ExternalPatchForceGPU.cu
    \brief Defines the GPU kernel for computing external-patch forces.

    Physics (four-body interaction for patched particles i, k with
    directors j = partner(i), l = partner(k)):

        p_hat_i = normalize(r_j - r_i)        patch direction of i
        p_hat_k = normalize(r_l - r_k)        patch direction of k
        r_hat   = normalize(r_k - r_i)        inter-particle direction

        f_i = sigma_bar(omega * (p_hat_i . r_hat - cos(alpha)))
        f_k = sigma_bar(omega * (p_hat_k . (-r_hat) - cos(alpha)))

    REPLACED BY cubic Hermite (smoothstep):
        u_lo = 1 - width
        t = clamp((u - u_lo) / width, 0, 1)
        f = 3t^2 - 2t^3
        V(r) = epsilon * (1 - r^2/rc^2)^2
        U_ik = f_i * f_k * V(r)

    Forces on four particles (i, j, k, l) via the chain rule:
      1) Radial: from dV/dr
      2) Envelope-position: from df/d(r_hat)
      3) Patch-direction: from df/d(p_hat) * dp_hat/dr

    GPU parallelization: one thread per local particle (the "i" role).
    Each thread computes forces only on its own particle i and partner j.
    Forces on neighbor k and partner l are computed by k's own thread
    when it processes the same pair from the reverse direction.
    Forces on i and j are accumulated in registers, then written via
    atomicAdd at the end of the neighbor loop (no per-neighbor atomics).
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

// ─── atomicAdd wrappers for portability ──────────────────────────────────────

#if HOOMD_LONGREAL_SIZE == 64
#if (__CUDA_ARCH__ < 600)
__device__ inline double myAtomicAdd(double* address, double val)
    {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
        {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
    return __longlong_as_double(old);
    }
#else
__device__ inline double myAtomicAdd(double* address, double val)
    {
    return atomicAdd(address, val);
    }
#endif
#endif

#ifdef __HIP_PLATFORM_HCC__
__device__ inline float myAtomicAdd(float* address, float val)
    {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;
    do
        {
        assumed = old;
        old = atomicCAS(address_as_uint,
                        assumed,
                        __float_as_uint(val + __uint_as_float(assumed)));
        } while (assumed != old);
    return __uint_as_float(old);
    }
#else
__device__ inline float myAtomicAdd(float* address, float val)
    {
    return atomicAdd(address, val);
    }
#endif

// ─── Sentinel for "particle not found" ──────────────────────────────────────

static constexpr unsigned int NOT_LOCAL_GPU = 0xFFFFFFFF;

// ─── Small kernel to zero force & virial arrays (replaces hipMemset) ────────

__global__ void gpu_zero_force_virial_kernel(ForceReal4* d_force,
                                              ForceReal* d_virial,
                                              const size_t virial_pitch,
                                              const unsigned int force_size)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < force_size)
        d_force[idx] = make_forcereal4(ForceReal(0), ForceReal(0),
                                        ForceReal(0), ForceReal(0));
    if (idx < virial_pitch)
        {
        d_virial[0 * virial_pitch + idx] = ForceReal(0);
        d_virial[1 * virial_pitch + idx] = ForceReal(0);
        d_virial[2 * virial_pitch + idx] = ForceReal(0);
        d_virial[3 * virial_pitch + idx] = ForceReal(0);
        d_virial[4 * virial_pitch + idx] = ForceReal(0);
        d_virial[5 * virial_pitch + idx] = ForceReal(0);
        }
    }

// ─── GPU kernel ──────────────────────────────────────────────────────────────

__global__ void gpu_compute_external_patch_forces_kernel(
    ForceReal4* d_force,
    ForceReal* d_virial,
    const size_t virial_pitch,
    const unsigned int N,
    const ForceReal4* d_pos,
    const unsigned int* d_tag,
    const unsigned int* d_rtag,
    const int* d_partner_tags,
    const unsigned int partner_tags_size,
    const unsigned int* d_n_neigh,
    const unsigned int* d_nlist,
    const size_t* d_head_list,
    BoxDim box,
    const ForceReal epsilon,
    const ForceReal rcutsq,
    const ForceReal width)
    {
    unsigned int idx_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_i >= N)
        return;

    // ── Check if particle i has a partner ──
    unsigned int tag_i = d_tag[idx_i];
    if (tag_i >= partner_tags_size)
        return;
    int partner_j_tag = d_partner_tags[tag_i];
    if (partner_j_tag < 0)
        return;

    unsigned int idx_j = d_rtag[(unsigned int)partner_j_tag];
    if (idx_j == NOT_LOCAL_GPU)
        return;

    // ── Load positions of i and partner j ──
    ForceReal4 pos_i_data = d_pos[idx_i];
    ForceReal3 pos_i = make_forcereal3(pos_i_data.x, pos_i_data.y, pos_i_data.z);

    ForceReal4 pos_j_data = d_pos[idx_j];
    ForceReal3 pos_j = make_forcereal3(pos_j_data.x, pos_j_data.y, pos_j_data.z);

    // ── Patch direction p_hat_i = (r_i - r_j) / |r_i - r_j| (outward) ──
    ForceReal3 d_ij;
    d_ij.x = pos_i.x - pos_j.x;
    d_ij.y = pos_i.y - pos_j.y;
    d_ij.z = pos_i.z - pos_j.z;
#ifdef HOOMD_HAS_FORCEREAL
    d_ij = box.minImageForceReal(d_ij);
#else
    d_ij = box.minImage(d_ij);
#endif

    ForceReal d_ij_sq = d_ij.x * d_ij.x + d_ij.y * d_ij.y + d_ij.z * d_ij.z;

    const ForceReal guard_dsq = ForceReal(1e-12);
    if (d_ij_sq < guard_dsq)
        return;

    ForceReal d_ij_inv = rsqrtf(d_ij_sq);
    vec3<ForceReal> p_hat_i(d_ij.x * d_ij_inv, d_ij.y * d_ij_inv, d_ij.z * d_ij_inv);
    ForceReal d_ij_mag = ForceReal(1.0) / d_ij_inv;

    // ── Hermite envelope constants ──
    const ForceReal u_lo = ForceReal(1.0) - width;
    const ForceReal w_inv = ForceReal(1.0) / width;

    // ── Register accumulators for particle i and partner j ──
    ForceReal4 force_i = make_forcereal4(ForceReal(0.0),
                                          ForceReal(0.0),
                                          ForceReal(0.0),
                                          ForceReal(0.0));
    ForceReal3 force_j = make_forcereal3(ForceReal(0.0),
                                          ForceReal(0.0),
                                          ForceReal(0.0));

    const ForceReal half = ForceReal(0.5);

    // ── Neighbor loop ──
    unsigned int n_neigh = d_n_neigh[idx_i];
    size_t head = d_head_list[idx_i];

    for (unsigned int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
        {
        unsigned int idx_k = d_nlist[head + neigh_idx];
        unsigned int tag_k = d_tag[idx_k];

        // Check if neighbor k has a partner
        if (tag_k >= partner_tags_size)
            continue;
        int partner_l_tag = d_partner_tags[tag_k];
        if (partner_l_tag < 0)
            continue;

        unsigned int idx_l = d_rtag[(unsigned int)partner_l_tag];
        if (idx_l == NOT_LOCAL_GPU)
            continue;

        // Position of k
        ForceReal4 pos_k_data = d_pos[idx_k];
        ForceReal3 pos_k = make_forcereal3(pos_k_data.x, pos_k_data.y, pos_k_data.z);

        // Separation dr = r_k - r_i
        ForceReal3 dr;
        dr.x = pos_k.x - pos_i.x;
        dr.y = pos_k.y - pos_i.y;
        dr.z = pos_k.z - pos_i.z;
#ifdef HOOMD_HAS_FORCEREAL
        dr = box.minImageForceReal(dr);
#else
        dr = box.minImage(dr);
#endif

        ForceReal rsq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
        if (rsq >= rcutsq || rsq < guard_dsq)
            continue;

        // Partner l position
        ForceReal4 pos_l_data = d_pos[idx_l];
        ForceReal3 pos_l = make_forcereal3(pos_l_data.x, pos_l_data.y, pos_l_data.z);

        // Patch direction p_hat_k = (r_k - r_l) / |r_k - r_l| (outward)
        ForceReal3 d_kl;
        d_kl.x = pos_k.x - pos_l.x;
        d_kl.y = pos_k.y - pos_l.y;
        d_kl.z = pos_k.z - pos_l.z;
#ifdef HOOMD_HAS_FORCEREAL
        d_kl = box.minImageForceReal(d_kl);
#else
        d_kl = box.minImage(d_kl);
#endif

        ForceReal d_kl_sq = d_kl.x * d_kl.x + d_kl.y * d_kl.y + d_kl.z * d_kl.z;
        if (d_kl_sq < guard_dsq)
            continue;

        ForceReal d_kl_inv = rsqrtf(d_kl_sq);
        vec3<ForceReal> p_hat_k(d_kl.x * d_kl_inv, d_kl.y * d_kl_inv, d_kl.z * d_kl_inv);

        // ─── Compute interaction ─────────────────────────────────
        ForceReal r_inv = rsqrtf(rsq);
        ForceReal r_mag = ForceReal(1.0) / r_inv;
        vec3<ForceReal> r_hat(dr.x * r_inv, dr.y * r_inv, dr.z * r_inv);

        // Radial potential V(r) = -epsilon * (1 - r^2/rc^2)^2  (attractive)
        ForceReal x = ForceReal(1.0) - rsq / rcutsq;
        ForceReal Vr = -epsilon * x * x;
        ForceReal dVdr = ForceReal(4.0) * epsilon * r_mag * x / rcutsq;

        // Angular envelope for particle i
        ForceReal u_i = dot(p_hat_i, r_hat);
        ForceReal t_i = (u_i - u_lo) * w_inv;
        t_i = (t_i < ForceReal(0.0)) ? ForceReal(0.0) : ((t_i > ForceReal(1.0)) ? ForceReal(1.0) : t_i);
        ForceReal fi = t_i * t_i * (ForceReal(3.0) - ForceReal(2.0) * t_i);
        ForceReal dfi_du = ForceReal(6.0) * t_i * (ForceReal(1.0) - t_i) * w_inv;

        // Angular envelope for particle k
        ForceReal u_k = -dot(p_hat_k, r_hat);
        ForceReal t_k = (u_k - u_lo) * w_inv;
        t_k = (t_k < ForceReal(0.0)) ? ForceReal(0.0) : ((t_k > ForceReal(1.0)) ? ForceReal(1.0) : t_k);
        ForceReal fk = t_k * t_k * (ForceReal(3.0) - ForceReal(2.0) * t_k);
        ForceReal dfk_du = ForceReal(6.0) * t_k * (ForceReal(1.0) - t_k) * w_inv;

        // Energy
        ForceReal pair_eng = fi * fk * Vr;

        // ─── Force computation ───────────────────────────────────

        // Channel 1: Radial force from V'(r)
        vec3<ForceReal> F_radial = fi * fk * dVdr * r_hat;

        // Channel 2: Envelope-position gradient
        vec3<ForceReal> dui_dr = (p_hat_i - u_i * r_hat) * r_inv;
        vec3<ForceReal> duk_dr = (-p_hat_k - u_k * r_hat) * r_inv;
        vec3<ForceReal> F_env = Vr * (dfi_du * dui_dr * fk + dfk_du * duk_dr * fi);

        vec3<ForceReal> F_pair_on_i = F_radial + F_env;

        // Channel 3: Patch-direction gradient
        //   Force on partner j from i's envelope's dependence on patch direction.
        //   Forces on k and l are NOT computed here — thread-k handles them
        //   when it processes this pair from the reverse direction.
        vec3<ForceReal> perp_i = r_hat - u_i * p_hat_i;
        vec3<ForceReal> F_j_patch
            = fk * Vr * dfi_du * perp_i * (ForceReal(1.0) / d_ij_mag);
        vec3<ForceReal> F_i_patch = -F_j_patch;

        // ─── Accumulate (full-weight forces, half-weight energy) ─
        //
        // Each thread writes forces only to particle i and partner j.
        // Thread-k computes k's and l's forces when it runs as central.
        // Energy uses factor 0.5 (standard full-nlist convention).

        vec3<ForceReal> F_i_total = F_pair_on_i + F_i_patch;
        force_i.x += F_i_total.x;
        force_i.y += F_i_total.y;
        force_i.z += F_i_total.z;
        force_i.w += half * pair_eng;

        force_j.x += F_j_patch.x;
        force_j.y += F_j_patch.y;
        force_j.z += F_j_patch.z;
        } // end neighbor loop

    // ── Write register-accumulated forces via atomicAdd ──────────
    // (other threads may also contribute to these particles)
    myAtomicAdd(&d_force[idx_i].x, force_i.x);
    myAtomicAdd(&d_force[idx_i].y, force_i.y);
    myAtomicAdd(&d_force[idx_i].z, force_i.z);
    myAtomicAdd(&d_force[idx_i].w, force_i.w);

    if (idx_j < N)
        {
        myAtomicAdd(&d_force[idx_j].x, force_j.x);
        myAtomicAdd(&d_force[idx_j].y, force_j.y);
        myAtomicAdd(&d_force[idx_j].z, force_j.z);
        }
    }

// ─── Kernel driver ───────────────────────────────────────────────────────────

hipError_t gpu_compute_external_patch_forces(ForceReal4* d_force,
                                             ForceReal* d_virial,
                                             const size_t virial_pitch,
                                             const unsigned int N,
                                             const unsigned int Nghosts,
                                             const ForceReal4* d_pos,
                                             const unsigned int* d_tag,
                                             const unsigned int* d_rtag,
                                             const int* d_partner_tags,
                                             const unsigned int partner_tags_size,
                                             const unsigned int* d_n_neigh,
                                             const unsigned int* d_nlist,
                                             const size_t* d_head_list,
                                             const BoxDim& box,
                                             const ForceReal epsilon,
                                             const ForceReal rcutsq,
                                             const ForceReal width,
                                             const int block_size)
    {
    assert(d_force);

    // Zero force and virial arrays via a small kernel (avoids hipMemset overhead)
    {
    unsigned int total = N + Nghosts;
    unsigned int zero_n = total > (unsigned int)virial_pitch
                              ? total
                              : (unsigned int)virial_pitch;
    dim3 zero_grid((zero_n + 255) / 256, 1, 1);
    dim3 zero_threads(256, 1, 1);
    hipLaunchKernelGGL((gpu_zero_force_virial_kernel),
                       dim3(zero_grid),
                       dim3(zero_threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       total);
    }

    if (N == 0)
        return hipSuccess;

    // Determine block size
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr,
                         (const void*)gpu_compute_external_patch_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min((unsigned int)block_size, max_block_size);

    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_compute_external_patch_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       d_tag,
                       d_rtag,
                       d_partner_tags,
                       partner_tags_size,
                       d_n_neigh,
                       d_nlist,
                       d_head_list,
                       box,
                       epsilon,
                       rcutsq,
                       width);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
