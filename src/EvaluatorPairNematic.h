// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file EvaluatorPairNematic.h
    \brief Defines the nematic pair potential evaluator.

    Potential:
        U = -epsilon * (n_i . n_j)^2 * (1 - r^2/r_c^2)^2

    where n_i = rotate(q_i, x_hat), n_j = rotate(q_j, x_hat) are the
    body-frame x-axes of the two particles, and the smooth compact envelope
    g(r) = (1 - r^2/r_c^2)^2 ensures both force and energy vanish continuously
    at the cutoff r_c.

    The potential is minimized when orientations are parallel OR anti-parallel
    (nematic symmetry).
*/

#ifndef __EVALUATOR_PAIR_NEMATIC_H__
#define __EVALUATOR_PAIR_NEMATIC_H__

#ifndef __HIPCC__
#include <string>
#endif

#ifdef ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {

class EvaluatorPairNematic
    {
    public:
    //! Per-type-pair parameters: just epsilon
    struct param_type
        {
        Scalar epsilon;

#ifdef ENABLE_HIP
        void set_memory_hint() const { }
#endif

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }
        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

        HOSTDEVICE param_type() : epsilon(0) { }

#ifndef __HIPCC__
        param_type(pybind11::dict v, bool managed)
            {
            epsilon = v["epsilon"].cast<Scalar>();
            }

        pybind11::object toPython()
            {
            pybind11::dict v;
            v["epsilon"] = epsilon;
            return v;
            }
#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(4)));
#else
        __attribute__((aligned(8)));
#endif

    //! Nullary shape type (we always use body x-axis, no per-type shape)
    struct shape_type
        {
        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }
        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }
        HOSTDEVICE shape_type() { }

#ifndef __HIPCC__
        shape_type(pybind11::object shape_params, bool managed) { }

        pybind11::object toPython()
            {
            return pybind11::none();
            }
#endif

#ifdef ENABLE_HIP
        void set_memory_hint() const { }
#endif
        };

    //! Constructor
    /*! \param _dr Displacement vector r_i - r_j
        \param _quat_i Quaternion of particle i
        \param _quat_j Quaternion of particle j
        \param _rcutsq Squared cutoff distance
        \param _params Per-type-pair parameters
    */
    HOSTDEVICE EvaluatorPairNematic(const Scalar3& _dr,
                                    const Scalar4& _quat_i,
                                    const Scalar4& _quat_j,
                                    const Scalar _rcutsq,
                                    const param_type& _params)
        : dr(_dr), quat_i(_quat_i), quat_j(_quat_j), rcutsq(_rcutsq),
          epsilon(_params.epsilon)
        {
        }

    HOSTDEVICE static bool needsShape()
        {
        return false;
        }

    HOSTDEVICE static bool needsTags()
        {
        return false;
        }

    HOSTDEVICE static bool needsCharge()
        {
        return false;
        }

    HOSTDEVICE static bool constexpr implementsEnergyShift()
        {
        return false;
        }

    HOSTDEVICE void setShape(const shape_type* shapei, const shape_type* shapej) { }
    HOSTDEVICE void setTags(unsigned int tagi, unsigned int tagj) { }
    HOSTDEVICE void setCharge(Scalar qi, Scalar qj) { }

    //! Evaluate force, energy, and torques
    /*! \param force   Output: force on particle i  (Newton III: force on j = -force)
        \param pair_eng Output: pair energy
        \param energy_shift  Whether to shift energy (not implemented)
        \param torque_i Output: torque on particle i
        \param torque_j Output: torque on particle j
        \return true if within cutoff, false otherwise
    */
    HOSTDEVICE bool evaluate(Scalar3& force,
                             Scalar& pair_eng,
                             bool energy_shift,
                             Scalar3& torque_i,
                             Scalar3& torque_j)
        {
        Scalar rsq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

        if (rsq >= rcutsq || rsq < Scalar(1e-12))
            return false;

        // Body-frame x-axis of each particle, rotated to lab frame
        vec3<Scalar> e_x(1, 0, 0);
        vec3<Scalar> n_i = rotate(quat<Scalar>(quat_i), e_x);
        vec3<Scalar> n_j = rotate(quat<Scalar>(quat_j), e_x);

        // Nematic order parameter: c = n_i . n_j
        Scalar c = dot(n_i, n_j);
        Scalar c2 = c * c;

        // Smooth compact envelope: x = 1 - r^2/r_c^2,  g = x^2
        Scalar x = Scalar(1.0) - rsq / rcutsq;
        Scalar g = x * x;

        // Energy: U = -epsilon * c^2 * g
        pair_eng = -epsilon * c2 * g;

        // --- Force on i (from radial envelope) ---
        // dU/dr_vec = -epsilon * c^2 * dg/dr_vec
        // dg/dr_vec = 2*x * dx/dr_vec = 2*x * (-2/r_c^2) * r_vec
        //           = -4*x / r_c^2 * dr   (since dr = r_i - r_j is the displacement)
        // F_i = -dU/dr_i = epsilon * c^2 * (-4*x / r_c^2) * dr
        // This is attractive (toward j) when x > 0 and c^2 > 0.
        Scalar f_mag = -Scalar(4.0) * epsilon * c2 * x / rcutsq;
        force.x = f_mag * dr.x;
        force.y = f_mag * dr.y;
        force.z = f_mag * dr.z;

        // --- Torques (from orientational dependence) ---
        // dU/dc = -2*epsilon*c*g
        // dU/dn_i = dU/dc * n_j = -2*epsilon*c*g * n_j
        // HOOMD convention (cf. EvaluatorPairYLZ): torque = cross(dU/dn, n)
        // torque_i = cross(-2*eps*c*g * n_j, n_i) = -2*eps*c*g * cross(n_j, n_i)
        //          = 2*eps*c*g * cross(n_i, n_j)
        // torque_j = cross(-2*eps*c*g * n_i, n_j) = -2*eps*c*g * cross(n_i, n_j)
        //          = -torque_i
        Scalar t_prefactor = Scalar(2.0) * epsilon * c * g;
        vec3<Scalar> ni_cross_nj = cross(n_i, n_j);

        torque_i.x = t_prefactor * ni_cross_nj.x;
        torque_i.y = t_prefactor * ni_cross_nj.y;
        torque_i.z = t_prefactor * ni_cross_nj.z;

        torque_j.x = -torque_i.x;
        torque_j.y = -torque_i.y;
        torque_j.z = -torque_i.z;

        return true;
        }

    DEVICE Scalar evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE Scalar evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    static std::string getName()
        {
        return "nematic";
        }

    static std::string getShapeParamName()
        {
        return "Shape";
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for nematic pair potential.");
        }
#endif

    protected:
    Scalar3 dr;
    Scalar4 quat_i, quat_j;
    Scalar rcutsq;
    Scalar epsilon;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_PAIR_NEMATIC_H__
