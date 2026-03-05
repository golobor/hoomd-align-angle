"""Align-angle and nematic pair force plugin for HOOMD-blue.

Provides:

* ``Align`` — an angle force that aligns an oriented particle's body-frame
  x-axis to the direction defined by two guide particles.
* ``NematicPair`` — an anisotropic pair potential that attracts particles
  with parallel or anti-parallel orientations.
"""

from hoomd.md.angle import Angle
from hoomd.md.pair.aniso import AnisotropicPair
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict

from hoomd.align_angle import _align_angle


class Align(Angle):
    r"""Orientation-alignment angle force.

    Args:
        None

    `Align` computes an angular potential that aligns the body-frame x-axis of
    particle *j* with the direction from particle *i* to particle *k*.

    For each angle :math:`(i, j, k)`:

    .. math::

        U = \frac{k}{2} \left(1 - \hat{n} \cdot \hat{d}\right)

    where :math:`\hat{d} = (\mathbf{r}_k - \mathbf{r}_i) / |\mathbf{r}_k -
    \mathbf{r}_i|` is the unit direction from *i* to *k*, and :math:`\hat{n} =
    \mathrm{rotate}(q_j, \hat{x})` is the body-frame x-axis of particle *j*
    rotated into the lab frame by its orientation quaternion :math:`q_j`.

    The force on particle *i* is:

    .. math::

        \mathbf{F}_i = -\frac{k}{2 |\mathbf{d}|}\left(\hat{n} - (\hat{n} \cdot
        \hat{d})\hat{d}\right)

    The force on particle *k* is :math:`\mathbf{F}_k = -\mathbf{F}_i` (Newton's
    third law).  There is no translational force on particle *j*.

    The torque on particle *j* (in the lab frame) is:

    .. math::

        \boldsymbol{\tau}_j = \frac{k}{2}\left(\hat{n} \times \hat{d}\right)

    Example::

        align = align_angle.Align()
        align.params["polymer"] = dict(k=10.0)
        sim.operations.integrator.forces.append(align)

    Attributes:
        params (TypeParameter[``angle type``, dict]):
            The parameter of the align potential for each angle type.
            The dictionary has the following key:

            * ``k`` (`float`, **required**) - spring constant
              :math:`[\mathrm{energy}]`
    """

    _cpp_class_name = "AlignAngleForceCompute"
    _ext_module = _align_angle

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            "params",
            "angle_types",
            TypeParameterDict(k=float, len_keys=1),
        )
        self._add_typeparam(params)


class NematicPair(AnisotropicPair):
    r"""Nematic orientation-dependent pair potential.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting mode (``"none"`` or ``"shift"``).

    `NematicPair` computes an anisotropic pair potential that attracts
    particles whose body-frame x-axes are parallel **or** anti-parallel
    (nematic symmetry):

    .. math::

        U_{ij} = -\epsilon \, (\hat{n}_i \cdot \hat{n}_j)^2
                 \left(1 - \frac{r_{ij}^2}{r_c^2}\right)^2

    where :math:`\hat{n} = \mathrm{rotate}(q, \hat{x})` is the body-frame
    x-axis rotated into the lab frame, and the smooth compact envelope
    :math:`g(r) = (1 - r^2/r_c^2)^2` ensures both force and energy vanish
    continuously at the cutoff :math:`r_c`.

    The potential energy is minimized (most negative) when
    :math:`(\hat{n}_i \cdot \hat{n}_j)^2 = 1` (parallel or anti-parallel) and
    vanishes when orientations are perpendicular.

    Example::

        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        nematic = align_angle.NematicPair(nlist=nlist, default_r_cut=3.0)
        nematic.params[("A", "A")] = dict(epsilon=5.0)
        sim.operations.integrator.forces.append(nematic)

    Attributes:
        params (TypeParameter[``particle types``, dict]):
            The parameters of the nematic potential for each particle type
            pair.  The dictionary has the following key:

            * ``epsilon`` (`float`, **required**) - interaction strength
              :math:`[\mathrm{energy}]`
    """

    _cpp_class_name = "AnisoPotentialPairNematic"
    _ext_module = _align_angle

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(epsilon=float, len_keys=2),
        )
        self._add_typeparam(params)
