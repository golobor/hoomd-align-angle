"""Align angle force plugin for HOOMD-blue.

Provides ``Align``, an angle force that aligns an oriented particle's body-frame
x-axis to the direction defined by two guide particles.
"""

from hoomd.md.angle import Angle
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
