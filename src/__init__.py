"""Director-alignment forces plugin for HOOMD-blue.

Provides:

* ``DirectorAlign`` — an angle force that aligns an oriented particle's
  body-frame x-axis to the direction defined by two guide particles.
* ``DirectorPair`` — an anisotropic pair potential that attracts particles
  with parallel or anti-parallel orientations.
* ``ExternalPatch`` — a patch interaction with externally defined patch
  directions (no quaternion DOFs required).
"""

import copy

import hoomd
from hoomd.md.angle import Angle
from hoomd.md.dihedral import Dihedral
from hoomd.md.force import Force
from hoomd.md.pair.aniso import AnisotropicPair
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict, ParameterDict
from hoomd.md import _md

from hoomd.align_angle import _align_angle


class DirectorAlign(Angle):
    r"""Orientation-alignment angle force.

    Args:
        None

    `DirectorAlign` computes an angular potential that aligns the body-frame
    x-axis of particle *i* with the direction from particle *j* to particle *k*.

    For each angle :math:`(i, j, k)`:

    .. math::

        U = \frac{k}{2}
            \left(1 - \cos\!\bigl(m\,\theta + \varphi_0\bigr)\right)

    where :math:`\theta = \arccos(\hat{n} \cdot \hat{d})` is the angle between
    the director and the target direction,
    :math:`\hat{d} = (\mathbf{r}_k - \mathbf{r}_j) / |\mathbf{r}_k -
    \mathbf{r}_j|`, :math:`\hat{n} = \mathrm{rotate}(q_i, \hat{x})`,
    :math:`m` is the ``multiplicity``, and :math:`\varphi_0` is the ``phase``.

    With ``multiplicity=1, phase=0`` (the defaults) this reduces to
    :math:`U = \frac{k}{2}(1 - \hat{n}\cdot\hat{d})`, aligning the director
    parallel to the target direction.

    Useful presets:

    * ``multiplicity=2, phase=0`` — nematic alignment (:math:`\hat{n}` or
      :math:`-\hat{n}` both minimise the energy).
    * ``multiplicity=1, phase=pi`` — anti-parallel alignment.

    Example::

        align = align_angle.DirectorAlign()
        align.params["polymer"] = dict(k=10.0)
        sim.operations.integrator.forces.append(align)

        # Nematic (head-tail symmetric) alignment:
        align.params["polymer"] = dict(k=10.0, multiplicity=2)

    Attributes:
        params (TypeParameter[``angle type``, dict]):
            The parameter of the align potential for each angle type.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - spring constant
              :math:`[\mathrm{energy}]`
            * ``multiplicity`` (`int`, **optional**, default 1) - angular
              multiplicity :math:`m`
            * ``phase`` (`float`, **optional**, default 0) - phase offset
              :math:`\varphi_0` in radians
    """

    _cpp_class_name = "AlignAngleForceCompute"
    _ext_module = _align_angle

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            "params",
            "angle_types",
            TypeParameterDict(k=float, multiplicity=1, phase=0.0, len_keys=1),
        )
        self._add_typeparam(params)


class DirectorPair(AnisotropicPair):
    r"""Director orientation-dependent pair potential.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): Energy shifting mode (``"none"`` or ``"shift"``).

    `DirectorPair` computes an anisotropic pair potential whose orientational
    coupling is controlled by the ``multiplicity`` and ``phase`` parameters:

    .. math::

        U_{ij} = -\epsilon \, \cos(m\,\alpha_{ij} + \varphi_0)
                 \left(1 - \frac{r_{ij}^2}{r_c^2}\right)^2

    where :math:`\alpha_{ij} = \arccos(\hat{n}_i \cdot \hat{n}_j)` is the
    angle between directors,
    :math:`\hat{n} = \mathrm{rotate}(q, \hat{x})` is the body-frame
    x-axis rotated into the lab frame, :math:`m` is the ``multiplicity``,
    :math:`\varphi_0` is the ``phase``, and the smooth compact envelope
    :math:`g(r) = (1 - r^2/r_c^2)^2` ensures both force and energy vanish
    continuously at the cutoff :math:`r_c`.

    * ``multiplicity = 1, phase = 0`` (**polar**, default): energy depends
      on :math:`\cos\alpha`, so only parallel orientations are favourable
      and anti-parallel ones are repulsive.

    * ``multiplicity = 2, phase = 0`` (**nematic**): energy depends on
      :math:`\cos 2\alpha`, so parallel and anti-parallel orientations are
      equally favourable.

    Example::

        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        director = align_angle.DirectorPair(nlist=nlist, default_r_cut=3.0)

        # Polar (default, multiplicity=1)
        director.params[("A", "A")] = dict(epsilon=5.0)

        # Nematic
        director.params[("A", "A")] = dict(epsilon=5.0, multiplicity=2)

        sim.operations.integrator.forces.append(director)

    Attributes:
        params (TypeParameter[``particle types``, dict]):
            The parameters of the director pair potential for each particle
            type pair.  The dictionary has the following keys:

            * ``epsilon`` (`float`, **required**) - interaction strength
              :math:`[\mathrm{energy}]`
            * ``multiplicity`` (`int`, **optional**, default 1) - angular
              multiplicity :math:`m`
            * ``phase`` (`float`, **optional**, default 0) - phase offset
              :math:`\varphi_0` in radians
    """

    _cpp_class_name = "AnisoPotentialPairNematic"
    _ext_module = _align_angle

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(epsilon=float, multiplicity=1, phase=0.0, len_keys=2),
        )
        self._add_typeparam(params)


class SinSqDihedral(Dihedral):
    r"""Sin²-multiplied dihedral force (singularity-free).

    `SinSqDihedral` computes a modified periodic dihedral potential where the
    torsional barrier is multiplied by :math:`\sin^2\theta_1\,\sin^2\theta_2`,
    the squared sines of the two bond angles at the central atoms.  This
    smoothly sends the potential and all forces to zero when any three
    consecutive atoms become collinear, eliminating the :math:`1/\sin^2\theta`
    singularity present in the standard dihedral formulation.

    .. math::

        U = \frac{k}{2}
            \bigl(1 + d\,\cos(n\phi - \phi_0)\bigr)\,
            \sin^2\!\theta_{abc}\;\sin^2\!\theta_{bcd}

    where :math:`\phi` is the dihedral angle of the quartet
    :math:`(a, b, c, d)`, :math:`\theta_{abc}` is the bond angle at atom
    :math:`b`, and :math:`\theta_{bcd}` is the bond angle at atom :math:`c`.

    When :math:`\theta_{abc} = \theta_{bcd} = 90°`, this reduces to the
    standard periodic dihedral :math:`(k/2)(1 + d\cos(n\phi - \phi_0))`.

    Example::

        sinsq = align_angle.SinSqDihedral()
        sinsq.params["A-A-A-A"] = dict(k=10.0, d=1, n=1, phi0=0)
        sim.operations.integrator.forces.append(sinsq)

    Attributes:
        params (TypeParameter[``dihedral type``, dict]):
            The parameter of the sin²-dihedral potential for each dihedral
            type.  The dictionary has the following keys:

            * ``k`` (`float`, **required**) - spring constant
              :math:`[\mathrm{energy}]`
            * ``d`` (`float`, **required**) - sign factor (``1`` or ``-1``)
            * ``n`` (`int`, **required**) - multiplicity
            * ``phi0`` (`float`, **optional**, default 0) - phase offset
              :math:`\phi_0` in radians
    """

    _cpp_class_name = "SinSqDihedralForceCompute"
    _ext_module = _align_angle

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            "params",
            "dihedral_types",
            TypeParameterDict(k=float, d=float, n=int, phi0=0.0, len_keys=1),
        )
        self._add_typeparam(params)


class ExternalPatch(Force):
    r"""Patch interaction with externally defined patch directions.

    Each designated particle *i* carries a virtual "patch" whose direction
    is the unit vector from *i* toward its partner *j*.  When two patched
    particles approach within ``r_cut``, they interact via:

    .. math::

        U_{ik} = f_i\,f_k\;\epsilon\!\left(1 - r^2/r_c^2\right)^2

    where :math:`f_i` is a cubic Hermite (smoothstep) angular envelope:

    .. math::

        t = \mathrm{clamp}\!\left(\frac{u - (1 - w)}{w},\, 0,\, 1\right),
        \qquad f = 3t^2 - 2t^3

    with :math:`u = \hat{p}_i \cdot \hat{r}_{ik}` and ``w`` = ``width``.
    The patch is fully active (:math:`f = 1`) when the cosine alignment
    :math:`u \ge 1` (perfect alignment) and fully inactive (:math:`f = 0`)
    when :math:`u \le 1 - w`.

    No orientational (quaternion) degrees of freedom are required — the
    "torque" on the patch direction manifests as non-central translational
    forces on particles *i* and *j* (and similarly on *k* and *l*).

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        r_cut (float): Cutoff radius for patch–patch interactions.

    Attributes:
        epsilon (float): Attraction strength.
        width (float): Hermite transition width in cosine space (default 0.5).
        r_cut (float): Cutoff radius.
        partners (list[tuple[int,int]]): List of ``(attractor_tag,
            director_tag)`` pairs defining patch directions.

    Example::

        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        patch = align_angle.ExternalPatch(nlist=nlist, r_cut=3.0)
        patch.epsilon = 5.0
        patch.width = 0.5
        patch.partners = [(0, 1), (2, 3)]
        sim.operations.integrator.forces.append(patch)
    """

    _cpp_class_name = "ExternalPatchForceCompute"
    _ext_module = _align_angle

    def __init__(self, nlist, r_cut):
        super().__init__()

        # Store nlist
        param_dict = ParameterDict(nlist=hoomd.md.nlist.NeighborList)
        param_dict["nlist"] = nlist
        self._param_dict.update(param_dict)

        # Store the parameters that will be forwarded to C++
        self._r_cut = float(r_cut)
        self._epsilon = 0.0
        self._width = 0.5

        # Partner list — stored Python-side, pushed to C++ at attach
        self._partners = []

    # ─── Properties ───────────────────────────────────────────────────────

    @property
    def epsilon(self):
        """float: Attraction strength."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = float(value)
        if self._attached:
            self._cpp_obj.setParams(self._make_params_dict())

    @property
    def width(self):
        """float: Hermite transition width in cosine space."""
        return self._width

    @width.setter
    def width(self, value):
        self._width = float(value)
        if self._attached:
            self._cpp_obj.setParams(self._make_params_dict())

    @property
    def r_cut(self):
        """float: Cutoff radius."""
        return self._r_cut

    @r_cut.setter
    def r_cut(self, value):
        self._r_cut = float(value)
        if self._attached:
            self._cpp_obj.setRCut(self._r_cut)

    @property
    def partners(self):
        """list[tuple[int,int]]: Attractor–director partner pairs."""
        return self._partners

    @partners.setter
    def partners(self, pairs):
        self._partners = list(pairs)
        if self._attached:
            self._cpp_obj.setPartners(self._partners)

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _make_params_dict(self):
        return dict(
            epsilon=self._epsilon,
            width=self._width,
            r_cut=self._r_cut,
        )

    def _attach_hook(self):
        # Attach the neighbor list
        if self.nlist._attached and self._simulation != self.nlist._simulation:
            self.nlist = copy.deepcopy(self.nlist)
        self.nlist._attach(self._simulation)
        self.nlist._cpp_obj.setStorageMode(
            _md.NeighborList.storageMode.full
        )

        # Construct the C++ object
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(self._ext_module, self._cpp_class_name)
        else:
            cpp_cls = getattr(
                self._ext_module, self._cpp_class_name + "GPU"
            )

        self._cpp_obj = cpp_cls(
            self._simulation.state._cpp_sys_def,
            self.nlist._cpp_obj,
        )

        # Push parameters and partners to C++
        self._cpp_obj.setParams(self._make_params_dict())
        if self._partners:
            self._cpp_obj.setPartners(self._partners)

    def _detach_hook(self):
        self.nlist._detach()

