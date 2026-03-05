# hoomd-align-angle

A [HOOMD-blue](https://hoomd-blue.readthedocs.io/) plugin that applies torques
to align an oriented particle along the direction defined by two guide particles,
using HOOMD's angle topology.

## Physics

For an angle group `(i, j, k)`:
- Particles `i` and `k` are **guide** particles that define a target direction
  `d̂ = (r_k − r_i) / |r_k − r_i|`.
- Particle `j` is the **oriented** particle whose body-frame x-axis `n̂ = rotate(q_j, x̂)`
  should align with `d̂`.

The potential energy is:

$$U = \frac{K}{2}(1 - \cos\theta), \quad \cos\theta = \hat{n} \cdot \hat{d}$$

Forces and torques:
- **Torque on j** (lab frame): $\tau_j = \frac{K}{2}\,\hat{n} \times \hat{d}$
- **Force on i**: $F_i = -\frac{K}{2|d|}\left(\hat{n} - \cos\theta\,\hat{d}\right)$
- **Force on k**: $F_k = -F_i$ (Newton's 3rd law)
- **No force on j** (potential couples only to j's orientation)

## Building

Requires HOOMD-blue ≥ 5.0.0 built with GPU support (HIP/CUDA).

```bash
cmake -B build -S . -DCMAKE_INSTALL_PREFIX=$(python -c "import hoomd; print(hoomd.__path__[0] + '/..')")
cmake --build build -j$(nproc)
cmake --install build
```

## Usage

```python
import hoomd
from hoomd import align_angle

align_force = align_angle.Align()
align_force.params["align"] = dict(k=20.0)

integrator = hoomd.md.Integrator(dt=0.005, methods=[...], forces=[align_force])
integrator.integrate_rotational_dof = True  # required!
```

**Important:** You must set `integrator.integrate_rotational_dof = True` for HOOMD
to integrate orientational degrees of freedom.

## Tests

```bash
python -m pytest src/pytest/test_align_angle.py
```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
