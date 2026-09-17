# ATLAS

Bayesian inference for pulsar timing arrays in JAX, with gradient-based sampling
over the full parameter space.

Standard PTA analyses proceed in stages: fix the timing solution, fit the white
noise per pulsar, linearise the timing model and marginalise it analytically,
then sample the red-noise and gravitational-wave-background parameters. The
staging is a computational convenience, and it means timing and white-noise
uncertainties never propagate into the background posterior. ATLAS samples
everything jointly — non-linear timing parameters, EFAC/EQUAD/ECORR, per-pulsar
intrinsic red noise, DM noise, deterministic sources and the correlated
background — in a single NumPyro model, using NUTS or HMC-within-Gibbs.

Most of the design follows from that requirement. The timing kernels are
differentiable, the Gaussian-process coefficients are sampled non-centred, the
mass matrix for the non-linear timing block is frozen from JUG's own covariance,
and the per-pulsar blocks are marginalised analytically wherever possible, all so
that a posterior of this dimension remains tractable.

## Installation

```
pip install -e .              # inference
pip install -e .[test]        # + pytest
pip install -e .[notebooks]   # + matplotlib, corner
```

Two dependencies are not on PyPI and are therefore not declared in
`pyproject.toml`:

- **JUG**, the timing back end, needed for the non-linear timing model and for
  generating Adaptus bases:
  `pip install git+https://github.com/MattTMiles/jug.git@dev`.
  PyPI also hosts an unrelated package called `jug`, so `import jug` succeeding
  proves nothing; check `import jug.delays.barycentric_jax`.
- **tempo2**, required by `libstempo`, which `enterprise` imports when it is
  available. `$TEMPO2` must point at a tempo2 runtime directory. Loading with
  `timing_package='tempo2'` handles TCB par files and tempo2's `BINARY T2`
  model, neither of which PINT reads.

Neither is needed to build a model or evaluate a likelihood. Only `ATLAS.pulsar`
and `ATLAS.signals.timing` import them, which is why the test suite runs on a
bare environment.

## Quick start

```python
import jax.numpy as jnp, jax.random as jrandom
from numpyro.infer import MCMC, NUTS

from ATLAS.data import PTA_Data
from ATLAS.model import model_maker
from ATLAS.model_builder import ModelBuilder
from ATLAS.psd_functions import hd_orf, powerlaw
from ATLAS.pulsar import load_pulsars

psrs = load_pulsars(parfiles, timfiles, timing_package="pint")

data = PTA_Data(psrs, num_gwb_bins=14, num_irn_bins=30,
                linear_timing=True, marg_timing=False)

m  = ModelBuilder(data=data)
wn = m.make_white_noise(stabilize_TNT=True)      # before make_red_noise
rn = m.make_red_noise("ltm|unc+cor->unc",
                      irn_psd_function=powerlaw,
                      gwb_psd_function=powerlaw,
                      orf_function=hd_orf,
                      irn_lower_bound_psd=jnp.array([-18., 0.]),
                      irn_upper_bound_psd=jnp.array([-11., 7.]),
                      gwb_lower_bound_psd=jnp.array([-18., 0.]),
                      gwb_upper_bound_psd=jnp.array([-11., 7.]))

raw = jnp.concat(data.raw_residuals)
lo, hi = wn.get_prior_bounds()

mcmc = MCMC(NUTS(model_maker, max_tree_depth=8), num_warmup=700, num_samples=700)
mcmc.run(jrandom.key(170817), raw_residuals=raw, super_sig=rn,
         vary_white=True, wn_lower_bound=lo, wn_upper_bound=hi,
         tm_model=None, helpers=None,
         marg_over_non_gwb=False, save_red_coeff=True)

coeff = mcmc.get_samples()["coeff"]      # [ndraw, npsr, ncol]
```

White noise must be constructed before red noise. `WhiteCov.__init__` registers
itself on the data object and `SuperSignal` reads it during construction; the
reverse order raises an `AttributeError` from inside a `functools.partial`.
Nothing enforces the ordering.

### Choosing what stays explicit

`marg_over_non_gwb` selects the likelihood, and with it the shape of the latent
vector `z_a` that NumPyro samples. The three configurations below differ only in
the arguments to `mcmc.run` and `PTA_Data`; everything above is shared.

**Every coefficient sampled non-centred** — the call above. `marg_over_non_gwb=False`
draws `z_a` of shape `[npsr, nmodes]`, one standard normal per basis column
including the timing block, and `lnposterior_reparam` maps it through the
standardising transform `c = ĉ + L⁻ᵀz`, where `L` is the Cholesky factor of the
posterior precision `Σ⁻¹ = TᵀN⁻¹T + diag(φ⁻¹)`. The sampler therefore explores
whitened coordinates while the recorded `coeff` are physical. With
`"ltm|unc+cor->unc"` and `num_irn_bins=30` above, that is the array-wide timing
width plus 60 red columns per pulsar.

`save_red_coeff=True` is what makes the coefficients retrievable — it registers
`numpyro.deterministic('coeff', coeff)`. Without it they are still sampled, and
still affect the posterior, but are discarded. The timing block of `coeff` is
what a back-transform through the design-matrix SVD turns into posteriors on
physical timing parameters.

**Only the background modes explicit** — the reduction that makes large arrays
tractable:

```python
mcmc.run(..., marg_over_non_gwb=True, save_red_coeff=True)
# z_a is now [npsr, 2 * num_gwb_bins]; coeff is [ndraw, npsr, 2*num_gwb_bins, 1]
```

`partial_marg_lnposterior` integrates the per-pulsar block analytically and keeps
only the `2 n_gwb` background coefficients. On 67 pulsars that is roughly 37,600
latent dimensions down to about 1,900. Note the trailing singleton axis on
`coeff`, which the non-marginalised path does not have.

**Timing model marginalised into `N`** — the staged-pipeline equivalent, for
comparison against conventional codes:

```python
data = PTA_Data(psrs, num_gwb_bins=14, num_irn_bins=30, marg_timing=True)
m    = ModelBuilder(data=data)                 # new data needs a new builder
wn   = m.make_white_noise(stabilize_TNT=True)  # and a fresh WhiteCov on it
rn   = m.make_red_noise("unc+cor->unc", ...)   # no `ltm|` prefix
mcmc.run(..., marg_over_non_gwb=False)         # z_a is [npsr, 2*num_irn_bins]
```

`marg_timing=True` sets `linear_timing_model_size = 0`, so `T` carries no timing
columns and the timing solution is projected out inside the noise matrix. Best
conditioned of the four treatments, and the only one with no timing coefficients
to recover.

Rebuilding the builder is not optional. `ModelBuilder` binds the dataset at
construction and `make_red_noise` reads `self.data`, so rebinding the name
`data` leaves an existing builder pointing at the previous `PTA_Data` — it would
build the *un*-marginalised model without complaint. The same applies to
`WhiteCov`, which registers itself on the data object it was given.

## Likelihood

For pulsar *a* the residuals are modelled as

```
    δt_a = M_a ε_a + T_a c_a + n_a ,        n_a ~ N(0, N_a) ,  c ~ N(0, φ)
```

where `M` is the timing design matrix, `T` collects the Fourier and low-rank
bases, and `N` carries EFAC, EQUAD and ECORR. Marginalising the Gaussian-process
coefficients gives the usual Woodbury form (van Haasteren & Levin 2013; van
Haasteren & Vallisneri 2014), with `C = N + TφTᵀ` and
`Σ = TᵀN⁻¹T + φ⁻¹`. For the background, `φ_ab(f) = Γ_ab P(f)` with `Γ` the
Hellings–Downs curve.

All of the data enters the likelihood through four arrays, computed once per
evaluation and referred to throughout the code as `helpers`:

```
    TNT        [npsr, ncol, ncol]     Tᵀ N⁻¹ T
    TNr        [npsr, ncol]           Tᵀ N⁻¹ δt
    rNr        scalar                 δtᵀ N⁻¹ δt,  summed over the array
    logdet_N   scalar                 ln det N,    summed over the array
```

Nothing downstream of `helpers` touches a TOA.

```
                        PTA_Data  (data.py)
                       /                   \
       WhiteCov (nMatrix/base.py)       SuperSignal (signals/factorized/base.py)
       N = EFAC²(σ² + EQUAD²) + ECORR   T = [ M | F_unc | F_dm | U_gtm | F_det ]
       Sherman–Morrison per epoch       column-slice map per signal
                       \                   /
                     helpers = (TNT, TNr, rNr, logdet_N)
                                  |
              lnposterior_reparam   /   partial_marg_lnposterior
                                  |
                          model_maker  (model.py)
                                  |
                  NUTS / MultiHMCGibbs  (samplers/canetoadracing.py)
```

With `vary_white=False` the helpers are built once outside the sampler and
passed in as a constant. With `vary_white=True` they are rebuilt at every
leapfrog step, over TOA-length arrays, and that rebuild dominates the cost of a
joint fit. `bench/bench_gradient.py` measures the ratio for a given
configuration.

Three entry points consume the helpers:

- `ln_likelihood_curn` marginalises all coefficients with a diagonal `φ`, i.e.
  the common-uncorrelated-red-noise model. Useful as an exact reference. It has
  no deterministic term and refuses a `det` block.
- `lnposterior_reparam` samples all coefficients non-centred, with the full
  ORF-correlated `φ`.
- `partial_marg_lnposterior` marginalises the per-pulsar block analytically
  (timing, intrinsic red noise, DM, Adaptus) and keeps only the background modes
  explicit. On a 67-pulsar configuration this reduces roughly 37,600 latent
  coefficients to about 1,900.

The latter two take an optional `D_params` for a deterministic signal; see
*Deterministic signals*.

The standardising transform in the latter two is built from the diagonal of
`φ⁻¹`, so it whitens exactly only when the ORF vanishes; with a non-trivial ORF
it acts as a CURN preconditioner. The densities are correct either way.

## Signal specification

Signals are declared with a compact string passed to `make_red_noise`:

```
    "ltm|unc+cor->unc;gtm"
     └┬┘ └───┬───┘└┬┘ └┬┘
      │      │     │   └── separate blocks, each with its own columns
      │      │     └────── representative: whose basis the shared group uses
      │      └──────────── shared group: these signals share columns
      └─────────────────── prepend the linear timing design matrix M
```

Recognised names are `unc` (intrinsic red noise), `cor` (correlated background),
`dm` (dispersion measure), `gtm` (Adaptus timing basis) and `det`
(deterministic).

No name is mandatory, `cor` included. A single-pulsar noise run wants a model
with no correlated process at all — a common process cannot be separated from
intrinsic red noise in one pulsar, so including one costs two unconstrained
parameters and a flat ridge in the posterior:

```python
data = PTA_Data([psr], num_irn_bins=30, marg_timing=True)
m    = ModelBuilder(data=data)                 # see above: new data, new builder
wn   = m.make_white_noise(stabilize_TNT=True)
rn   = m.make_red_noise("unc", irn_psd_function=powerlaw,
                        irn_lower_bound_psd=jnp.array([-18., 0.]),
                        irn_upper_bound_psd=jnp.array([-11., 7.]))
mcmc.run(..., marg_over_non_gwb=False)
```

`SuperSignal` selects `PerPulsarRedNoise` instead of `CorrelatedPulsarRedNoise`
when there is no `cor` block, and no ORF is needed. `partial_marg_lnposterior`
is unavailable here by construction — it keeps the background block and
marginalises the rest, so with no background there is nothing to keep; it raises
a `ValueError` saying so. Use `marg_over_non_gwb=False`.

Signals in a shared group *overlap* rather than sit adjacent. The background is
modelled only in the lowest frequency bins, so `cor` occupies the first
`2 n_gwb` columns of the `unc` block: two independent processes on the same
basis columns, with distinct coefficient vectors. In `φ` the overlap is an
addition, and only the background bins acquire off-diagonal ORF terms — the
remainder are inverted by reciprocal rather than Cholesky.

For `"ltm|unc+cor->unc;dm,gtm"` with 4 background bins, 6 red-noise bins, 3 DM
bins, 8 Adaptus modes and a 6-column timing block:

```
    col   0 ..  5    timing    M, zero-padded to the array-wide maximum
    col   6 .. 17    unc       F_irn        (cor occupies 6..13, nested)
    col  18 .. 23    dm        F_dm
    col  24 .. 31    gtm       U_adaptus
```

Timing design matrices are zero-padded to a common width so the array is one
batched `[npsr, ntoa, ncol]` tensor. Padded columns carry unit prior precision
rather than the near-flat `1e-40` given to real timing columns, so that HMC sees
curvature instead of an unbounded flat direction.

This layout is hardcoded around the five signal names above. Adding a sixth
requires edits in both `parameterized.py` classes as well as
`signals/factorized/base.py`; replacing it with a component registry is planned.

## Deterministic signals

A deterministic signal — a continuous wave from an individual supermassive
black-hole binary, or anything else with a closed-form waveform — enters through
the same basis as everything else. The waveform is evaluated on an evenly spaced
grid, Tukey-windowed over a span extended either side of `Tspan` (to keep the
FFT of a non-periodic signal from ringing), transformed, and the resulting
Fourier coefficients are mapped onto the observed TOAs by a design matrix
`F_det` (Gundersen & Cornish 2025). Those columns are the `det` block.

What separates it from a stochastic block is not the basis but the prior: it has
none. Its coefficients are a deterministic function of the source parameters, so
they are *computed* from `D_params` rather than sampled, and the source
parameters are what the sampler explores.

```python
data = PTA_Data(psrs, num_gwb_bins=14, num_irn_bins=30, num_det_bins=60,
                marg_timing=True)              # num_det_bins is required by ;det
...
rn = m.make_red_noise("unc+cor->unc;det",
                      irn_psd_function=powerlaw, gwb_psd_function=powerlaw,
                      orf_function=hd_orf, ...,
                      det_delay_function=cw_delay_evolve_float64,
                      det_parameter_bounds=cw_bounds)      # [nparam, 2], min in col 0
```

`det_delay_function(toas, psr_pos, source_params, psr_phases, psr_dists)` returns
delays **in seconds**, shape `[npsr, ntoa]`, and must be JAX-traceable and
differentiable in `source_params`.
`signals/deterministic/det_signals.py:cw_delay_evolve_float64` is the evolving
circular-binary waveform including the pulsar term, with `source_params` ordered
`[log10 M_c, log10 f_gw, cos ι, ψ, log10 h, cos θ, φ, Φ₀]`. A waveform with no
pulsar term takes `Deterministic(..., with_psr_params=False)`, which is
constructed directly rather than through `ModelBuilder`, and is then called with
`psr_phases=None, psr_dists=None`.

Both reparameterised likelihoods take the source parameters as
`D_params = (det_params, psr_phases, psr_dists)`:

```python
lnpost, coeff = rn.lnposterior_reparam(helpers, red_params, z, D_params=D)
lnpost, coeff = rn.partial_marg_lnposterior(helpers, red_params, z_gwb, D_params=D)
```

`z` is one standard normal per *stochastic* column — `rn.nmodes_reparam`, not
`rn.nmodes`. `nmodes` is the full width of `T` and counts the deterministic
columns, which `z` does not touch. The two are equal for every model without a
`det` block, which is why `jnp.zeros((npsr, rn.nmodes))` appears throughout the
notebooks and tools; with one it is too wide, and the likelihood says so rather
than broadcasting.

For `"ltm|unc+cor->unc;det"` with 4 background bins, 6 red-noise bins, 20
deterministic bins and a 6-column timing block:

```
    col   0 ..  5    timing    M, zero-padded          ┐
    col   6 .. 17    unc       F_irn (cor at 6..13)    ┘ z, width nmodes_reparam = 18
    col  18 .. 57    det       F_det                     a_det(D_params), width 40
```

`model_maker` drives the stochastic sites only and raises for a `det` block: the
source parameters need priors it cannot infer — a pulsar-distance prior in
particular is a per-array input, and `PTA_Data` carries `psr_pos` but no
distances. Write the model instead, which is the whole of it:

```python
def model():
    z   = numpyro.sample('z_a', dist.Normal(0, 1).expand((rn.npsrs, rn.nmodes_reparam)))
    red = numpyro.sample('red_noise', dist.Uniform(rn.model.lower_prior_lim_all,
                                                   rn.model.upper_prior_lim_all))
    cw  = numpyro.sample('cw', dist.Uniform(rn.det_signal.det_param_mins,
                                            rn.det_signal.det_param_maxs))
    phases = numpyro.sample('psr_phases',
                            dist.Uniform(0., 2 * jnp.pi).expand((rn.npsrs,)))
    dist_z = numpyro.sample('psr_dist_z', dist.Normal().expand((rn.npsrs,)))
    dists  = numpyro.deterministic('psr_dists', dist_z * pdist_err + pdist_mean)

    lnpost, coeff = rn.lnposterior_reparam(helpers, red, z,
                                           D_params=(cw, phases, dists))
    numpyro.factor('lnpost', lnpost + 0.5 * jnp.sum(z ** 2))
```

The `+ ½ Σz²` is the same cancellation `model_maker` makes: the density returned
is in the reparameterised coordinate and NumPyro has already added the `N(0,1)`
prior for `z`.

The identity that pins all of this is that a deterministic block must be exactly
equivalent to subtracting the same waveform from the data by hand:
`lnpost(δt, D_params) == lnpost_no_det(δt − F_det a_det)`. `tests/test_identities.py`
asserts it for both timing treatments, along with the gradient in the source
parameters against central differences.

## Timing model

Four treatments coexist, selected by different flags:

| treatment | selected by | notes |
|---|---|---|
| marginalised into `N` | `marg_timing=True` | projection inside the noise matrix; no timing columns in `T`; best conditioned, `cond(Σ) ~ 5×10⁵` |
| linear, sampled | `linear_timing=True` | `M` prepended to `T`, coefficients sampled under a near-flat prior; currently the production path; `cond(Σ)` reaches `10¹⁸` |
| non-linear | pass `tm_model` to `model_maker` | physical timing parameters through differentiable JUG kernels; needs a frozen dense mass matrix |
| Adaptus | `;gtm` in the model string | PCA of prior-predictive timing residuals used as a GP basis; combines with either linear option |

The first two are mutually exclusive and nothing currently enforces that.

## Normalisation

All three likelihoods drop the same additive constants, so they are mutually
consistent but are not normalised log-densities:

1. `−(N_toa/2) ln 2π` from the likelihood. Values are larger than normalised
   ones by that amount — of order `5.5×10⁵` at NANOGrav 15-year scale.
2. On the `linear_timing=True` path only, `−½ n_tm ln(10⁴⁰)` per pulsar over
   real (unpadded) timing columns. The flat timing prior enters `Σ` as
   `φ⁻¹ = 10⁻⁴⁰`, but its log-determinant is never added.

Both are parameter-independent, so posteriors, acceptance ratios and Bayes
factors at fixed `N_toa` are unaffected. Absolute evidences are not, nor are
comparisons across different `N_toa` (for instance including or excluding a
pulsar), nor comparisons against codes that normalise fully. The module
docstring of `tests/reference.py` states the conventions precisely, and the test
suite asserts the offset rather than assuming it.

## Tests

```
pip install -e .[test]
pytest                            # ~30 s, CPU, no data files required
pytest -m golden                  # + a 36-pulsar MDC1 fit (GPU, ~10 min)
python tools/measure_margins.py   # achieved margins -> tools/noise_floors.json
python tools/regress.py --null    # harness self-test; must be exactly 0.0
python tools/regress.py --ref-a HEAD~1 --ref-b HEAD
python bench/bench_gradient.py --case ltm-gtm --fixture ng15_3
```

The suite is anchored on `tests/reference.py`, a dense float64 NumPy evaluation
of `ln N(δt; 0, N + TφTᵀ)` that shares no code with the package. Agreement is
currently at the `10⁻¹⁶`–`10⁻¹⁵` level for the likelihoods and `3×10⁻⁸` for
gradients against central differences.

Fixtures are duck-typed rather than `enterprise` objects: `tests/fixtures/`
defines the eight attributes ATLAS reads from a pulsar and stores real MDC1 and
NANOGrav data as small `.npz` files with provenance. Regenerating them
(`tools/make_fixtures.py`) requires PINT or tempo2; using them does not.

`tests/harness.py:CORPUS` is the reference list of working model strings, shared
with the regression harness:

| case | model string | columns |
|---|---|---|
| `curn` | `unc+cor->unc` | 12 |
| `curn-margtm` | `unc+cor->unc`, `marg_timing=True` | 12 |
| `ltm` | `ltm\|unc+cor->unc` | 18 |
| `ltm-dm` | `ltm\|unc+cor->unc;dm` | 24 |
| `ltm-gtm` | `ltm\|unc+cor->unc;gtm` | 26 |
| `ltm-dm-gtm` | `ltm\|unc+cor->unc;dm,gtm` | 32 |

`tools/regress.py` compares two git revisions' log-densities and gradients
elementwise, and is intended as a gate before refactoring.

## Package layout

| path | lines | contents |
|---|---|---|
| `data.py` | 143 | `PTA_Data`: arrays plus run configuration |
| `model_builder.py` | 122 | `ModelBuilder`: the construction API |
| `model.py` | 104 | `model_maker`: the NumPyro model |
| `nMatrix/base.py` | 1192 | white-noise covariance and its solves |
| `signals/factorized/base.py` | 1518 | `Red`, `GaussianTiming`, `SuperSignal`, the likelihoods |
| `parameterized.py` | 1623 | `φ` assembly and inversion |
| `signals/signals_utils.py` | 565 | model-string parser, column bookkeeping, timing SVD |
| `signals/correlated/base.py` | 608 | `Correlated`: background signal and ORF |
| `signals/timing/` | 2266 | non-linear timing model (JUG) |
| `signals/deterministic/` | 713 | continuous-wave and other deterministic signals |
| `samplers/canetoadracing.py` | 678 | vendored `MultiHMCGibbs` kernels |
| `psd_functions.py` | 501 | PSD and ORF library |
| `sim.py` | 357 | simulation |
| `experimental/` | 4832 | not reachable from any entry point; see its docstring |

The rows above account for 15,222 lines; the package is 16,853 across 34 modules.

## Known issues

- `signals/deterministic/base.py:JointDeterministic` is a stranded earlier
  implementation of the deterministic path and cannot be constructed: it calls
  `SuperSignal.__init__` with the pre-refactor `signal_helper=` schema and
  raises `TypeError`. It also subtracts `tref` from its TOA grid before handing
  it to a delay function that subtracts `tref` again. The live path is the `det`
  block documented under *Deterministic signals*, which shares none of this
  code; `notebooks/CW_demo.ipynb` predates it and calls the stranded class.
- `model_maker` does not sample deterministic parameters — the priors involved
  are per-array inputs it has no access to — so a `det` block has to be driven
  from a hand-written NumPyro model. It raises rather than half-sampling.
  `ln_likelihood_curn` has no deterministic term at all and refuses one too.
- An `ltm|` prefix is only honoured when the shared group names a representative
  with `->`. Without one, `build_basis` leaves `M` out of the basis while the
  column map still claims it is there, so `"ltm|unc"` and `"ltm|unc;cor"` both
  produce slices past the end of the basis. Sampled linear timing therefore
  needs a `->` in the string; `marg_timing=True` is unaffected.
- `ln_likelihood_curn` is unusable whenever `SuperSignal` selects
  `PerPulsarRedNoise` — that is, on any model with no `cor` block — because the
  two `parameterized` classes disagree over whether `get_phi_mat_CURN` returns
  an array or an `(array, psd_common)` tuple. The same mismatch blocks a
  directly supplied `gtm_psd`. The other two likelihoods are unaffected.
- `lnposterior_reparam` and `partial_marg_lnposterior` return `coeff` with
  different ranks — `[npsr, ncol]` and `[npsr, ncol, 1]` respectively.
- `linear_timing=True` combined with a model string lacking an `ltm|` prefix is
  an inconsistent configuration that fails as a raw broadcast error rather than
  a message.
- The chromatic index is implemented but connected to nothing:
  `SuperSignal.update_red_basis` has no callers.
- `stabilize_TNT` is a no-op for any pulsar with padded timing columns, since
  those columns are exactly zero and the shift is proportional to
  `min(diag(TNT))`.

## References

- Hellings & Downs 1983, ApJ 265, L39
- Lentati et al. 2013, PRD 87, 104021
- van Haasteren & Levin 2013, MNRAS 428, 1147
- van Haasteren & Vallisneri 2014, PRD 90, 104012
- Taylor 2021, *The Nanohertz Gravitational Wave Astronomer*, arXiv:2105.13270
- Agazie et al. 2023, ApJL 951, L8
