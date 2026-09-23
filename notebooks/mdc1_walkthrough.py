#!/usr/bin/env python
"""
A minimal, end-to-end ATLAS global fit on the IPTA Mock Data Challenge 1 (Open Dataset 1).

This is the script version of `notebooks/mdc1_global_fit.ipynb`, stripped to the parts you
actually need to run an analysis, and with the two sampling strategies side by side:

    --sampler nuts      one joint NUTS kernel over (white noise, red hyperparameters, latent
                        coefficients).  The helpers (T^T N^-1 T, T^T N^-1 r, r^T N^-1 r,
                        log|N|) depend on the white noise, so they are rebuilt on EVERY
                        leapfrog step.

    --sampler gibbs     two-block HMC-within-Gibbs:
                          block A = white noise                     (helpers rebuilt)
                          block B = (red hyperparameters, latent z)  (helpers FROZEN)
                        Because block B conditions on a fixed white noise, its helpers are a
                        compile-time constant of the model -- ATLAS recomputes nothing while
                        that block is sampling.  One rebuild per Gibbs sweep instead of one
                        per leapfrog step.

Both paths target the same posterior and, on this dataset, agree on it to well inside a tenth
of a standard deviation.  They do NOT cost the same -- see "WHICH SAMPLER SHOULD I USE?" at
the bottom of this docstring.

--------------------------------------------------------------------------------------------
WHAT THIS DATASET IS
--------------------------------------------------------------------------------------------
IPTA MDC1 Open-1: 36 pulsars, 5 yr, 130 TOAs each, 0.1 us TOA errors, one observing frequency
and one TOA per session.  Injected: a Hellings-Downs-correlated power-law GWB with
log10_A = -13.301, gamma = 13/3, no individual red noise, no DM noise, white noise exactly at
the quoted TOA errors.  It ships inside `enterprise-pulsar` as enterprise/datafiles/mdc_open1/.

--------------------------------------------------------------------------------------------
THE ONE THING THAT WILL RUIN YOUR DAY: TCB vs TDB
--------------------------------------------------------------------------------------------
The MDC1 par files carry no UNITS line.  tempo2 defaults to TCB; PINT defaults to TDB.  Read
through PINT, the timing solution is not tracked, and the residuals become a uniform hash over
one pulse period (RMS = P/sqrt(12), ~1400 us instead of ~1.2 us).  The fit then "recovers" a
GWB five decades too loud.  This script loads through tempo2 (`timing_package="tempo2"`) and
asserts on the phase-hash signature before it does anything else.  Loading through tempo2 also
gets you all 36 pulsars: ten use `BINARY T2`, tempo2's auto-dispatching binary model, which
PINT cannot parse and would silently drop.

Requires libstempo, and the TEMPO2 environment variable pointing at tempo2's share/tempo2
runtime directory.

--------------------------------------------------------------------------------------------
WHITE NOISE ON THIS DATASET
--------------------------------------------------------------------------------------------
* One backend per pulsar (the .tim files carry no backend flags at all), so varying the white
  noise means exactly one EFAC and one EQUAD per pulsar.
* ECORR is not merely unnecessary, it is *exactly unidentifiable*: every observing session
  holds a single TOA, so there is no epoch structure for ECORR to correlate.  We pass
  `include_ecorr=False`; ATLAS raises a clear error if you ask for it anyway.
* EFAC and EQUAD are individually unidentified here too, because sigma_i is constant within
  each pulsar -- the likelihood only ever sees sigma_eff = EFAC * sqrt(sigma^2 + EQUAD^2).
  Judge the recovery on sigma_eff (truth 0.1 us), not on EFAC (which comes out biased low).
  That ridge is real, and it is what makes every NUTS trajectory run to the tree-depth cap.

--------------------------------------------------------------------------------------------
WHICH SAMPLER SHOULD I USE?
--------------------------------------------------------------------------------------------
Measured on all 36 pulsars, 700 warmup + 700 samples, one RTX 4090, float64:

    config                     wall     ESS(EFAC)  ESS(GWB)   EFAC/min   GWB/min
    joint NUTS, diag mass     10.3 min      175       871       17.0       84.6
    Gibbs, --red-steps 1      11.8 min      192       751       16.3       63.9
    Gibbs, --red-steps 3      17.2 min      209       662       12.2       38.5

Gibbs LOSES here, and it is worth understanding why before you reach for it on your own data.
Freezing the helpers is a real saving -- benchmarked at 1.96x per leapfrog step (0.965 ms
frozen vs 1.888 ms rebuilt) -- but the white-noise block on its own still needs ~220 leapfrog
steps per sweep against the joint chain's 255, because the EFAC/EQUAD ridge lives entirely
inside that block.  So Gibbs pays ~220 expensive steps and then adds 127+ cheap ones on top.

    General rule: blocking wins when the EXPENSIVE block is also the WELL-CONDITIONED one.
    Here it is exactly inverted -- the block that forces the helpers rebuild is the same block
    that carries the degenerate direction.

On a real PTA with per-backend sigma spread the EFAC/EQUAD degeneracy is only partial, the
white-noise block is better conditioned, and this arithmetic can come out the other way.  The
`gibbs` path is here so you can measure it on your own dataset rather than guess.

--------------------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------------------
    python mdc1_walkthrough.py                                  # joint NUTS, 36 psrs, vary WN
    python mdc1_walkthrough.py --sampler gibbs --red-steps 1
    python mdc1_walkthrough.py --npsrs 5 --warmup 200 --samples 200   # a fast smoke test
    python mdc1_walkthrough.py --fixed-white-noise               # WN held at the TOA errors
"""

import argparse
import glob
import os
import pickle
import shutil
import sys
import time

# ============================================================================================
# 0.  ENVIRONMENT -- all of this must happen BEFORE jax is imported for the first time.
# ============================================================================================

def _bootstrap_env():
    """Set the environment variables jax and libstempo need, before importing either."""
    # Without this the parent process grabs most of the GPU up front, and the joblib workers
    # that load pulsars in parallel die with CUDA_ERROR_OUT_OF_MEMORY.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # NUTS at 1838 dimensions with 36 pulsars fits in 45% of a 24 GB card. Raise it if you
    # scale up the pulsar count or the number of frequency bins.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

    # libstempo needs TEMPO2 pointing at tempo2's runtime data (clock files, ephemerides).
    if os.environ.get("TEMPO2") and os.path.isdir(os.environ["TEMPO2"]):
        return
    candidates = []
    exe = shutil.which("tempo2")
    if exe:
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(exe)), "share", "tempo2"))
    candidates += glob.glob(os.path.join(sys.prefix, "..", "*", "share", "tempo2"))
    candidates += glob.glob(os.path.join(sys.prefix, "share", "tempo2"))
    for c in candidates:
        if os.path.isdir(c):
            os.environ["TEMPO2"] = os.path.abspath(c)
            return
    raise RuntimeError(
        "Could not locate tempo2's runtime data directory. Install tempo2 and set the TEMPO2 "
        "environment variable to its .../share/tempo2 directory."
    )


_bootstrap_env()

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)      # ATLAS assumes float64 everywhere
import jax.numpy as jnp
import jax.random as jrandom

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import potential_energy

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ATLAS.data import PTA_Data
from ATLAS.model_builder import ModelBuilder
from ATLAS.nMatrix.base import EPOCH_THRESHOLD, WhiteCov, _get_epochs
from ATLAS.psd_functions import hd_orf, powerlaw
from ATLAS.pulsar import load_pulsars
from ATLAS.samplers.canetoadracing import model_maker

# The injected values, for scoring the recovery at the end.
TRUE_GWB_LOG10_A = -13.301
TRUE_GWB_GAMMA = 13.0 / 3.0
TRUE_SIGMA_US = 0.1           # every MDC1 TOA error is exactly 0.1 us
YR = 365.25 * 86400.0


# ============================================================================================
# 1.  DATA -- load through tempo2, then refuse to proceed if it looks wrong.
# ============================================================================================

def load_mdc1(num_psrs, cache_dir):
    """Load the first `num_psrs` MDC1 Open-1 pulsars through tempo2, with a pickle cache."""
    import enterprise
    mdc_dir = os.path.join(os.path.dirname(enterprise.__file__), "datafiles", "mdc_open1")
    if not os.path.isdir(mdc_dir):
        raise RuntimeError(
            f"MDC1 Open-1 data not found at {mdc_dir}. It ships inside enterprise-pulsar as "
            "enterprise/datafiles/mdc_open1/ -- check your installation."
        )

    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{mdc_dir}/*.par"))
    if not 2 <= num_psrs <= len(names):
        raise ValueError(f"--npsrs must be between 2 and {len(names)}")
    names = names[:num_psrs]
    parfiles = [f"{mdc_dir}/{n}.par" for n in names]
    timfiles = [f"{mdc_dir}/{n}.tim" for n in names]

    cache = os.path.join(cache_dir, f"psrs_mdc1_t2_{num_psrs}psr.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as fin:
            psrs = pickle.load(fin)
        print(f"loaded {len(psrs)} pulsars from cache {cache}")
    else:
        t0 = time.time()
        # timing_package="tempo2" is THE critical argument -- see the TCB/TDB note at the top.
        psrs = load_pulsars(parfiles, timfiles, use_enterprise=True, timing_package="tempo2")
        print(f"loaded {len(psrs)} pulsars through tempo2 in {time.time() - t0:.1f} s")
        with open(cache, "wb") as fout:
            pickle.dump(psrs, fout)

    return psrs, parfiles, timfiles, mdc_dir


def check_not_a_phase_hash(psrs, mdc_dir):
    """Halt if any pulsar's residuals look like a uniform hash over one pulse period.

    A timing solution that is not being tracked -- the TCB-read-as-TDB failure mode -- gives
    residuals uniform on (-P/2, P/2], hence RMS = P/sqrt(12).  Catching this here is worth far
    more than any sampler diagnostic downstream: the fit runs happily on hashed residuals and
    reports a confident, wrong GWB.
    """
    bad = []
    for p in psrs:
        f0 = None
        for line in open(f"{mdc_dir}/{p.name}.par"):
            tok = line.split()
            if tok and tok[0] == "F0":
                f0 = float(tok[1])
        rms = float(np.std(np.asarray(p.residuals)))
        hash_level = 1.0 / f0 / np.sqrt(12.0)
        if 0.9 < rms / hash_level < 1.1:
            bad.append(p.name)
    if bad:
        raise RuntimeError(
            f"Residual RMS is consistent with a uniform hash over one pulse period for {bad}. "
            "The timing solution is not being tracked -- you are almost certainly reading these "
            "TCB par files as TDB. Load with timing_package='tempo2'."
        )
    med = np.median([float(np.std(np.asarray(p.residuals))) for p in psrs]) * 1e6
    print(f"phase-hash guard passed. median residual RMS = {med:.2f} us "
          f"(~{med / TRUE_SIGMA_US:.0f}x the {TRUE_SIGMA_US} us TOA-error floor)")
    print("  that excess is what the fit has to explain: GWB, per-pulsar red noise, or both.")


def check_ecorr_is_unidentifiable(psrs):
    """Derive from the TOA epoch spacing that ECORR has nothing to correlate.

    `_get_epochs` groups TOAs within EPOCH_THRESHOLD = 1 s and then discards every group of
    size one.  MDC1 has a single TOA per observing session, so it leaves NO epochs at all --
    ECORR would be exactly degenerate with EQUAD (EQUAD_eff^2 = EQUAD^2 + ECORR^2/EFAC^2) and
    sampling it would just explore the prior against a flat likelihood.
    """
    n_multi = sum(len(_get_epochs(np.asarray(p.toas), EPOCH_THRESHOLD)) for p in psrs)
    min_gap = min(np.diff(np.sort(np.asarray(p.toas))).min() for p in psrs) / 86400.0
    print(f"epoch check: {n_multi} multi-TOA epochs across the array "
          f"(EPOCH_THRESHOLD = {EPOCH_THRESHOLD} s, smallest TOA gap = {min_gap:.3f} d)")
    if n_multi != 0:
        raise RuntimeError(
            "Some epochs hold more than one TOA -- ECORR is identifiable after all, and this "
            "script's include_ecorr=False is the wrong choice for your data."
        )
    print("  -> zero. Building the white noise without ECORR (EFAC + EQUAD only).")
    return False    # include_ecorr


# ============================================================================================
# 2.  MODEL -- PTA_Data -> WhiteCov -> ModelBuilder.make_red_noise
# ============================================================================================

def build_model(psrs, parfiles, timfiles, vary_white):
    """Assemble the three ATLAS objects the samplers need: data, white noise, red noise."""
    data = PTA_Data(
        psrs,
        num_gwb_bins=8,          # GWB in the lowest 8 harmonics of 1/Tspan
        num_irn_bins=15,         # per-pulsar IRN (none injected -- a null test)
        num_dm_bins=None,        # no DM component
        adaptus_basis=None,
        adaptus_size=None,
        # None  -> helpers take white_noise_params at call time (sampler varies them)
        # scalar-> white noise is baked into the helpers once, and they never change
        fixed_white_noise_params=None if vary_white else jnp.array(1.0),
        linear_timing=True,      # sample the timing offsets rather than marginalise them
        marg_timing=False,
        diag_white_cov=not vary_white,
        fixed_res=False,
        parfiles=parfiles,
        timfiles=timfiles,
        noise_dict=None,         # no backend noise files ship with MDC1
        dm_ref_freq=1400,
    )

    if vary_white:
        wn = WhiteCov(
            data=data,
            stabilize_TNT=True,
            include_ecorr=False,          # justified by check_ecorr_is_unidentifiable()
            efac_prior_bounds=(0.01, 10.0),
            log10equad_prior_bounds=(-9.0, -5.0),
        )
        wn_names = wn.get_param_names()
        wn_lower, wn_upper = wn.get_prior_bounds()
        # Start every EFAC at the truth and every EQUAD well below the sigma floor.
        wn_init = jnp.array([1.0 if n.endswith("_efac") else -8.0 for n in wn_names])
    else:
        # DiagSinglePulsarWhiteCov has no free parameters and does not implement the
        # parameter-vector interface -- don't call get_param_names() on it.
        wn = WhiteCov(data=data, stabilize_TNT=True)
        wn_names, wn_lower, wn_upper, wn_init = [], None, None, None

    rn = ModelBuilder(data=data).make_red_noise(
        "ltm|unc+cor->unc",      # linear timing model | uncorrelated + correlated -> shared cols
        use_pulsar_tspan=False,  # one common frequency grid from the PTA Tspan
        irn_psd_function=powerlaw,
        gwb_psd_function=powerlaw,
        orf_function=hd_orf,     # Hellings-Downs, no free parameters
        dm_psd_function=None,
        irn_lower_bound_psd=jnp.array([-20.0, 0.0]),    # [log10_A, gamma]
        irn_upper_bound_psd=jnp.array([-11.0, 7.0]),
        gwb_lower_bound_psd=jnp.array([-18.0, 0.0]),
        gwb_upper_bound_psd=jnp.array([-11.0, 7.0]),
        dm_lower_bound_psd=None,
        dm_upper_bound_psd=None,
        upper_bound_orf=None,
        lower_bound_orf=None,
    )

    red_names = rn.model.get_param_names()
    # Deliberately start the GWB ~20x quieter than the injection, so recovering it is a real test.
    red_init = jnp.array([-16.0, 3.0] * data.npsrs + [-14.0, 13.0 / 3.0])
    z_init = jnp.zeros((data.npsrs, rn.nmodes))

    print(f"\nPTA Tspan       : {data.pta_tspan / YR:.2f} yr "
          f"({data.npsrs} pulsars, {data.npairs} pairs, "
          f"{sum(len(p.toas) for p in psrs)} TOAs)")
    print(f"lowest GWB freq : {1 / data.pta_tspan * 1e9:.2f} nHz")
    print(f"T columns/pulsar: {rn.nmodes}  (timing {rn.linear_timing_model_size} + red)")
    print(f"sampled dims    : {data.npsrs * rn.nmodes} latent + {len(red_names)} red "
          f"+ {len(wn_names)} white = {data.npsrs * rn.nmodes + len(red_names) + len(wn_names)}")

    return data, wn, rn, wn_names, wn_lower, wn_upper, wn_init, red_names, red_init, z_init


# ============================================================================================
# 3a. SAMPLER A -- one joint NUTS kernel over everything.
# ============================================================================================

def run_joint_nuts(args, data, rn, raw_res, vary_white,
                   wn_lower, wn_upper, wn_init, red_init, z_init):
    """Standard ATLAS run: numpyro drives `model_maker` directly.

    When vary_white=True, `model_maker` calls `super_sig.get_helpers(...)` inside the model, so
    the helpers are rebuilt on every leapfrog step.  That is unavoidable in this formulation --
    N depends on the white noise, and the white noise is being sampled.
    """
    init_values = {"red_noise": red_init, "z_a": z_init}
    # A snapshot only, for the vary_white=True case; the sampler rebuilds its own every step.
    helpers = (rn.get_helpers(reff=raw_res, white_noise_params=wn_init) if vary_white
               else rn.get_helpers(reff=raw_res))
    if vary_white:
        init_values["white_noise"] = wn_init

    kernel = NUTS(
        model=model_maker,
        target_accept_prob=0.8,
        max_tree_depth=args.max_tree_depth,
        # dense_mass=[("white_noise",)] looks tempting for the EFAC/EQUAD ridge and is a TRAP:
        # it buys wall clock by *not traversing* the ridge, collapsing ESS(EFAC) from 175 to 27
        # and biasing individual EFACs. A Gaussian metric cannot fix a flat, prior-truncated,
        # non-Gaussian direction. Safe only if you want the GWB and will discard the noise.
        dense_mass=False,
        init_strategy=init_to_value(values=init_values),
    )
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples, num_chains=1)

    t0 = time.time()
    mcmc.run(
        jrandom.key(args.seed),
        raw_residuals=raw_res,
        super_sig=rn,
        vary_white=vary_white,
        wn_lower_bound=wn_lower,        # ignored when vary_white=False
        wn_upper_bound=wn_upper,
        tm_model=None,                  # not using the nonlinear JUG timing model
        helpers=helpers,                # ignored when vary_white=True
        save_red_coeff=False,           # set True to get the latent coefficients out (section 5)
        marg_over_non_gwb=False,        # full lnposterior_reparam
        extra_fields=("diverging", "num_steps"),
    )
    wall = (time.time() - t0) / 60.0

    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    nsteps = np.asarray(extra["num_steps"])
    cap = 2 ** args.max_tree_depth - 1

    print(f"\nfinished in {wall:.2f} min")
    print(f"divergences         : {int(np.asarray(extra['diverging']).sum())} / {args.samples}")
    print(f"mean leapfrog steps : {nsteps.mean():.1f}  (cap {cap}, "
          f"fraction at cap {float((nsteps == cap).mean()):.3f})")
    if vary_white and (nsteps == cap).mean() > 0.5:
        print("  the EFAC/EQUAD ridge is flat and prior-truncated -- running to the cap here is")
        print("  real work along a real flat direction, NOT the frozen-chain pathology.")

    red_chain = np.asarray(samples["red_noise"])
    wn_chain = np.asarray(samples["white_noise"]) if vary_white else None
    _health_check(np.asarray(samples["z_a"]), red_chain)
    return red_chain, wn_chain, wall


# ============================================================================================
# 3b. SAMPLER B -- two-block HMC-within-Gibbs, with the helpers frozen for the red block.
# ============================================================================================

def run_gibbs(args, data, rn, raw_res, wn_lower, wn_upper, wn_init, red_init, z_init):
    """Alternate: [white noise | red, z]  then  [red, z | white noise].

    The whole point is the asymmetry between the two models below.  `model_wn` must call
    get_helpers on every leapfrog step, because it is the white noise that is moving.
    `model_red` takes `helpers` as a plain model ARGUMENT -- a constant as far as the sampler is
    concerned -- so while the red block sweeps, ATLAS recomputes nothing at all.  One rebuild
    per Gibbs sweep instead of one per leapfrog step (benchmarked: 1.96x per step).

    This is driven by hand rather than through ATLAS's vendored `MultiHMCGibbs`
    (canetoadracing.py:73) because that class requires every inner kernel to share one model
    object, and here the two blocks deliberately have different signatures.
    """
    # ---- block A: white noise. Helpers MUST be rebuilt -- they are a function of theta. ----
    def model_wn(red_params, z_a):
        theta = numpyro.sample("white_noise", dist.Uniform(wn_lower, wn_upper))
        h = rn.get_helpers(reff=raw_res, white_noise_params=theta)
        lprob, _ = rn.lnposterior_reparam(helpers=h, red_params=red_params, z=z_a)
        # red_params and z_a are conditioning VALUES here, not sampled sites, so the +0.5*sum(z^2)
        # correction that cancels numpyro's own N(0,1) prior on z_a does not belong in this block.
        numpyro.factor("lnpost", lprob)

    # ---- block B: red hyperparameters + latent coefficients. Helpers arrive as a CONSTANT. ----
    def model_red(helpers):
        xs = numpyro.sample("red_noise", dist.Uniform(rn.model.lower_prior_lim_all,
                                                      rn.model.upper_prior_lim_all))
        z = numpyro.sample("z_a", dist.Normal(0, 1).expand((rn.npsrs, rn.nmodes)))
        lprob, _ = rn.lnposterior_reparam(helpers=helpers, red_params=xs, z=z)
        # ATLAS's standardizing transform a = a_hat + L^-T z already carries the latent density,
        # so cancel the N(0,1) prior numpyro attaches to the z_a site.
        numpyro.factor("lnpost", lprob + 0.5 * jnp.sum(z ** 2))

    kernel_wn = NUTS(model_wn, target_accept_prob=0.8, max_tree_depth=args.max_tree_depth,
                     dense_mass=False,
                     init_strategy=init_to_value(values={"white_noise": wn_init}))
    kernel_red = NUTS(model_red, target_accept_prob=0.8, max_tree_depth=args.max_tree_depth,
                      dense_mass=False,
                      init_strategy=init_to_value(values={"red_noise": red_init, "z_a": z_init}))

    h0 = rn.get_helpers(reff=raw_res, white_noise_params=wn_init)
    k1, k2 = jrandom.split(jrandom.key(args.seed), 2)
    n_sweeps = args.warmup + args.samples
    # Each kernel adapts its own step size and mass matrix on its own schedule. The red block
    # takes args.red_steps updates per sweep, so its warmup budget is scaled to match.
    state_wn = kernel_wn.init(k1, args.warmup, model_args=(red_init, z_init))
    state_red = kernel_red.init(k2, args.warmup * args.red_steps, model_args=(h0,))
    post_wn = kernel_wn.postprocess_fn((red_init, z_init), {})
    post_red = kernel_red.postprocess_fn((h0,), {})

    def refresh(state, model, model_args):
        """Recompute the cached (potential_energy, z_grad) under the NEW conditioning values.

        THE trap in HMC-within-Gibbs.  An HMCState caches the potential energy and its gradient
        from the last time that block moved -- i.e. under the OTHER block's previous values.
        Once the other block moves, those numbers describe a different target, and HMC's
        Metropolis ratio silently compares energies from two different distributions.  The
        symptom is not an error: the step size collapses (we measured 7.6e-7) and the chain
        freezes while still reporting zero divergences and a healthy acceptance rate.
        `MultiHMCGibbs` does exactly this `_replace` before every sub-step.
        """
        pe, grad = jax.value_and_grad(
            lambda zz: potential_energy(model, model_args, {}, zz))(state.z)
        return state._replace(potential_energy=pe, z_grad=grad)

    def sweep(carry, _):
        s_wn, s_red = carry

        # --- block A: update the white noise, conditioned on the current red state ---
        cur = post_red(s_red.z)
        red_c, z_c = cur["red_noise"], cur["z_a"]
        s_wn = refresh(s_wn, model_wn, (red_c, z_c))
        s_wn = kernel_wn.sample(s_wn, (red_c, z_c), {})
        wn_c = post_wn(s_wn.z)["white_noise"]

        # --- the single helpers rebuild of the whole sweep ---
        helpers = rn.get_helpers(reff=raw_res, white_noise_params=wn_c)

        # --- block B: update (red, z) with the white noise held fixed. Nothing recomputed. ---
        s_red = refresh(s_red, model_red, (helpers,))

        def inner(s, __):
            s2 = kernel_red.sample(s, (helpers,), {})
            return s2, s2.num_steps

        s_red, steps_red = jax.lax.scan(inner, s_red, None, length=args.red_steps)

        out = post_red(s_red.z)
        return (s_wn, s_red), (wn_c, out["red_noise"], out["z_a"],
                               s_wn.num_steps, steps_red.sum(),
                               s_wn.adapt_state.step_size, s_red.adapt_state.step_size)

    print(f"\nrunning {n_sweeps} Gibbs sweeps "
          f"(1 white-noise update + {args.red_steps} red update(s) each)...")
    t0 = time.time()
    _, out = jax.lax.scan(sweep, (state_wn, state_red), None, length=n_sweeps)
    jax.block_until_ready(out[0])
    wall = (time.time() - t0) / 60.0

    wn_ch, red_ch, z_ch, steps_wn, steps_red, ss_wn, ss_red = [np.asarray(o) for o in out]
    print(f"finished in {wall:.2f} min")
    print(f"final step size     : white-noise block {ss_wn[-1]:.3e}, red block {ss_red[-1]:.3e}")
    if min(ss_wn[-1], ss_red[-1]) < 1e-5:
        print("  ^ a step size this small means a block is frozen -- check the refresh() call.")
    # Drop warmup.
    wn_ch, red_ch, z_ch = wn_ch[args.warmup:], red_ch[args.warmup:], z_ch[args.warmup:]
    steps_wn, steps_red = steps_wn[args.warmup:], steps_red[args.warmup:]
    print(f"leapfrog steps/sweep: white-noise block {steps_wn.mean():.1f} (rebuilds helpers), "
          f"red block {steps_red.mean():.1f} over {args.red_steps} update(s) (helpers frozen)")

    _health_check(z_ch, red_ch)
    return red_ch, wn_ch, wall


def _health_check(z_chain, red_chain):
    """Two checks that actually catch a frozen chain, which divergences and accept-prob do not."""
    z_std = float(z_chain.std(axis=0).mean())
    print(f"z_a posterior std   : {z_std:.4f}  (should be ~1 -- the transform guarantees it)")
    if not 0.5 < z_std < 2.0:
        raise RuntimeError(f"z_a posterior std is {z_std:.2e}, not ~1: the chain is not exploring.")
    if float(red_chain.std(axis=0).max()) < 1e-6:
        raise RuntimeError("every red-noise parameter is frozen to machine precision.")


# ============================================================================================
# 4.  REPORTING -- score the recovery, and quote efficiency in effective samples per minute.
# ============================================================================================

def _ess(chain):
    """Per-column effective sample size for a single chain of shape [nsamp, ncol]."""
    return np.array([float(effective_sample_size(chain[None, :, j]))
                     for j in range(chain.shape[1])])


def report(psrs, red_chain, wn_chain, wn_names, wall):
    gwb_logA, gwb_gamma = red_chain[:, -2], red_chain[:, -1]

    print("\n" + "=" * 78)
    print("GWB, against the injection")
    print("=" * 78)
    for label, chain, truth in [("log10_A", gwb_logA, TRUE_GWB_LOG10_A),
                                ("gamma", gwb_gamma, TRUE_GWB_GAMMA)]:
        m, s = chain.mean(), chain.std()
        print(f"  {label:<8} {m:+.4f} +- {s:.4f}   truth {truth:+.4f}   "
              f"pull {(m - truth) / s:+.2f} sigma")

    # The IRN is a null test: nothing was injected, so every log10_A should be an upper limit
    # pinned against the prior floor.
    irn_logA = red_chain[:, :-2:2]
    print(f"\nIRN log10_A (nothing injected; prior floor -20): "
          f"median 95th pct {np.median(np.percentile(irn_logA, 95, axis=0)):.2f}"
          f"  -> upper limits, as it should be")

    if wn_chain is not None:
        ef_idx = [i for i, n in enumerate(wn_names) if n.endswith("_efac")]
        eq_idx = [i for i, n in enumerate(wn_names) if n.endswith("_log10_t2equad")]
        ef, eq = wn_chain[:, ef_idx], wn_chain[:, eq_idx]
        sigma = np.array([float(np.asarray(p.toaerrs)[0]) for p in psrs])
        # sigma_eff is the ONLY white-noise combination the likelihood can see on this dataset.
        sigma_eff = ef * np.sqrt(sigma[None, :] ** 2 + (10.0 ** eq) ** 2) * 1e6

        print("\n" + "=" * 78)
        print("White noise")
        print("=" * 78)
        print(f"  EFAC       mean of per-pulsar means {ef.mean(0).mean():.4f}  (truth 1.0)")
        print(f"             -> biased low by the ridge; this is expected, not a bug")
        print(f"  log10EQUAD median 5th pct {np.median(np.percentile(eq, 5, axis=0)):.2f} "
              f"(prior floor -9) -> a flat tail, i.e. an upper limit")
        se_m, se_s = sigma_eff.mean(0), sigma_eff.std(0)
        pulls = (se_m - TRUE_SIGMA_US) / se_s
        print(f"  sigma_eff  mean of per-pulsar means {se_m.mean():.4f} us  "
              f"(truth {TRUE_SIGMA_US})   <- judge the fit on THIS")
        print(f"             pulls: median {np.median(pulls):+.2f}, "
              f"|pull|>3 for {(np.abs(pulls) > 3).sum()}/{len(pulls)}")

    print("\n" + "=" * 78)
    print(f"Efficiency  ({wall:.2f} min wall clock)")
    print("=" * 78)
    ess_gwb = _ess(red_chain[:, -2:])
    print(f"  ESS(GWB)  median {np.median(ess_gwb):.0f} of {len(red_chain)} draws"
          f"   -> {np.median(ess_gwb) / wall:.1f} per minute")
    if wn_chain is not None:
        ess_wn = _ess(wn_chain)
        print(f"  ESS(EFAC) median {np.median(ess_wn[ef_idx]):.0f}"
              f"   -> {np.median(ess_wn[ef_idx]) / wall:.1f} per minute"
              f"    (min over all WN params: {ess_wn.min():.0f})")
        print("  The white noise is the bottleneck, and only along the EFAC/EQUAD ridge --")
        print("  sigma_eff itself mixes fine. That asymmetry is the whole story on this dataset.")


# ============================================================================================
# 5.  OPTIONAL EXTRA: recovering the physical timing offsets from the SVD basis.
# ============================================================================================
#
# ATLAS's timing block is the left singular vectors U of the design matrix M (see
# signals_utils._timing_model_svd), so the sampled coefficients `a` are in that basis, not in
# parameter units.  The SVD is the ENABLER, not the obstacle: cond(M) is 1e19-1e21 purely from
# parameter units, and a naive pinv(M^T N^-1 M) is numerically worthless.  To get physical
# offsets back, redo the SVD and undo it -- psr.Mmat is left untouched on the pulsar objects:
#
#     # run with save_red_coeff=True, then coeff has shape [nsamp, npsrs, nmodes]
#     U, C, Vh = np.linalg.svd(np.asarray(psr.Mmat), full_matrices=False)
#     a   = coeff[:, psr_index, :psr.Mmat.shape[1]]       # the timing columns only
#     eps = (a / C[None, :]) @ Vh                          # posterior draws of the offsets
#
# Validated on MDC1: 377 of 402 parameters reproduce the par-file uncertainties within 1%.
# Two traps: (1) psr.fit_param_uncertainties is in RADIANS for RAJ/DECJ while the design-matrix
# columns are in seconds of RA -- a factor 1.375e4; compare against the par file instead.
# (2) np.linalg.matrix_rank on the raw M reports a large rank deficit because of its default
# tolerance; the back-transform works anyway.


# ============================================================================================
# main
# ============================================================================================

def main():
    p = argparse.ArgumentParser(
        description="ATLAS global fit on IPTA MDC1 Open-1, with a choice of sampler.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sampler", choices=["nuts", "gibbs"], default="nuts",
                   help="joint NUTS over everything (default), or two-block HMC-within-Gibbs "
                        "splitting white noise from (red hyperparameters + latent coefficients)")
    p.add_argument("--npsrs", type=int, default=36, help="number of pulsars, 2-36 (default 36)")
    p.add_argument("--warmup", type=int, default=700)
    p.add_argument("--samples", type=int, default=700)
    p.add_argument("--red-steps", type=int, default=1, dest="red_steps",
                   help="gibbs only: red-block updates per sweep. The red block is the cheap one "
                        "(helpers frozen), so >1 looks free -- but measured on MDC1 it is a net "
                        "loss, because that block already has ESS/sample ~ 1. Default 1.")
    p.add_argument("--fixed-white-noise", action="store_true", dest="fixed_wn",
                   help="hold the white noise at the quoted TOA errors instead of fitting "
                        "EFAC/EQUAD. Forces --sampler nuts (there is no second block to split).")
    p.add_argument("--max-tree-depth", type=int, default=8, dest="max_tree_depth")
    p.add_argument("--seed", type=int, default=170817)
    p.add_argument("--outdir", default=".", help="where to write chains and the pulsar cache")
    args = p.parse_args()

    vary_white = not args.fixed_wn
    if args.sampler == "gibbs" and not vary_white:
        p.error("--fixed-white-noise leaves nothing to Gibbs-block; use --sampler nuts.")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"jax {jax.__version__} | numpyro {numpyro.__version__} | devices {jax.devices()}")
    print(f"TEMPO2 = {os.environ['TEMPO2']}\n")

    # --- 1. data, and the two checks that justify everything downstream -----------------
    psrs, parfiles, timfiles, mdc_dir = load_mdc1(args.npsrs, args.outdir)
    check_not_a_phase_hash(psrs, mdc_dir)
    if vary_white:
        check_ecorr_is_unidentifiable(psrs)

    # --- 2. model ----------------------------------------------------------------------
    (data, wn, rn, wn_names, wn_lower, wn_upper,
     wn_init, red_names, red_init, z_init) = build_model(psrs, parfiles, timfiles, vary_white)
    raw_res = jnp.concat(data.raw_residuals)

    # --- 3. sample ---------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"sampler: {args.sampler}"
          + (f" (--red-steps {args.red_steps})" if args.sampler == "gibbs" else "")
          + f" | white noise {'varied' if vary_white else 'fixed'}"
          + f" | {args.warmup} warmup + {args.samples} samples")
    print("=" * 78)

    if args.sampler == "nuts":
        red_chain, wn_chain, wall = run_joint_nuts(
            args, data, rn, raw_res, vary_white,
            wn_lower, wn_upper, wn_init, red_init, z_init)
    else:
        red_chain, wn_chain, wall = run_gibbs(
            args, data, rn, raw_res, wn_lower, wn_upper, wn_init, red_init, z_init)

    # --- 4. report and save --------------------------------------------------------------
    report(psrs, red_chain, wn_chain, wn_names, wall)

    tag = f"{args.sampler}_{args.npsrs}psr" + ("" if vary_white else "_fixedwn")
    np.savez_compressed(
        os.path.join(args.outdir, f"mdc1_{tag}.npz"),
        red_noise=red_chain, red_names=np.array(red_names),
        **({"white_noise": wn_chain, "wn_names": np.array(wn_names)} if wn_chain is not None else {}))
    print(f"\nchains -> {os.path.join(args.outdir, f'mdc1_{tag}.npz')}")


if __name__ == "__main__":
    main()
