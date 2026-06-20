"""repro.py -- shared reproducibility metadata for the fully parabolic-parabolic
3D Keller-Segel run drivers (validation-closure task section 8).

Every production run must record: exact sys.argv, a command.txt, git commit, UTC
timestamp, hostname, Python/numpy/JAX/jaxlib versions, JAX device, and (the caller
adds) all PDE parameters, M/N/K_dyn/K_test, tau/T/cadence, seed, injection kernel,
buffer factor + capacity, max occupancy, population_control=false, runtime, and the
abort status/reason.  This module centralizes the environment-derived fields so the
radial and tetra drivers record an identical, complete record.
"""
import os, sys, json, socket, subprocess, datetime


def git_hash(here):
    try:
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_dirty(here):
    """True if the working tree differs from HEAD (uncommitted changes present)."""
    try:
        out = subprocess.check_output(["git", "-C", here, "status", "--porcelain"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return None


def pkg_versions():
    import numpy as np
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        import jax
        out["jax"] = jax.__version__
        out["devices"] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    try:
        import jaxlib
        out["jaxlib"] = jaxlib.__version__
    except Exception:
        pass
    return out


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def env_record(here):
    """Environment-derived reproducibility fields (caller merges in run params)."""
    return dict(git=git_hash(here), git_dirty=git_dirty(here),
                utc_timestamp=utc_now().isoformat(), hostname=socket.gethostname(),
                argv=list(sys.argv), versions=pkg_versions(),
                jax_platforms=os.environ.get("JAX_PLATFORMS", ""),
                jax_enable_x64=os.environ.get("JAX_ENABLE_X64", ""))


def write_command_txt(run_dir):
    """Persist the exact command line for one-line reproduction."""
    with open(os.path.join(run_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")


def dump(run_dir, manifest, cfg=None):
    """Write manifest.json (+ optional config_used.json) and command.txt."""
    os.makedirs(run_dir, exist_ok=True)
    write_command_txt(run_dir)
    json.dump(manifest, open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)
    if cfg is not None:
        json.dump(cfg, open(os.path.join(run_dir, "config_used.json"), "w"), indent=2)
