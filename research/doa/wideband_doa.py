"""
Wideband DOA estimation for low-altitude UAV signals -- minimal viable experiment.

Goal: quantify how coherent multipath and finite bandwidth interact, and show
where classical wideband estimators (ISM / CSM) break down.

Model
-----
One emitter (the UAV) reaches the array through J coherent propagation paths.
Path j has azimuth phi_j, excess delay tau_j, complex amplitude a_j and Doppler
shift fd_j.  At sub-band centre frequency f_k and snapshot time t_n the array
output is

    x_kn = ( sum_j a_j exp(-j2*pi*f_k*tau_j) exp(+j2*pi*fd_j*t_n) v(f_k, phi_j) ) s_kn + n_kn

with s_kn ~ CN(0,1) the (common) emitter waveform and v(f, phi) the array
steering vector at frequency f.  Because all paths share s_kn they are fully
coherent, so the per-sub-band signal covariance has rank 1 no matter how many
paths there are -- that is precisely the difficulty.

Estimators compared
-------------------
NB   : narrowband MUSIC on the centre-frequency sub-band (model order 1)
ISM  : incoherent MUSIC, spectra averaged over sub-bands
CSM  : coherent signal-subspace (focusing to f0, preliminaries from ISM)
SS   : spatial smoothing MUSIC (recovers the coherent paths, order = J)

Each estimator returns a single azimuth (the source direction); SS returns the
strongest of its peaks.

Every estimator is scored against the true direct-path azimuth.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

C_LIGHT = 299_792_458.0


# --------------------------------------------------------------------------- #
# array
# --------------------------------------------------------------------------- #
def ula_positions(m: int, f0: float, spacing_lambda: float = 0.5) -> np.ndarray:
    """Uniform linear array along y, broadside boresight at 0 deg."""
    d = spacing_lambda * C_LIGHT / f0
    return np.arange(m, dtype=float) * d


def steering(pos: np.ndarray, f: float, phi_rad: np.ndarray) -> np.ndarray:
    """Steering matrix, shape (M, len(phi)).  Broadside = 0 rad."""
    u = np.sin(np.atleast_1d(phi_rad))
    return np.exp(-2j * np.pi * f * np.outer(pos, u) / C_LIGHT)


# --------------------------------------------------------------------------- #
# scenario
# --------------------------------------------------------------------------- #
@dataclass
class Path:
    phi_deg: float
    tau_ns: float
    amp: float
    phase_deg: float = 0.0


@dataclass
class Scenario:
    f0_ghz: float = 2.4
    n_elem: int = 16
    phi0_deg: float = 30.0
    # direct path + three coherent echoes (ground bounce, two facades)
    paths: list[Path] = field(
        default_factory=lambda: [
            Path(30.0, 0.0, 1.00, 0.0),     # direct
            Path(55.0, 13.0, 0.50, 40.0),   # ground bounce  (4 m longer)
            Path(10.0, 25.0, 0.35, -110.0), # near facade
            Path(-5.0, 45.0, 0.22, 75.0),   # far facade
        ]
    )
    velocity_mps: float = 15.0      # UAV radial-ish motion, along broadside
    t_obs_s: float = 0.10
    n_snap: int = 100
    snr_db: float = -5.0            # per element, per sub-band, direct path
    bandwidth_mhz: float = 60.0
    n_sub: int = 20
    n_mc: int = 30


def build_freqs(sc: Scenario) -> np.ndarray:
    f0 = sc.f0_ghz * 1e9
    bw = sc.bandwidth_mhz * 1e6
    return f0 + np.linspace(-bw / 2, bw / 2, sc.n_sub)


def doppler_hz(phi_deg: float, velocity: float, f0: float) -> float:
    """Projection of the UAV velocity onto the path direction."""
    lam = C_LIGHT / f0
    return velocity / lam * np.cos(np.deg2rad(phi_deg))


def simulate_covariances(sc: Scenario, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return R (n_sub, M, M) and the sub-band frequencies."""
    f0 = sc.f0_ghz * 1e9
    pos = ula_positions(sc.n_elem, f0)
    freqs = build_freqs(sc)
    M = sc.n_elem
    J = len(sc.paths)

    phis = np.array([p.phi_deg for p in sc.paths])
    taus = np.array([p.tau_ns for p in sc.paths]) * 1e-9
    amps = np.array([p.amp for p in sc.paths]) * np.exp(1j * np.deg2rad([p.phase_deg for p in sc.paths]))
    fds = np.array([doppler_hz(p.phi_deg, sc.velocity_mps, f0) for p in sc.paths])

    t = np.linspace(0.0, sc.t_obs_s, sc.n_snap)
    # snapshot.  Doppler is the same for every sub-band (narrow fractional BW).
    snap_phase = np.exp(2j * np.pi * np.outer(t, fds))          # (N, J)

    # noise power: direct path carries unit power per element
    sigma2 = 10 ** (-sc.snr_db / 10.0)

    R = np.zeros((len(freqs), M, M), dtype=complex)
    for k, fk in enumerate(freqs):
        V = steering(pos, fk, np.deg2rad(phis))                 # (M, J)
        V = V * amps[None, :] * np.exp(-2j * np.pi * fk * taus)[None, :]
        # (N, M) sum over paths for every snapshot
        atom = (snap_phase @ V.T)                               # (N, M)
        s = (rng.standard_normal(sc.n_snap) + 1j * rng.standard_normal(sc.n_snap)) / np.sqrt(2)
        X = atom * s[:, None] + np.sqrt(sigma2 / 2) * (
            rng.standard_normal((sc.n_snap, M)) + 1j * rng.standard_normal((sc.n_snap, M))
        )
        # (1/N) sum_n x_n x_n^H with x_n = X[n, :]  (column vector per snapshot)
        R[k] = X.T @ X.conj() / sc.n_snap
    return R, freqs


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def music_spectrum(R: np.ndarray, pos: np.ndarray, f: float, grid_rad: np.ndarray,
                   n_sources: int) -> tuple[np.ndarray, np.ndarray]:
    M = R.shape[0]
    w, E = np.linalg.eigh(R)
    order = np.argsort(w)[::-1]
    E = E[:, order]
    En = E[:, n_sources:]                       # noise subspace
    if En.shape[1] == 0:
        En = E[:, -1:]
    A = steering(pos, f, grid_rad)
    proj = En.conj().T @ A
    P = 1.0 / np.maximum(np.sum(np.abs(proj) ** 2, axis=0), 1e-15)
    return P, E


def dominant_peak(P: np.ndarray, grid_rad: np.ndarray) -> float:
    return float(np.rad2deg(grid_rad[int(np.argmax(P))]))


def top_peaks(P: np.ndarray, grid_rad: np.ndarray, n: int, min_sep_deg: float = 5.0) -> np.ndarray:
    idx = np.argsort(P)[::-1]
    picked: list[int] = []
    for i in idx:
        if all(abs(np.rad2deg(grid_rad[i] - grid_rad[j])) >= min_sep_deg for j in picked):
            picked.append(int(i))
        if len(picked) >= n:
            break
    return np.rad2deg(grid_rad[picked])


def spatial_smooth(R: np.ndarray, m_sub: int) -> np.ndarray:
    """Forward-only spatial smoothing of one covariance matrix."""
    M = R.shape[0]
    L = M - m_sub + 1
    Rs = np.zeros((m_sub, m_sub), dtype=complex)
    for l in range(L):
        Rs += R[l : l + m_sub, l : l + m_sub]
    return Rs / L


def est_nb(R: np.ndarray, freqs: np.ndarray, pos: np.ndarray, grid: np.ndarray) -> float:
    k0 = len(freqs) // 2
    P, _ = music_spectrum(R[k0], pos, freqs[k0], grid, n_sources=1)
    return dominant_peak(P, grid)


def est_ism(R: np.ndarray, freqs: np.ndarray, pos: np.ndarray, grid: np.ndarray) -> float:
    acc = np.zeros(len(grid))
    for k, fk in enumerate(freqs):
        P, _ = music_spectrum(R[k], pos, fk, grid, n_sources=1)
        acc += P / P.max()
    return dominant_peak(acc, grid)


def est_csm(R: np.ndarray, freqs: np.ndarray, pos: np.ndarray, grid: np.ndarray,
            n_paths: int) -> float:
    f0 = freqs[len(freqs) // 2]
    # preliminaries: top peaks of the incoherent spectrum
    acc = np.zeros(len(grid))
    for k, fk in enumerate(freqs):
        P, _ = music_spectrum(R[k], pos, fk, grid, n_sources=1)
        acc += P / P.max()
    prelim = top_peaks(acc, grid, n=n_paths)

    A0 = steering(pos, f0, np.deg2rad(prelim))
    Rf = np.zeros_like(R[0])
    for k, fk in enumerate(freqs):
        Ak = steering(pos, fk, np.deg2rad(prelim))
        # T = A0 Ak^H (Ak Ak^H)^-1   (least-squares focusing onto f0)
        T = A0 @ Ak.conj().T @ np.linalg.pinv(Ak @ Ak.conj().T)
        Rf += T @ R[k] @ T.conj().T
    Rf /= len(freqs)
    P, _ = music_spectrum(Rf, pos, f0, grid, n_sources=n_paths)
    return dominant_peak(P, grid)


def est_ss(R: np.ndarray, freqs: np.ndarray, pos: np.ndarray, grid: np.ndarray,
           n_paths: int) -> float:
    m_sub = max(8, pos.size - 6)
    Rs = np.zeros((m_sub, m_sub), dtype=complex)
    for k, fk in enumerate(freqs):
        Rs += spatial_smooth(R[k], m_sub)
    Rs /= len(freqs)
    P, _ = music_spectrum(Rs, pos[:m_sub], freqs[len(freqs) // 2], grid, n_sources=n_paths)
    return dominant_peak(P, grid)


# --------------------------------------------------------------------------- #
# experiment driver
# --------------------------------------------------------------------------- #
def run_point(sc: Scenario, grid: np.ndarray, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    pos = ula_positions(sc.n_elem, sc.f0_ghz * 1e9)
    errs: dict[str, list[float]] = {k: [] for k in ("NB", "ISM", "CSM", "SS")}
    for _ in range(sc.n_mc):
        R, freqs = simulate_covariances(sc, rng)
        truth = sc.phi0_deg
        errs["NB"].append(est_nb(R, freqs, pos, grid) - truth)
        errs["ISM"].append(est_ism(R, freqs, pos, grid) - truth)
        errs["CSM"].append(est_csm(R, freqs, pos, grid, n_paths=len(sc.paths)) - truth)
        errs["SS"].append(est_ss(R, freqs, pos, grid, n_paths=len(sc.paths)) - truth)
    out: dict[str, float] = {}
    for k, v in errs.items():
        a = np.asarray(v)
        out[k] = float(np.sqrt(np.mean(a ** 2)))
    return out


def main() -> None:
    t0 = time.time()
    grid = np.deg2rad(np.arange(-90.0, 90.0 + 1e-9, 0.5))
    bandwidths = [5, 10, 20, 40, 60, 80]

    results = {"meta": {}, "sweeps": {}}

    print("=== sweep 1: azimuth RMSE vs bandwidth (multipath, moving UAV) ===")
    rows = []
    for bw in bandwidths:
        sc = Scenario(bandwidth_mhz=bw, velocity_mps=15.0)
        r = run_point(sc, grid, seed=42)
        r["bw"] = bw
        rows.append(r)
        print(f"  B={bw:3d} MHz  NB={r['NB']:6.2f}  ISM={r['ISM']:6.2f}  "
              f"CSM={r['CSM']:6.2f}  SS={r['SS']:6.2f}   (deg RMSE)")
    results["sweeps"]["bw_moving"] = rows

    print("=== sweep 2: static UAV (v=0) -- delay diversity only ===")
    rows2 = []
    for bw in bandwidths:
        sc = Scenario(bandwidth_mhz=bw, velocity_mps=0.0)
        r = run_point(sc, grid, seed=42)
        r["bw"] = bw
        rows2.append(r)
        print(f"  B={bw:3d} MHz  NB={r['NB']:6.2f}  ISM={r['ISM']:6.2f}  "
              f"CSM={r['CSM']:6.2f}  SS={r['SS']:6.2f}")
    results["sweeps"]["bw_static"] = rows2

    print("=== sweep 3: single-path control (no multipath) ===")
    rows3 = []
    for bw in bandwidths:
        sc = Scenario(bandwidth_mhz=bw, velocity_mps=15.0)
        sc.paths = [Path(30.0, 0.0, 1.0, 0.0)]
        r = run_point(sc, grid, seed=42)
        r["bw"] = bw
        rows3.append(r)
        print(f"  B={bw:3d} MHz  NB={r['NB']:6.2f}  ISM={r['ISM']:6.2f}  "
              f"CSM={r['CSM']:6.2f}  SS={r['SS']:6.2f}")
    results["sweeps"]["single_path"] = rows3

    results["meta"] = {"elapsed_s": time.time() - t0, "grid_step_deg": 0.5}
    with open("doa_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\ndone in {time.time() - t0:.1f} s -> doa_results.json")


if __name__ == "__main__":
    main()
