"""Phase 9 -- honest zonal vs modal vs ML reconstruction comparison.

ML (ridge slopes->Zernike) is trained on MEASURED slopes from several
independent screen realizations (diverse, full-rank training data -- a single
frozen-flow sequence is rank-deficient) against clean target coefficients, then
tested on a held-out realization, exactly as modal/zonal are evaluated.
"""
import numpy as np, time
from dataclasses import replace
from scipy.ndimage import map_coordinates
from aowfs import SimConfig
from aowfs.io import from_synthetic
from aowfs.optics import displacement_to_gradient
from aowfs.recon import Centroider, ZernikeReconstructor, FriedZonalReconstructor, MLReconstructor

NM = 45
base = SimConfig()
modal_ref = None


def measure(seed, nf):
    cfg = replace(base, n_frames=nf, turbulence=replace(base.turbulence, seed=seed))
    ds = from_synthetic(cfg); seq, truth, geom = ds.sequence, ds.truth, ds.sequence.geometry
    cen = Centroider(cfg, geom, seq.reference_centroids)
    rec = ZernikeReconstructor(cfg, geom, n_modes=NM)
    g_meas = np.array([displacement_to_gradient(cen.measure(seq.frames[i]), cfg) for i in range(nf)])
    g_true = np.array([truth.true_slopes[i] for i in range(nf)])
    a_true = np.array([rec.coeffs(g_true[i]) for i in range(nf)])
    return cfg, seq, truth, geom, rec, g_meas, g_true, a_true


# --- training set: several independent realizations -----------------------
TRAIN_SEEDS = [11, 22, 33, 44, 55]
Gm, Gt, A = [], [], []
for s in TRAIN_SEEDS:
    _, _, _, _, _, gm, gt, at = measure(s, 200)
    Gm.append(gm); Gt.append(gt); A.append(at)
Gm = np.concatenate(Gm); Gt = np.concatenate(Gt); A = np.concatenate(A)
print(f"training samples: {Gm.shape[0]} (from {len(TRAIN_SEEDS)} realizations), input dim {2*Gm.shape[1]}")

# --- test realization (held out) ------------------------------------------
cfg, seq, truth, geom, modal, gm_te, gt_te, a_te = measure(1234, 300)
zon = FriedZonalReconstructor(cfg, geom)
P, p, pm = cfg.pupil_grid_px, cfg.screen_px_per_subap, geom.pupil_mask
c = (P - 1) / 2; yy, xx = np.mgrid[0:P, 0:P]; rho = np.sqrt((xx-c)**2+(yy-c)**2)/(P/2); interior = rho < 0.85
coord = (np.arange(P)+0.5)/p; cc, rr = np.meshgrid(coord, coord)

# alpha tuning on a validation split of the training set
vi = np.arange(int(len(A)*0.85), len(A)); ti = np.arange(0, int(len(A)*0.85))
best = None
for al in [1e-2, 1e-1, 1e0, 1e1, 1e2]:
    ml = MLReconstructor(cfg, geom, n_modes=NM, alpha=al).fit(Gm[ti], A[ti])
    err = np.mean([np.sum((ml.coeffs(Gm[i]) - A[i])**2) for i in vi])
    if best is None or err < best[1]:
        best = (al, err)
alpha = best[0]
ml = MLReconstructor(cfg, geom, n_modes=NM, alpha=alpha).fit(Gm, A)
ml_noiseless = MLReconstructor(cfg, geom, n_modes=NM, alpha=1e-2).fit(Gt, A)
print(f"ML ridge alpha (tuned) = {alpha:g}")


def rms_vs_truth(maps):
    e, t0 = [], []
    for k in range(maps.shape[0]):
        d = (maps[k] - truth.phase_screens[k])[interior]; e.append(np.var(d - d.mean()))
        tt = truth.phase_screens[k][interior]; t0.append(np.var(tt - tt.mean()))
    return np.sqrt(np.mean(e)), np.sqrt(np.mean(t0))


def zonal_map(g):
    return map_coordinates(np.nan_to_num(zon.to_grid(zon.reconstruct(g))), [rr, cc], order=1, mode="nearest")


modal_maps = np.array([modal.reconstruct(gm_te[i]) for i in range(300)])
ml_maps = np.array([ml.reconstruct(gm_te[i]) for i in range(300)])
zon_maps = np.array([zonal_map(gm_te[i]) for i in range(300)])
mz, trms = rms_vs_truth(modal_maps); mlr, _ = rms_vs_truth(ml_maps); zr, _ = rms_vs_truth(zon_maps)


def tmatvec(M, s, n=3000):
    for _ in range(100): M @ s
    t = time.perf_counter()
    for _ in range(n): M @ s
    return (time.perf_counter() - t) / n * 1e6


sflat = np.concatenate([gm_te[0][:, 0], gm_te[0][:, 1]])
print("\n=== Phase 9: reconstruction comparison (held-out realization, noisy) ===")
print(f"interior truth RMS: {trms:.3f} rad   (lower RMS err = better)")
print(f"{'method':<14}{'RMS err [rad]':>14}{'rel %':>8}{'infer us':>10}{'DOF':>7}")
print(f"{'zonal':<14}{zr:>14.3f}{100*zr/trms:>8.1f}{tmatvec(zon.recon_matrix,sflat):>10.2f}{zon.n_corners:>7}")
print(f"{'modal':<14}{mz:>14.3f}{100*mz/trms:>8.1f}{tmatvec(modal.recon_matrix,sflat):>10.2f}{NM:>7}")
print(f"{'ML (ridge)':<14}{mlr:>14.3f}{100*mlr/trms:>8.1f}{tmatvec(ml.recon_matrix,sflat):>10.2f}{NM:>7}")

# noiseless tie-check
em, el = [], []
for i in range(300):
    dm = (modal.reconstruct(gt_te[i]) - truth.phase_screens[i])[pm]; em.append(np.var(dm-dm.mean()))
    dl = (ml_noiseless.reconstruct(gt_te[i]) - truth.phase_screens[i])[pm]; el.append(np.var(dl-dl.mean()))
print(f"\nnoiseless tie-check residual var: modal {np.mean(em):.4f}  ML {np.mean(el):.4f} rad^2")

# Noll trend for ML (noiseless), modes 5..35
print("\nNoll-variance trend (ML, noiseless residual var rad^2):")
for nm in [5, 10, 20, 35]:
    mlt = MLReconstructor(cfg, geom, n_modes=nm, alpha=1e-2)
    At = np.array([ZernikeReconstructor(cfg, geom, n_modes=nm).coeffs(Gt[i]) for i in range(0, len(Gt), 3)])
    mlt.fit(Gt[::3], At)
    v = []
    for i in range(0, 300, 3):
        d = (mlt.reconstruct(gt_te[i]) - truth.phase_screens[i])[pm]; v.append(np.var(d-d.mean()))
    print(f"  {nm:>2} modes: {np.mean(v):.3f}")
