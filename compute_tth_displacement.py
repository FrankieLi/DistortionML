import os

import logging
import time
import h5py

from matplotlib import pyplot as plt

import numpy as np

from tqdm import tqdm

from hexrd import constants as ct
from hexrd import gridutil
from hexrd import instrument
from hexrd.transforms import xfcapi

import pinhole_camera_module as phutil

logger = logging.getLogger(__name__)


# =============================================================================
# %% PARAMETERS
# ============================================================================='
resources_path = './resources'
ref_config = 'reference_instrument.hexrd'

# geometric paramters for source and pinhole (typical TARDIS)
#
# !!! All physical dimensions in mm
#
# !!! This is the minimal set we'd like to do the MCMC over; would like to also
#     include detector translation and at least rotation about its own normal.
rho = 32.                 # source distance
ph_radius = 0.200         # pinhole radius
ph_thickness = 0.100      # pinhole thickness
layer_standoff = 0.150    # offset to sample layer
layer_thickness = 0.01    # layer thickness

# Target voxel size
voxel_size = 0.015


# =============================================================================
# %% OBJECT INSTANTIATION
# =============================================================================

# load instrument and grab the detecor (one for now)
instr = instrument.HEDMInstrument(
    h5py.File(os.path.join(resources_path, ref_config), 'r')
)
det_key, det = next(iter(instr.detectors.items()))  # !!! only one
bhat = np.atleast_2d(instr.beam_vector)

# generate voxel coordinates in within critial radius
rho_crit = phutil.compute_critical_voxel_radius(
    layer_standoff, ph_radius, ph_thickness
)
rho_crit -= voxel_size  # make sure "voxel" is within the critical radius

# also need the reference pixel angles as computed from the origin
ref_ptth, ref_peta = det.pixel_angles()

# generate voxel coordinates, mask, and flatten them
voxel_generator_xy = gridutil.make_tolerance_grid(
    voxel_size, 2*rho_crit, 1, adjust_window=True
)[1]
vx, vy = np.meshgrid(voxel_generator_xy, voxel_generator_xy)
rhoc_mask = np.sum(np.stack([vx**2 + vy**2], axis=0), axis=0) <= rho_crit**2
vx = vx[rhoc_mask].flatten()
vy = vy[rhoc_mask].flatten()

# FIXME: currently only doing a SINGLE LAYER;
#        will need to compute layer-specific critical radii
vcrds = np.vstack([vx, vy, np.ones_like(vx)*layer_standoff]).T


# =============================================================================
# %% GRAND LOOP
# =============================================================================
def grand_loop(coords, detector, bhat, rho, pinhole_radius, pinhole_thickness,
               perf_acc=None):
    #coords = coords[:10] # limit...

    setup_t0 = time.perf_counter_ns()
    # need the cartesian pixel coordinates
    py, px = detector.pixel_coords
    pixel_xys = np.vstack([px.flatten(), py.flatten()]).T

    # loop over voxels to aggregate pixel angles and contributing voxel count
    master_ptth = np.zeros(detector.shape, dtype=float)
    voxel_count = np.zeros(detector.shape, dtype=float)
    reduced_rmat = np.ascontiguousarray(detector.rmat[:, :2].T)  # transpose for np.dot
    setup_t1 = time.perf_counter_ns()

    loop_t0 = time.perf_counter_ns()
    acc_cobv = 0
    acc_phc = 0
    acc_other = 0
    acc_xy2g = 0
    for iv, coord in enumerate(tqdm(coords)):
        # need new beam vector from curent voxel coordinate
        cobv_t0 = time.perf_counter_ns()
        new_bv = phutil.compute_offset_beam_vector(bhat, rho, np.atleast_2d(coord))
        cobv_t1 = time.perf_counter_ns()
        det.bvec = new_bv

        # mask detector pixels
        phc_t0 = time.perf_counter_ns()
        mask = phutil.pinhole_constraint(
            pixel_xys, np.array(coord),
            reduced_rmat, detector.tvec,
            pinhole_radius, pinhole_thickness
        )  # no reshape # .reshape(det.shape)
        phc_t1 = time.perf_counter_ns()

        other_t0 = time.perf_counter_ns()
        if np.any(mask):
            # compute pixel angles that satisfy the pinhole constraint
            #print(f"it {iv}: mask has {np.sum(mask)} pixels set. {pixel_xys.shape} {mask.shape}")
            reduced_xys = pixel_xys[mask, :]
            mask = mask.reshape(detector.shape)
            ptth = np.nan*np.ones(detector.shape)
            xy2g_t0 = time.perf_counter_ns()
            angs, _ = xfcapi.detectorXYToGvec(
                reduced_xys, detector.rmat, ct.identity_3x3,
                detector.tvec, ct.zeros_3, np.array(coord),
                beamVec=new_bv)
            acc_xy2g += time.perf_counter_ns() - xy2g_t0
            ptth[mask] = angs[0]

            master_ptth = np.nansum(
                np.stack([master_ptth, ptth], axis=0),
                axis=0
            )
            voxel_count += mask
        other_t1 = time.perf_counter_ns()
        acc_cobv += cobv_t1 - cobv_t0
        acc_phc += phc_t1 - phc_t0
        acc_other += other_t1 - other_t0

    loop_t1 = time.perf_counter_ns()

    if perf_acc is not None:
        perf_acc['gl_setup'] = perf_acc.get('gl_setup', 0) + (setup_t1 - setup_t0)
        perf_acc['gl_loop'] = perf_acc.get('gl_loop', 0) + (loop_t1 - loop_t0)
        perf_acc['gl_computeoffset'] = perf_acc.get('gl_compute_offset', 0) + acc_cobv
        perf_acc['gl_constraint'] = perf_acc.get('gl_constraint', 0) + acc_phc
        perf_acc['gl_other'] = perf_acc.get('gl_other', 0) + acc_other
        perf_acc['gl_xy2g'] = perf_acc.get('gl_xy2g', 0) + acc_xy2g
    return master_ptth, voxel_count

perf_results = dict()

master_ptth, voxel_count = grand_loop(vcrds, det, bhat, rho, ph_radius, ph_thickness, perf_acc=perf_results)

for key, val in perf_results.items():
    print(f"{key}: {float(val)/10**9}")

# apply panel buffer if applicable
corr = np.array(master_ptth/voxel_count - ref_ptth, dtype=np.float32)
if det.panel_buffer.ndim == 2:
    corr[~det.panel_buffer] = np.nan

# =============================================================================
# %% plotting
# =============================================================================

fig, ax = plt.subplots()
mappable = ax.imshow(np.degrees(corr), cmap=plt.cm.gnuplot2)
fig.colorbar(mappable)
plt.show()

"""
# This custom formatter removes trailing zeros, e.g. "1.0" becomes "1", and
# then adds a percent sign.
def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s}" if plt.rcParams["text.usetex"] else f"{s}"

fig, ax = plt.subplots()
mappable = ax.imshow(np.degrees(pv['intensities']), cmap=plt.cm.inferno, extent=pv['extent'])
ax.axis('auto')
cbar = fig.colorbar(mappable)
cbar.set_label(r'$2\theta_s-2\theta_n$ [deg]')
fig.suptitle(r'$2\theta_s-2\theta_n$, $20\mu\mathrm{m}$ voxels, $150\mu\mathrm{m}$ standoff')
ax.set_xlim(10, 110)
ax.set_xlabel(r'nominal Bragg angle, $2\theta_n$ [deg]')
ax.set_ylabel(r'azimuth, $\eta_n$ [deg]')
CS = ax.contour(np.array(pv['tth_coordinates']),
                np.array(pv['eta_coordinates']),
                np.degrees(pv['intensities']), colors='w',
                levels=[0.2, 0.3, 0.5, 0.8, 1.1])
ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=12)
ax.grid(True)
"""
