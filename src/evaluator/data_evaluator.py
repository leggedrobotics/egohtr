from __future__ import annotations

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from abc import ABC

import torch
from src.config.configs import SequenceContext, PipelineConfig
from scipy.interpolate import interp1d
from src.body_model.smplx_helper import (
    EVAL_JOINT_NAMES, 
    SMPLX_JOINT_NAMES, 
    LOWER_LIMBS_JOINT_NAMES, 
    HEAD_JOINT_NAMES, 
    TRUNK_JOINT_NAMES, 
    UPPER_LIMBS_JOINT_NAMES,
) 
from src.evaluator.eval_helper import batch_compute_similarity_transform_torch, compute_similarity_transform_torch

METRICS_UNIT = {
    "mpjpe": "m",
    "pa_mpjpe": "m",
    "jitter": "km/s^3",
}

JOINT_NUMBERS = {
    "all_joints": len(EVAL_JOINT_NAMES),
    "upper_limbs": len(UPPER_LIMBS_JOINT_NAMES),
    "lower_limbs": len(LOWER_LIMBS_JOINT_NAMES),
    "trunk": len(TRUNK_JOINT_NAMES),
    "head": len(HEAD_JOINT_NAMES),  
}

class Timeseries:
    """ A simple wrapper for time-aligned data with interpolation support."""

    time: float
    values: dict[np.ndarray]

    def __init__(self, time: float, key: str, value: np.ndarray):
        assert time.shape[0] == value.shape[0], "Time and value must have the same length."
        self.time = time
        self.values = {key: value}
    
    def copy(self):
        new_ts = Timeseries(self.time.copy(), "dummy", np.zeros_like(self.time))
        new_ts.values = {key: val.copy() for key, val in self.values.items()}
        return new_ts

    def append(self, key: str, value: np.ndarray):
        """Append or overwrites a timeseries value."""
        assert value.shape[0] == self.time.shape[0], "Value must have the same length as time."
        self.values[key] = value

    def interpolate(self, new_time: np.ndarray):
        """Returns a new Timeseries object aligned to new_time."""
        new_ts = Timeseries(new_time, "dummy", np.zeros_like(new_time))
        new_ts.values = {} 
        for key, val in self.values.items():
            f = interp1d(self.time, val, axis=0, fill_value="extrapolate")
            new_ts.values[key] = f(new_time)
        return new_ts
    
    def interpolate_mask(self, target_ts: Timeseries, key: str = "mask") -> np.ndarray:
        f = interp1d(self.time, self.values["mask"], axis=0, kind="nearest", bounds_error=False, fill_value="extrapolate")
        target_ts.append(key, f(target_ts.time).astype(bool))
    
    #joint indices
    @property
    def upper_limbs(self):
        return np.array([SMPLX_JOINT_NAMES.index(name) for name in UPPER_LIMBS_JOINT_NAMES])
    @property
    def lower_limbs(self):
        return np.array([SMPLX_JOINT_NAMES.index(name) for name in LOWER_LIMBS_JOINT_NAMES])
    @property
    def trunk(self):
        return np.array([SMPLX_JOINT_NAMES.index(name) for name in TRUNK_JOINT_NAMES])
    @property
    def head(self):
        return np.array([SMPLX_JOINT_NAMES.index(name) for name in HEAD_JOINT_NAMES])

    #eval joint indices
    @property
    def eval_upper_limbs(self):
        return np.array([EVAL_JOINT_NAMES.index(name) for name in UPPER_LIMBS_JOINT_NAMES])
    @property
    def eval_lower_limbs(self):
        return np.array([EVAL_JOINT_NAMES.index(name) for name in LOWER_LIMBS_JOINT_NAMES])
    @property
    def eval_trunk(self):
        return np.array([EVAL_JOINT_NAMES.index(name) for name in TRUNK_JOINT_NAMES])
    @property
    def eval_head(self):
        return np.array([EVAL_JOINT_NAMES.index(name) for name in HEAD_JOINT_NAMES])
    @property
    def eval_all_joints(self):
        return np.array([EVAL_JOINT_NAMES.index(name) for name in EVAL_JOINT_NAMES])
    
    #joint timeseries
    @property
    def upper_limbs_ts(self):
        """ Return timeseries of upper limbs joints only. """
        return Timeseries(self.time, "joints", self.values["joints"][:, self.eval_upper_limbs, :])
    @property
    def lower_limbs_ts(self):
        return Timeseries(self.time, "joints", self.values["joints"][:, self.eval_lower_limbs, :])
    @property
    def trunk_ts(self):
        return Timeseries(self.time, "joints", self.values["joints"][:, self.eval_trunk, :])
    @property
    def head_ts(self):
        return Timeseries(self.time, "joints", self.values["joints"][:, self.eval_head, :])
    
    



class EgoHTGEvaluator(ABC):
    """ A base class for evaluating ego-human trajectory generation. 
    This class is designed to be extended with specific evaluation metrics and plotting functions."""


    metrics: dict = {}
    josh: list[Timeseries] = []
    egoallo: Timeseries = None
    egomc_opt: Timeseries = None
    
    def __init__(self, pip_cfg: PipelineConfig):
        self.pip_cfg = pip_cfg
        self.reset()

    def reset(self) -> None:
        """Reset all per-sequence evaluation state.

        This makes it safe to re-use the same evaluator instance across multiple
        sequences in a single Python process.
        """
        # Loaded data handles
        self.gen_data = None
        self.opt_data = None
        self.josh_data = []
        self.egoallo_load = None

        # Timeseries
        self.egomc = None
        self.exomc = None
        self.egomc_opt = None
        self.josh = []
        self.josh_w100 = None
        self.egoallo = None

        # Interval/mask bookkeeping
        self.intervals_dict = {}

        # Outputs/accumulators
        self.metrics = {}
        self.plots = {}
    
    def run(self, ctx: SequenceContext, seq_cfg: dict):
        # When running multiple sequences in one process, make sure no state leaks.
        self.reset()
        self._load_data(ctx, seq_cfg)
        
        for type, intervals in self.intervals_dict.items():
            print(f"[INFO]: Evaluating movement type: {type} in intervals {intervals}")
            self._validate_intervals(intervals)
            self._set_masks(ctx, intervals)
            self._compute_metrics(ctx, type)
            self._plot_metrics(ctx, type, intervals)
            self._save_plots(ctx.plots_dir, type)
        
        self._save_metrics(ctx.metrics_file)

    
    def align_timeseries(func):
        """Decorator to ensure two timeseries are aligned before calculation."""
        def wrapper(self, pred: Timeseries, gt: Timeseries, *args, **kwargs):
            if not np.array_equal(pred.time, gt.time):
                gt = gt.interpolate(pred.time)
            return func(self, pred, gt, *args, **kwargs)
        return wrapper
        

    def _load_data(self, ctx: SequenceContext, seq_cfg: dict):
        print("[INFO]: Loading data for visualization")
        eval_joint_idx = [SMPLX_JOINT_NAMES.index(name) for name in EVAL_JOINT_NAMES]
        
        self.gen_data = np.load(ctx.gen_seq, allow_pickle=True)
        egomc_joints = self.gen_data['smplx_egomc_joints'][:, eval_joint_idx, :]
        egomc_timestamps = self.gen_data['egomc_timestamp']
        self.egomc = Timeseries(egomc_timestamps, "joints", egomc_joints)
        seq = {"full_sequence": [[self.egomc.time[0], self.egomc.time[-1]]]} 
        self.intervals_dict = {**seq_cfg["intervals"], **seq} if seq_cfg else seq 
        
        if ctx.exomc_exists:
            exomc_joints = self.gen_data['smplx_exomc_joints'][:, eval_joint_idx, :]
            exomc_timestamps = self.gen_data['exomc_timestamp']
            self.exomc = Timeseries(exomc_timestamps, "joints", exomc_joints)
        if ctx.opt_seq.exists():
            self.opt_data = np.load(ctx.opt_seq, allow_pickle=True)
            opt_joints = self.opt_data['smplx_egomc_joints'][:, eval_joint_idx, :]

            try:
                self.egomc_opt = Timeseries(egomc_timestamps, "joints", opt_joints)
            except Exception as e:
                print(f"[ERROR]: Failed to create optimized timeseries: {e}")
                self.egomc_opt = None

        if ctx.josh_data_exists and self.pip_cfg.multi_aria:
            for file in ctx.josh_data:
                print (f"[INFO]: Loading Josh data from {file}")
                try:
                    self.josh_data.append(joblib.load(file))
                    init_frame = int(file.stem.split("_")[-2]) # assuming file name format is "scene_{start_frame}_{end_frame}.pkl"
                    end_frame = int(file.stem.split("_")[-1])
        
                    #align josh data
                    aria_obs_T_origin_device = self.gen_data["aria_obs_T_origin_device"]
                    aria_obs_T_device_camera = self.gen_data["aria_obs_T_device_camera"]
                    #world alginement
                    fp_T_origin_camera = aria_obs_T_origin_device[init_frame] @ aria_obs_T_device_camera

                    josh_joints = np.array(self.josh_data[-1]["pred_joints"]).squeeze()[:, eval_joint_idx, :]
                    josh_pred_cam = np.array(self.josh_data[-1]["pred_cam"])    

                    #align
                    josh_joints_aligned_w = np.einsum('ik,tjk->tji', fp_T_origin_camera[:3, :3], josh_joints) + fp_T_origin_camera[:3, 3]

                    #extend josh data to full sequence length by padding with 0s 
                    pad_diff =  self.gen_data["egomc_timestamp"].shape[0] - josh_joints.shape[0]
                    pad_before = init_frame
                    pad_after = pad_diff - pad_before
                    josh_joints_aligned_w = np.pad(josh_joints_aligned_w, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
                    # josh_joints_aligned_w100 = np.pad(josh_joints_aligned_w100, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
                    
                    self.josh_data[-1]["mask"] = np.pad(np.ones((end_frame - init_frame + 1,)), ((pad_before, pad_after),), mode='constant').astype(bool)
                    self.josh.append(Timeseries(self.gen_data["egomc_timestamp"], "joints", josh_joints_aligned_w))
                    # self.josh_w100 = Timeseries(self.gen_data["egomc_timestamp"], "joints", josh_joints_aligned_w100)
                except Exception as e:
                    print(f"[ERROR]: Did not load {file}: {e}")
                    #remove corrupted file from josh data list
                    self.josh_data.pop()

            if len(self.josh) == 0:
                print("[WARNING]: No valid Josh data loaded.")

        if ctx.egoallo_data_exists:
            self.egoallo_load = np.load(ctx.egoallo_data, allow_pickle=True)
            egoallo_joints = self.egoallo_load["joint_positions"]
            aria_T_origin_device = self.gen_data["aria_T_origin_device"]
            aria_T_device_camera = self.gen_data["aria_T_device_camera"]

            #check for corrupted last frames (empty rotation matrix) and pad last valid frame #TODO find route case of corruption
            valid_frames = np.all(aria_T_origin_device[:, :3, :3] != 0, axis=(1, 2))
            if not np.all(valid_frames):
                last_valid_frame = np.where(valid_frames)[0][-1]
                aria_T_origin_device[last_valid_frame + 1:] = np.tile(aria_T_origin_device[last_valid_frame], (aria_T_origin_device.shape[0] - last_valid_frame - 1, 1, 1))
               
                if np.sum(~valid_frames) > 5:
                    raise ValueError(f"More than 5 corrupted frames detected in aria_T_origin_device. Last valid frame: {last_valid_frame}, total frames: {aria_T_origin_device.shape[0]}")


            #align egoallo joints to world frame
            if aria_T_origin_device.shape[0] != egoallo_joints.shape[0]:
                min_length = min(aria_T_origin_device.shape[0], egoallo_joints.shape[0])
                if ctx.exomc_exists and self.exomc.time.shape[0] < min_length:
                    min_length = self.exomc.time.shape[0]
                aria_T_origin_device = aria_T_origin_device[:min_length]
                aria_T_device_camera = aria_T_device_camera[:min_length]
                egoallo_joints = egoallo_joints[:min_length]
            T_origin_camera = aria_T_origin_device @ aria_T_device_camera
            egoallo_joints_aligned = (
                np.einsum('tik,tjk->tji', T_origin_camera[:, :3, :3], egoallo_joints)
                + T_origin_camera[:, :3, 3][:, None, :]
            )
            self.egoallo_data = {}
            self.egoallo_data["mask"] = np.ones((egoallo_joints.shape[0],), dtype=bool)
            self.egoallo = Timeseries(self.gen_data["egomc_timestamp"][:min_length], "joints", egoallo_joints_aligned[:, eval_joint_idx, :])
            

    def _validate_intervals(self, intervals: list):
        """Ensure intervals are logical."""
        seq_start = self.egomc.time[0]
        seq_end = self.egomc.time[-1]
        
        # sequence bounds
        for i, (start, end) in enumerate(intervals):
            
            if start >= end:
                raise ValueError(f"Interval {i} is invalid: ({start}, {end}).")
            elif (start < seq_start) and (end < seq_start):
                print(f"[WARNING]: Interval {i} is invalid: ({start}, {end}) removed.")
                intervals[i] = [0, 0]
            elif (start > seq_end) and (end > seq_end):
                print(f"[WARNING]: Interval {i} is invalid: ({start}, {end}) removed.")
                intervals[i] = [0, 0]
            elif (start < seq_start):
                print(f"[WARNING]: Interval {i} is invalid: ({start}, {end}) changed to ({seq_start}, {end})")
                intervals[i][0] = seq_start
            elif (end > seq_end):
                print(f"[WARNING]: Interval {i} is invalid: ({start}, {end}) changed to ({start}, {seq_end})")
                intervals[i][1] = seq_end

        #overlaps
        sorted_intervals = sorted(
            [(i, iv[0], iv[1]) for i, iv in enumerate(intervals)],
            key=lambda x: x[2],  # sort by end time
        )

        for (i_prev, s_prev, e_prev), (i_curr, s_curr, e_curr) in zip(sorted_intervals, sorted_intervals[1:]):
            if s_curr < e_prev:  # overlap (allow s_curr == e_prev as "touching")
                raise ValueError(
                    f"Overlapping intervals detected between index {i_prev} ({s_prev}, {e_prev}) "
                    f"and index {i_curr} ({s_curr}, {e_curr})."
                )

    def _compute_metrics(self, ctx: SequenceContext, type: str):
        print("[INFO]: Computing evaluation metrics")
        self.metrics[type] = {}

        if ctx.exomc_exists:
            self.metrics[type].update({ 
                "mpjpe": self._get_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"]),
                "mpjpe_upper_limbs": self._get_mpjpe(self.egomc.upper_limbs_ts, self.exomc.upper_limbs_ts, self.egomc.values["mask"]),
                "mpjpe_lower_limbs": self._get_mpjpe(self.egomc.lower_limbs_ts, self.exomc.lower_limbs_ts, self.egomc.values["mask"]),
                "mpjpe_trunk": self._get_mpjpe(self.egomc.trunk_ts, self.exomc.trunk_ts, self.egomc.values["mask"]),
                "mpjpe_head": self._get_mpjpe(self.egomc.head_ts, self.exomc.head_ts, self.egomc.values["mask"]),
                "pa-mpjpe": self._get_pa_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_all_joints),
                "pa-mpjpe_upper_limbs": self._get_pa_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_upper_limbs),
                "pa-mpjpe_lower_limbs": self._get_pa_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_lower_limbs),
                "pa-mpjpe_trunk": self._get_pa_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_trunk),
                "pa-mpjpe_head": self._get_pa_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_head),
                "w100-mpjpe": self._get_w_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_all_joints, 100),
                "w100-mpjpe_upper_limbs": self._get_w_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_upper_limbs, 100),
                "w100-mpjpe_lower_limbs": self._get_w_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_lower_limbs, 100),
                "w100-mpjpe_trunk": self._get_w_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_trunk, 100),
                "w100-mpjpe_head": self._get_w_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_head, 100),
                "w1000-mpjpe": self._get_w_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_all_joints, 1000),
                "wa100-mpjpe": self._get_wa_mpjpe(self.egomc, self.exomc, self.egomc.values["mask"], self.egomc.eval_all_joints, 100),
                "rte": self._get_root_translation_error(self.egomc, self.exomc, self.egomc.values["mask"]),
                "jitter_egomc": self._get_joint_jitter(self.egomc, self.egomc.values["mask"]),
                "jitter_exomc": self._get_joint_jitter(self.exomc, self.exomc.values["mask"]),
                "frames": int(np.sum(self.egomc.values["mask"])),
            })
            if self.egomc_opt is not None:
                self.metrics[type].update({
                    "mpjpe_opt": self._get_mpjpe(self.egomc_opt, self.exomc, self.egomc.values["mask"]),
                    "mpjpe_upper_limbs_opt": self._get_mpjpe(self.egomc_opt.upper_limbs_ts, self.exomc.upper_limbs_ts, self.egomc.values["mask"]),
                    "mpjpe_lower_limbs_opt": self._get_mpjpe(self.egomc_opt.lower_limbs_ts, self.exomc.lower_limbs_ts, self.egomc.values["mask"]),
                    "mpjpe_trunk_opt": self._get_mpjpe(self.egomc_opt.trunk_ts, self.exomc.trunk_ts, self.egomc.values["mask"]),
                    "mpjpe_head_opt": self._get_mpjpe(self.egomc_opt.head_ts, self.exomc.head_ts, self.egomc.values["mask"]),
                    "pa-mpjpe_opt": self._get_pa_mpjpe(self.egomc_opt, self.exomc, self.egomc.values["mask"], self.egomc.eval_all_joints),
                    "pa-mpjpe_upper_limbs_opt": self._get_pa_mpjpe(self.egomc_opt, self.exomc, self.egomc.values["mask"], self.egomc.eval_upper_limbs),
                    "pa-mpjpe_lower_limbs_opt": self._get_pa_mpjpe(self.egomc_opt, self.exomc, self.egomc.values["mask"], self.egomc.eval_lower_limbs),
                    "pa-mpjpe_trunk_opt": self._get_pa_mpjpe(self.egomc_opt, self.exomc, self.egomc.values["mask"], self.egomc.eval_trunk),
                    "pa-mpjpe_head_opt": self._get_pa_mpjpe(self.egomc_opt, self.exomc, self.egomc.values["mask"], self.egomc.eval_head),
                    "w100-mpjpe_opt": self._get_w_mpjpe(self.egomc_opt, self.exomc, self.egomc.values["mask"], self.egomc.eval_all_joints, 100),
                    "wa100-mpjpe_opt": self._get_wa_mpjpe(self.egomc_opt, self.exomc, self.egomc.values["mask"], self.egomc.eval_all_joints, 100),
                    "rte_opt": self._get_root_translation_error(self.egomc_opt, self.exomc, self.egomc.values["mask"]),
                    "jitter_opt": self._get_joint_jitter(self.egomc_opt, self.egomc.values["mask"]),
                    "frames_opt": int(np.sum(self.egomc.values["mask"])),
                })

        # Benchmark
        eval_data = { 
            "egomc": self.egomc_opt if self.egomc_opt is not None else self.egomc
        }
        if ctx.exomc_exists:
            eval_data["exomc"] = self.exomc
        
        for key, gt_data in eval_data.items():
            if len(self.josh) > 0:
                
                mpjpe = []
                mpjpe_upper_limbs = []
                mpjpe_lower_limbs = []
                mpjpe_trunk = []
                mpjpe_head = []
                mpjpe_pa = []
                mpjpe_pa_upper_limbs = []
                mpjpe_pa_lower_limbs = []
                mpjpe_pa_trunk = []
                mpjpe_pa_head = []
                mpjpe_w100 = []
                mpjpe_w1000 = []
                mpjpe_wa100 = []
                rte = []
                jitter = [] 
                frames = []
                
                for i in range(len(self.josh_data)):
                    mpjpe.append(self._get_mpjpe(self.josh[i], gt_data, self.josh[i].values[f"mask_{key}"]))
                    mpjpe_upper_limbs.append(self._get_mpjpe(self.josh[i].upper_limbs_ts, gt_data.upper_limbs_ts, self.josh[i].values[f"mask_{key}"]))
                    mpjpe_lower_limbs.append(self._get_mpjpe(self.josh[i].lower_limbs_ts, gt_data.lower_limbs_ts, self.josh[i].values[f"mask_{key}"]))
                    mpjpe_trunk.append(self._get_mpjpe(self.josh[i].trunk_ts, gt_data.trunk_ts, self.josh[i].values[f"mask_{key}"]))
                    mpjpe_head.append(self._get_mpjpe(self.josh[i].head_ts, gt_data.head_ts, self.josh[i].values[f"mask_{key}"]))
                    mpjpe_pa.append(self._get_pa_mpjpe(self.josh[i], gt_data, self.josh[i].values[f"mask_{key}"], gt_data.eval_all_joints))
                    mpjpe_pa_upper_limbs.append(self._get_pa_mpjpe(self.josh[i], gt_data, self.josh[i].values[f"mask_{key}"], gt_data.eval_upper_limbs))
                    mpjpe_pa_lower_limbs.append(self._get_pa_mpjpe(self.josh[i], gt_data, self.josh[i].values[f"mask_{key}"], gt_data.eval_lower_limbs))
                    mpjpe_pa_trunk.append(self._get_pa_mpjpe(self.josh[i], gt_data, self.josh[i].values[f"mask_{key}"], gt_data.eval_trunk))
                    mpjpe_pa_head.append(self._get_pa_mpjpe(self.josh[i], gt_data, self.josh[i].values[f"mask_{key}"], gt_data.eval_head))
                    mpjpe_w100.append(self._get_w_mpjpe(self.josh[i], gt_data, self.josh[i].values[f"mask_{key}"], gt_data.eval_all_joints, 100))
                    mpjpe_w1000.append(self._get_w_mpjpe(self.josh[i], gt_data, self.josh[i].values[f"mask_{key}"], gt_data.eval_all_joints, 1000))
                    mpjpe_wa100.append(self._get_wa_mpjpe(self.josh[i], gt_data, self.josh[i].values[f"mask_{key}"], gt_data.eval_all_joints, 100))
                    rte.append(self._get_root_translation_error(self.josh[i], gt_data, self.josh[i].values[f"mask_{key}"]))
                    jitter.append(self._get_joint_jitter(self.josh[i], self.josh[i].values[f"mask_{key}"]))  
                    frames.append(int(np.sum(self.josh[i].values[f"mask_{key}"])))


                self.metrics[type].update({
                    f"mpjpe_josh_{key}": np.mean(mpjpe),
                    f"mpjpe_upper_limbs_josh_{key}": np.mean(mpjpe_upper_limbs),
                    f"mpjpe_lower_limbs_josh_{key}": np.mean(mpjpe_lower_limbs),
                    f"mpjpe_trunk_josh_{key}": np.mean(mpjpe_trunk),
                    f"mpjpe_head_josh_{key}": np.mean(mpjpe_head),
                    f"pa-mpjpe_josh_{key}": np.mean(mpjpe_pa),
                    f"pa-mpjpe_upper_limbs_josh_{key}": np.mean(mpjpe_pa_upper_limbs),
                    f"pa-mpjpe_lower_limbs_josh_{key}": np.mean(mpjpe_pa_lower_limbs),
                    f"pa-mpjpe_trunk_josh_{key}": np.mean(mpjpe_pa_trunk),
                    f"pa-mpjpe_head_josh_{key}": np.mean(mpjpe_pa_head),
                    f"w100-mpjpe_josh_{key}": np.mean(mpjpe_w100),
                    f"w1000-mpjpe_josh_{key}": np.mean(mpjpe_w1000),
                    f"wa100-mpjpe_josh_{key}": np.mean(mpjpe_wa100),
                    f"rte_josh_{key}": np.mean(rte),
                    # "pa-mpjpe": self._get_pa_mpjpe(self.josh, self.egomc, self.josh.values["mask"], self.egomc.eval_all_joints),
                    f"jitter_josh_{key}": np.mean(jitter),
                    f"frames_josh_{key}": np.sum(np.array(frames)).astype(int),
                })
                
            if self.egoallo is not None:
                self.metrics[type].update({
                    f"mpjpe_egoallo_{key}": self._get_mpjpe(self.egoallo, gt_data, self.egoallo.values[f"mask_{key}"]),
                    f"mpjpe_upper_limbs_egoallo_{key}": self._get_mpjpe(self.egoallo.upper_limbs_ts, gt_data.upper_limbs_ts, self.egoallo.values[f"mask_{key}"]),
                    f"mpjpe_lower_limbs_egoallo_{key}": self._get_mpjpe(self.egoallo.lower_limbs_ts, gt_data.lower_limbs_ts, self.egoallo.values[f"mask_{key}"]),
                    f"mpjpe_trunk_egoallo_{key}": self._get_mpjpe(self.egoallo.trunk_ts, gt_data.trunk_ts, self.egoallo.values[f"mask_{key}"]),
                    f"mpjpe_head_egoallo_{key}": self._get_mpjpe(self.egoallo.head_ts, gt_data.head_ts, self.egoallo.values[f"mask_{key}"]),  
                    f"pa-mpjpe_egoallo_{key}": self._get_pa_mpjpe(self.egoallo, gt_data, self.egoallo.values[f"mask_{key}"], gt_data.eval_all_joints),
                    f"pa-mpjpe_upper_limbs_egoallo_{key}": self._get_pa_mpjpe(self.egoallo, gt_data, self.egoallo.values[f"mask_{key}"], gt_data.eval_upper_limbs),
                    f"pa-mpjpe_lower_limbs_egoallo_{key}": self._get_pa_mpjpe(self.egoallo, gt_data, self.egoallo.values[f"mask_{key}"], gt_data.eval_lower_limbs),
                    f"pa-mpjpe_trunk_egoallo_{key}": self._get_pa_mpjpe(self.egoallo, gt_data, self.egoallo.values[f"mask_{key}"], gt_data.eval_trunk),
                    f"pa-mpjpe_head_egoallo_{key}": self._get_pa_mpjpe(self.egoallo, gt_data, self.egoallo.values[f"mask_{key}"], gt_data.eval_head),
                    f"w100-mpjpe_egoallo_{key}": self._get_w_mpjpe(self.egoallo, gt_data, self.egoallo.values[f"mask_{key}"], gt_data.eval_all_joints, 100),
                    f"wa100-mpjpe_egoallo_{key}": self._get_wa_mpjpe(self.egoallo, gt_data, self.egoallo.values[f"mask_{key}"], gt_data.eval_all_joints, 100),
                    f"rte_egoallo_{key}": self._get_root_translation_error(self.egoallo, gt_data, self.egoallo.values[f"mask_{key}"]),
                    f"jitter_egoallo_{key}": self._get_joint_jitter(self.egoallo, self.egoallo.values[f"mask_{key}"]),
                    f"frames_egoallo_{key}": int(np.sum(self.egoallo.values[f"mask_{key}"])),
                })

    def _plot_metrics(self, ctx, type: str, intervals: list):
        """Plot computed metrics per frame."""
        print("[INFO]: Plotting evaluation metrics")
        self.plots = {}

        if ctx.exomc_exists:
            # Compute per-frame metrics for plotting
            egomc_mpjpe_per_frame = self._get_mpjpe_per_frame(self.egomc, self.exomc)
            egomc_mpjpe_upper_limbs_per_frame = self._get_mpjpe_per_frame(self.egomc.upper_limbs_ts, self.exomc.upper_limbs_ts)
            egomc_mpjpe_lower_limbs_per_frame = self._get_mpjpe_per_frame(self.egomc.lower_limbs_ts, self.exomc.lower_limbs_ts)
            egomc_mpjpe_trunk_per_frame = self._get_mpjpe_per_frame(self.egomc.trunk_ts, self.exomc.trunk_ts)
            egomc_mpjpe_head_per_frame = self._get_mpjpe_per_frame(self.egomc.head_ts, self.exomc.head_ts)
            
            egomc_pa_mpjpe_per_frame = self._get_pa_mpjpe_per_frame(self.egomc, self.exomc, self.egomc.eval_all_joints)
            egomc_pa_mpjpe_upper_limbs_per_frame = self._get_pa_mpjpe_per_frame(self.egomc, self.exomc, self.egomc.eval_upper_limbs)
            egomc_pa_mpjpe_lower_limbs_per_frame = self._get_pa_mpjpe_per_frame(self.egomc, self.exomc, self.egomc.eval_lower_limbs)
            egomc_pa_mpjpe_trunk_per_frame = self._get_pa_mpjpe_per_frame(self.egomc, self.exomc, self.egomc.eval_trunk)
            egomc_pa_mpjpe_head_per_frame = self._get_pa_mpjpe_per_frame(self.egomc, self.exomc, self.egomc.eval_head)

            egomc_jitter_per_frame = self._get_joint_jitter_per_frame(self.egomc.values["joints"])
            exomc_jitter_per_frame = self._get_joint_jitter_per_frame(self.exomc.values["joints"])

            # Apply mask to per-frame metrics (set to NaN where mask is False)
            egomc_mpjpe_per_frame = np.where(self.egomc.values["mask"], egomc_mpjpe_per_frame, np.nan)
            egomc_mpjpe_upper_limbs_per_frame = np.where(self.egomc.values["mask"], egomc_mpjpe_upper_limbs_per_frame, np.nan)
            egomc_mpjpe_lower_limbs_per_frame = np.where(self.egomc.values["mask"], egomc_mpjpe_lower_limbs_per_frame, np.nan)
            egomc_mpjpe_trunk_per_frame = np.where(self.egomc.values["mask"], egomc_mpjpe_trunk_per_frame, np.nan)
            egomc_mpjpe_head_per_frame = np.where(self.egomc.values["mask"], egomc_mpjpe_head_per_frame, np.nan)
            
            egomc_pa_mpjpe_upper_limbs_per_frame = np.where(self.egomc.values["mask"], egomc_pa_mpjpe_upper_limbs_per_frame, np.nan)
            egomc_pa_mpjpe_lower_limbs_per_frame = np.where(self.egomc.values["mask"], egomc_pa_mpjpe_lower_limbs_per_frame, np.nan)
            egomc_pa_mpjpe_trunk_per_frame = np.where(self.egomc.values["mask"], egomc_pa_mpjpe_trunk_per_frame, np.nan)
            egomc_pa_mpjpe_head_per_frame = np.where(self.egomc.values["mask"], egomc_pa_mpjpe_head_per_frame, np.nan)
            egomc_pa_mpjpe_per_frame = np.where(self.egomc.values["mask"], egomc_pa_mpjpe_per_frame, np.nan)
            
            egomc_jitter_per_frame = np.where(self.egomc.values["mask"], egomc_jitter_per_frame, np.nan)
            exomc_jitter_per_frame = np.where(self.exomc.values["mask"], exomc_jitter_per_frame, np.nan)

            # Create timeseries objects for plotting
            egomc_plot = Timeseries(self.egomc.time, "mpjpe_all_joints", egomc_mpjpe_per_frame)
            egomc_plot.append("mpjpe_upper_limbs", egomc_mpjpe_upper_limbs_per_frame)
            egomc_plot.append("mpjpe_lower_limbs", egomc_mpjpe_lower_limbs_per_frame)
            egomc_plot.append("mpjpe_trunk", egomc_mpjpe_trunk_per_frame)
            egomc_plot.append("mpjpe_head", egomc_mpjpe_head_per_frame)
            
            egomc_plot.append("pa-mpjpe_all_joints", egomc_pa_mpjpe_per_frame)
            egomc_plot.append("pa-mpjpe_upper_limbs", egomc_pa_mpjpe_upper_limbs_per_frame)
            egomc_plot.append("pa-mpjpe_lower_limbs", egomc_pa_mpjpe_lower_limbs_per_frame)
            egomc_plot.append("pa-mpjpe_trunk", egomc_pa_mpjpe_trunk_per_frame)
            egomc_plot.append("pa-mpjpe_head", egomc_pa_mpjpe_head_per_frame)
            
            egomc_plot.append("jitter", egomc_jitter_per_frame)
            exomc_plot = Timeseries(self.exomc.time, "jitter", exomc_jitter_per_frame)

            self.plots.update({
                "egomc": self._plot_timeseries("EgoMoCap", egomc_plot, type, mask=self.egomc.values["mask"]),
                "exomc": self._plot_timeseries("ExoMoCap", exomc_plot, type, mask=self.exomc.values["mask"]),
            })
        
        if len(self.josh) > 0:
            #print first segment
            josh_mpjpe_eval_data_per_frame = self._get_mpjpe_per_frame(self.josh[0], self.egomc)
            josh_mpjpe_upper_limbs_eval_data_per_frame = self._get_mpjpe_per_frame(self.josh[0].upper_limbs_ts, self.egomc.upper_limbs_ts)
            josh_mpjpe_lower_limbs_eval_data_per_frame = self._get_mpjpe_per_frame(self.josh[0].lower_limbs_ts, self.egomc.lower_limbs_ts)
            josh_mpjpe_trunk_eval_data_per_frame = self._get_mpjpe_per_frame(self.josh[0].trunk_ts, self.egomc.trunk_ts)
            josh_mpjpe_head_eval_data_per_frame = self._get_mpjpe_per_frame(self.josh[0].head_ts, self.egomc.head_ts)
            # josh_pa_mpjpe_eval_data_per_frame = self._get_pa_mpjpe_per_frame(self.josh[0], self.egomc, self.egomc.eval_all_joints)
            josh_jitter_eval_data_per_frame = self._get_joint_jitter_per_frame(self.josh[0].values["joints"])

            josh_mpjpe_eval_data_per_frame = np.where(self.josh[0].values["mask_egomc"], josh_mpjpe_eval_data_per_frame, np.nan)
            josh_mpjpe_upper_limbs_eval_data_per_frame = np.where(self.josh[0].values["mask_egomc"], josh_mpjpe_upper_limbs_eval_data_per_frame, np.nan)
            josh_mpjpe_lower_limbs_eval_data_per_frame = np.where(self.josh[0].values["mask_egomc"], josh_mpjpe_lower_limbs_eval_data_per_frame, np.nan)
            josh_mpjpe_trunk_eval_data_per_frame = np.where(self.josh[0].values["mask_egomc"], josh_mpjpe_trunk_eval_data_per_frame, np.nan)
            josh_mpjpe_head_eval_data_per_frame = np.where(self.josh[0].values["mask_egomc"], josh_mpjpe_head_eval_data_per_frame, np.nan)
            # josh_pa_mpjpe_eval_data_per_frame = np.where(self.josh[0].values["mask_egomc"], josh_pa_mpjpe_eval_data_per_frame, np.nan)
            josh_jitter_eval_data_per_frame = np.where(self.josh[0].values["mask_egomc"], josh_jitter_eval_data_per_frame, np.nan)

            josh_plot = Timeseries(self.josh[0].time, "mpjpe_eval_data", josh_mpjpe_eval_data_per_frame)
            josh_plot.append("mpjpe_upper_limbs_eval_data", josh_mpjpe_upper_limbs_eval_data_per_frame)
            josh_plot.append("mpjpe_lower_limbs_eval_data", josh_mpjpe_lower_limbs_eval_data_per_frame)
            josh_plot.append("mpjpe_trunk_eval_data", josh_mpjpe_trunk_eval_data_per_frame)
            josh_plot.append("mpjpe_head_eval_data", josh_mpjpe_head_eval_data_per_frame)
            # josh_plot.append("pa-mpjpe_eval_data", josh_pa_mpjpe_eval_data_per_frame)
            josh_plot.append("jitter", josh_jitter_eval_data_per_frame)

            self.plots.update({
                "josh": self._plot_timeseries("Evaluate JOSH against EgoMoCap", josh_plot, type, mask=self.josh[0].values["mask_egomc"]),
            })

        if self.egoallo is not None:
            egoallo_mpjpe_eval_data_per_frame = self._get_mpjpe_per_frame(self.egoallo, self.egomc)
            egoallo_mpjpe_upper_limbs_eval_data_per_frame = self._get_mpjpe_per_frame(self.egoallo.upper_limbs_ts, self.egomc.upper_limbs_ts)
            egoallo_mpjpe_lower_limbs_eval_data_per_frame = self._get_mpjpe_per_frame(self.egoallo.lower_limbs_ts, self.egomc.lower_limbs_ts)
            egoallo_mpjpe_trunk_eval_data_per_frame = self._get_mpjpe_per_frame(self.egoallo.trunk_ts, self.egomc.trunk_ts)
            egoallo_mpjpe_head_eval_data_per_frame = self._get_mpjpe_per_frame(self.egoallo.head_ts, self.egomc.head_ts)
            egoallo_pa_mpjpe_eval_data_per_frame = self._get_pa_mpjpe_per_frame(self.egoallo, self.egomc, self.egomc.eval_all_joints)
            egoallo_pa_mpjpe_upper_limbs_eval_data_per_frame = self._get_pa_mpjpe_per_frame(self.egoallo, self.egomc, self.egomc.eval_upper_limbs)
            egoallo_pa_mpjpe_lower_limbs_eval_data_per_frame = self._get_pa_mpjpe_per_frame(self.egoallo, self.egomc, self.egomc.eval_lower_limbs)
            egoallo_pa_mpjpe_trunk_eval_data_per_frame = self._get_pa_mpjpe_per_frame(self.egoallo, self.egomc, self.egomc.eval_trunk)
            egoallo_pa_mpjpe_head_eval_data_per_frame = self._get_pa_mpjpe_per_frame(self.egoallo, self.egomc, self.egomc.eval_head)
            egoallo_jitter_eval_data_per_frame = self._get_joint_jitter_per_frame(self.egoallo.values["joints"])

            egoallo_mpjpe_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_mpjpe_eval_data_per_frame, np.nan)
            egoallo_mpjpe_upper_limbs_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_mpjpe_upper_limbs_eval_data_per_frame, np.nan)
            egoallo_mpjpe_lower_limbs_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_mpjpe_lower_limbs_eval_data_per_frame, np.nan)
            egoallo_mpjpe_trunk_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_mpjpe_trunk_eval_data_per_frame, np.nan)
            egoallo_mpjpe_head_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_mpjpe_head_eval_data_per_frame, np.nan)
            egoallo_pa_mpjpe_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_pa_mpjpe_eval_data_per_frame, np.nan)
            egoallo_pa_mpjpe_upper_limbs_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_pa_mpjpe_upper_limbs_eval_data_per_frame, np.nan)
            egoallo_pa_mpjpe_lower_limbs_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_pa_mpjpe_lower_limbs_eval_data_per_frame, np.nan)
            egoallo_pa_mpjpe_trunk_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_pa_mpjpe_trunk_eval_data_per_frame, np.nan)
            egoallo_pa_mpjpe_head_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_pa_mpjpe_head_eval_data_per_frame, np.nan)
            egoallo_jitter_eval_data_per_frame = np.where(self.egoallo.values["mask_egomc"], egoallo_jitter_eval_data_per_frame, np.nan)
            
            egoallo_plot = Timeseries(self.egoallo.time, "mpjpe_all_joints", egoallo_mpjpe_eval_data_per_frame)
            egoallo_plot.append("mpjpe_upper_limbs", egoallo_mpjpe_upper_limbs_eval_data_per_frame)
            egoallo_plot.append("mpjpe_lower_limbs", egoallo_mpjpe_lower_limbs_eval_data_per_frame)
            egoallo_plot.append("mpjpe_trunk", egoallo_mpjpe_trunk_eval_data_per_frame)
            egoallo_plot.append("mpjpe_head", egoallo_mpjpe_head_eval_data_per_frame)
            egoallo_plot.append("pa-mpjpe_all_joints", egoallo_pa_mpjpe_eval_data_per_frame)
            egoallo_plot.append("pa-mpjpe_upper_limbs", egoallo_pa_mpjpe_upper_limbs_eval_data_per_frame)
            egoallo_plot.append("pa-mpjpe_lower_limbs", egoallo_pa_mpjpe_lower_limbs_eval_data_per_frame)
            egoallo_plot.append("pa-mpjpe_trunk", egoallo_pa_mpjpe_trunk_eval_data_per_frame)
            egoallo_plot.append("pa-mpjpe_head", egoallo_pa_mpjpe_head_eval_data_per_frame)
            egoallo_plot.append("jitter", egoallo_jitter_eval_data_per_frame)   
            
            self.plots.update({
                "egoallo": self._plot_timeseries("Evaluate EgoAllo against EgoMoCap", egoallo_plot, type, mask=self.egoallo.values["mask_egomc"]),
            })
            

    def _mask_to_spans(self, time: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
        """Convert a boolean mask into contiguous (start, end) spans in time."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or time.shape[0] != mask.shape[0]:
            raise ValueError("time and mask must be 1D arrays of the same length")

        spans = []
        in_span = False
        start_t = None

        for t, active in zip(time, mask):
            if active and not in_span:
                in_span = True
                start_t = t
            elif not active and in_span:
                in_span = False
                spans.append((start_t, t))

        if in_span and start_t is not None:
            spans.append((start_t, time[-1]))

        return spans


    def _plot_timeseries(self, name: str, ts_masked: Timeseries, type: str, mask: np.ndarray) -> plt.Figure:
        """
        Groups metrics by prefix and plots them in an (N, 1) vertical stack.
        """
        # 1. Group keys by prefix (e.g., 'mpjpe_head' and 'mpjpe_trunk' both go to 'mpjpe')
        groups = defaultdict(list)
        for key in ts_masked.values.keys():
            prefix = key.split('_')[0]
            groups[prefix].append(key)
        
        prefixes = list(groups.keys())
        n_rows = len(prefixes)
        if n_rows == 0:
            return None

        # 2. Create (N, 1) Subplots
        fig, axes = plt.subplots(
            n_rows, 1, 
            figsize=(12, 4 * n_rows), 
            sharex=True, 
            squeeze=False
        )
        fig.suptitle(f"{name} Metrics - {type.upper()}", fontsize=16, fontweight='bold') 
        

        t = ts_masked.time

        # 3. Plot each group in its own subplot
        for i, prefix in enumerate(prefixes):
            ax = axes[i, 0]
            
            # Add shaded background for intervals
            spans = self._mask_to_spans(t, mask)
            for j, (start, end) in enumerate(spans):
                label = "Eval Window" if (i == 0 and j == 0) else None
                ax.axvspan(start, end, color='tab:green', alpha=0.1, label=label)

            # Plot all series belonging to this prefix group
            for key in groups[prefix]:
                y = ts_masked.values[key]
                # Use the suffix as the label for clarity (e.g., 'upper_limbs' instead of 'mpjpe_upper_limbs')
                label = key.replace(f"{prefix}_", "") if key != prefix else key
                # add joint number to label if it's a mpjpe or pa-mpjpe metric
                if prefix in ["mpjpe", "pa-mpjpe"]:
                    joint_num = JOINT_NUMBERS.get(label, None)
                    if joint_num is not None:
                        label += f" ({joint_num})"
                ax.plot(t, y, linewidth=1.5, label=label)
                #limite the y-axis to 95th percentile of the data for better visualization
                y_99 = np.nanpercentile(y, 99)
                ax.set_ylim(0, y_99 * 1.5)

            ax.set_title(f"Metric: {prefix.upper()}", loc='left', fontweight='semibold')
            ax.grid(True, alpha=0.2, linestyle=':')
            ax.set_ylabel(prefix.upper() + (f" ({METRICS_UNIT[prefix]})" if prefix in METRICS_UNIT else ""))
            ax.legend(loc='upper right', frameon=True, fontsize='small', ncol=2)

        axes[-1, 0].set_xlabel("Time (s)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig
        

    def _save_plots(self, path, type: str):
        """Save plotted metrics to files."""
        if not hasattr(self, "plots"):
            print("[WARNING]: No plots to save.")
            return
        print(f"[INFO]: Saving evaluation plots to {path}")
        path.mkdir(parents=True, exist_ok=True)
        for key, fig_dict in self.plots.items():
            fig_path = os.path.join(path, f"{key}_{type}.png")
            fig_dict.savefig(fig_path)
            plt.close(fig_dict)


    def _save_metrics(self, path):
        """Save computed metrics to a text file."""
        if not hasattr(self, "metrics"):
            print("[WARNING]: No metrics to save.")
            return
        print(f"[INFO]: Saving evaluation metrics to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for type, metrics in self.metrics.items():  
                f.write(f"[{type}]\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")
        self._print_metrics_summary(self.metrics, name="Evaluation Metrics")


    def _print_metrics_summary(self, metrics: dict, name: str):
        print(f"\n[INFO]: Overview of {name}:")

        print("\n" + "="*30)
        for type, metric_dict in metrics.items():
            print(f"-- {type.upper()} --")
            for k, v in metric_dict.items():
                if isinstance(v, (float, np.float32, np.float64)):
                    print(f"{k:<40}: {v:.4f}")
                if isinstance(v, (int, np.int64, np.int32)):
                    print(f"{k:<40}: {v}")
        print("="*30 + "\n")
        

    @align_timeseries
    def _get_root_translation_error(self, pred_ts: Timeseries, gt_ts: Timeseries, mask: np.ndarray) -> float:
        """Compute the root translation error between predicted and ground truth joints. Error of rigid aligned trajectory 
        and normalized by total displacement. Notice: Total displacement is taken from predicted trajectory due to incomplete ground truth in some sequences.
        
        Args:
            pred_ts: Predicted time series.
            gt_ts: Ground truth time series.
            mask: Boolean mask indicating valid frames for evaluation.
        Returns:
            root_translation_error: Average percentage root translation error.
        """
        pred_root_torch = torch.from_numpy(pred_ts.values["joints"][mask, 0:1, :]).float()  # Shape: (N_frames, N_joints, 3)
        gt_root_torch = torch.from_numpy(gt_ts.values["joints"][mask, 0:1, :]).float()

        # Compute the similarity transform
        pred_joints_aligned = compute_similarity_transform_torch(pred_root_torch, gt_root_torch, fixed_scale=True)
        # Compute the root translation error
        root_translation_error = torch.norm(pred_joints_aligned[:, 0] - gt_root_torch[:, 0], dim=-1)

        # Get total displacement over *contiguous* valid runs.
        displacement = np.diff(pred_ts.values["joints"][:, 0, :], axis=0)  # (N_frames-1, 3)
        displacement = np.linalg.norm(displacement, axis=-1)  # (N_frames-1,)
        valid_steps = mask[:-1] & mask[1:]
        masked_displacement = displacement[valid_steps]
        total_displacement = float(np.sum(masked_displacement))
        if total_displacement <= 0:
            return float("nan")

        return (torch.mean(root_translation_error) / total_displacement).item() * 100  # Return percentage


    @align_timeseries
    def _get_mpjpe_per_frame(self, pred_ts: Timeseries, gt_ts: Timeseries) -> np.ndarray:
        """Compute Mean Per Joint Position Error (MPJPE) per frame between predicted and ground truth joints.
        
        Args:
            pred_joints: Predicted joint positions of shape (N_frames, N_joints, 3).
            gt_joints: Ground truth joint positions of shape (N_frames, N_joints, 3).
        Returns:
            mpjpe_per_frame: MPJPE per frame of shape (N_frames,).
        """        
        error = np.linalg.norm(pred_ts.values["joints"] - gt_ts.values["joints"], axis=-1)  # Shape: (N_frames, N_joints)
        mpjpe_per_frame = np.mean(error, axis=-1)  # Shape: (N_frames,)
        return mpjpe_per_frame
    

    
    
    def _get_mpjpe(self, pred_ts: Timeseries, gt_ts: Timeseries, mask: np.ndarray) -> float:
        """Compute Mean Per Joint Position Error (MPJPE) between predicted and ground truth joints.
        
        Args:
            pred_ts: Predicted time series.
            gt_ts: Ground truth time series.
        Returns:
            mpjpe: Mean Per Joint Position Error.
        """
        #Algn root joint
        root_joint_idx = 0
        pred_ts_aligned = pred_ts.copy()
        pred_ts_aligned.values["joints"] = pred_ts.values["joints"] - pred_ts.values["joints"][:, root_joint_idx:root_joint_idx+1, :]
        gt_ts_aligned = gt_ts.copy()
        gt_ts_aligned.values["joints"] = gt_ts.values["joints"] - gt_ts.values["joints"][:, root_joint_idx:root_joint_idx+1, :]

        mpjpe_per_frame = self._get_mpjpe_per_frame(pred_ts_aligned, gt_ts_aligned)
        mpjpe = np.mean(mpjpe_per_frame[mask])

        return mpjpe
    
    
    @align_timeseries
    def _get_pa_mpjpe_per_frame(self, pred_ts: Timeseries, gt_ts: Timeseries, joint_mask: np.ndarray) -> np.ndarray:
        """Compute Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE) per frame between predicted and ground truth joints.
        
        Args:
            pred_ts: Predicted time series.
            gt_ts: Ground truth time series.
        Returns:
            pa_mpjpe_per_frame: Procrustes Aligned Mean Per Joint Position Error per frame of shape (N_frames,).
        """        
        assert pred_ts.values["joints"].shape == gt_ts.values["joints"].shape, "Predicted and ground truth joints must have the same shape."

        pred_ts_torch = torch.from_numpy(pred_ts.values["joints"]).float()  # Shape: (N_frames, N_joints, 3)
        gt_ts_torch = torch.from_numpy(gt_ts.values["joints"]).float()
        
        pred_ts_torch_pa = batch_compute_similarity_transform_torch(pred_ts_torch, gt_ts_torch)
        error = torch.norm(gt_ts_torch[:, joint_mask, :] - pred_ts_torch_pa[:, joint_mask, :], dim=-1)  # Shape: (N_frames, masked joints)
        pa_mpjpe_per_frame = torch.mean(error, dim=-1).cpu().numpy()  # Shape: (N_frames,)
        return pa_mpjpe_per_frame
    
    
    def _get_pa_mpjpe(self, pred_ts: Timeseries, gt_ts: Timeseries, mask: np.ndarray, joint_mask: np.ndarray) -> float:
        """Compute Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE) between predicted and ground truth joints.
        
        Args:
            pred_ts: Predicted time series.
            gt_ts: Ground truth time series.
        Returns:
            pa_mpjpe: Procrustes Aligned Mean Per Joint Position Error.
        """
        pa_mpjpe_per_frame = self._get_pa_mpjpe_per_frame(pred_ts, gt_ts, joint_mask)
        pa_mpjpe = np.mean(pa_mpjpe_per_frame[mask])
        return pa_mpjpe
    

    def _get_mpjpe_per_segment(self, pred_ts: Timeseries, gt_ts: Timeseries, mask: np.ndarray, joint_mask: np.ndarray, seg_length: int, type: str) -> np.ndarray:
        """Compute Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE) per segment between predicted and ground truth joints aligned 
        to get a single optimal alignment for the entire N-frame sequence.
        
        Args:
            pred_ts: Predicted time series.
            gt_ts: Ground truth time series.
            mask: Boolean mask indicating valid frames for evaluation.
            joint_mask: Boolean mask indicating valid joints for evaluation.
            seg_length: Length of segments.
        Returns:
            pa_mpjpe_per_segment: Procrustes Aligned Mean Per Joint Position Error per segment of shape (N_segments,).
        """
        assert pred_ts.values["joints"].shape == gt_ts.values["joints"].shape, "Predicted and ground truth joints must have the same shape."
        align_frames = 2 if type == "w" else None

        N_frames, N_joints, _ = pred_ts.values["joints"].shape
        pa_mpjpe_segments = np.zeros(N_frames//seg_length + 1)

        for i in range(0, N_frames, seg_length):
            end_idx = min(i + seg_length, N_frames)
            if end_idx - i < 2:  # Skip segments that are too short for alignment
                continue

            #find start and end index of valid frames given by the mask, take the biggest segment of valid frames within the current segment
            # masked segment has to be in one piece 
            segment_mask = mask[i:end_idx]
            if not np.any(segment_mask):  # Skip segments with no valid frames
                continue
            #find the longest contiguous segment of valid frames within the current segment
            valid = np.where(segment_mask)[0]
            splits = np.where(np.diff(valid) != 1)[0] + 1
            runs = np.split(valid, splits)
            best = max(runs, key=len)
            start = i + best[0]
            end = i + best[-1] + 1
            if end - start < 2:  # Skip segments that are too short for alignment
                continue
           
            pred_joints_torch = torch.from_numpy(pred_ts.values["joints"][start:end]).float()  # Shape: (1, joints, 3)
            gt_joints_torch = torch.from_numpy(gt_ts.values["joints"][start:end]).float()

            #align current segment 
            pred_joints_pa = compute_similarity_transform_torch(pred_joints_torch, gt_joints_torch, frames=align_frames)
            error = torch.norm(gt_joints_torch[:, joint_mask, :] - pred_joints_pa[:, joint_mask, :], dim=-1)
            pa_mpjpe_segments[i//seg_length] = torch.mean(error).cpu().numpy()  # Average over masked frames in segment

        pa_mpjpe_segments = pa_mpjpe_segments[pa_mpjpe_segments > 0]
        return pa_mpjpe_segments  # Average over segments with valid frames

    @align_timeseries
    def _get_wa_mpjpe(self, pred_ts: Timeseries, gt_ts: Timeseries, mask: np.ndarray, joint_mask: np.ndarray, seg_length: int) -> float:
        """Compute Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE) between predicted and ground truth joints aligned 
        over the entire N-frame sequence.
        
        Args:
            pred_ts: Predicted time series.
            gt_ts: Ground truth time series.
            mask: Boolean mask indicating valid frames for evaluation.
            joint_mask: Boolean mask indicating valid joints for evaluation.
            seg_length: Length of segments.
        Returns:
            wa_mpjpe: Procrustes World Aligned Mean Per Joint Position Error.
        """
        assert pred_ts.values["joints"].shape == gt_ts.values["joints"].shape, "Predicted and ground truth joints must have the same shape."
        wa_mpjpe_segments = self._get_mpjpe_per_segment(pred_ts, gt_ts, mask, joint_mask, seg_length, type="wa")

        return np.mean(wa_mpjpe_segments)  # Average over segments with valid frames
    
    
    @align_timeseries
    def _get_w_mpjpe(self, pred_ts: Timeseries, gt_ts: Timeseries, mask: np.ndarray, joint_mask: np.ndarray, seg_length: int) -> float:
        """Compute World Mean Per Joint Position Error (PA-MPJPE) between predicted and ground truth joints initaly 
        aligned at every N-frame sequence.
        
        Args:
            pred_ts: Predicted time series.
            gt_ts: Ground truth time series.
            mask: Boolean mask indicating valid frames for evaluation.
            joint_mask: Boolean mask indicating valid joints for evaluation.
            seg_length: Length of segments.
        Returns:
            w_mpjpe: World Mean Per Joint Position Error.
        """
        assert pred_ts.values["joints"].shape == gt_ts.values["joints"].shape, "Predicted and ground truth joints must have the same shape."
        w_mpjpe_segments = self._get_mpjpe_per_segment(pred_ts, gt_ts, mask, joint_mask, seg_length, type="w")
    
        return np.mean(w_mpjpe_segments)  # Average over segments with valid frames
    
        
    def _get_joint_jitter_per_frame(self, joints: np.ndarray) -> np.ndarray:
        """Compute jitter per frame as the joint position 3th derivative.
        
        Args:
            joints: Joint positions of shape (N_frames, N_joints, 3).
        Returns:
            jitter_per_frame: Jitter per frame of shape (N_frames - 3,).
        """ 
        assert joints.shape[0] > 3, "At least 4 frames are required to compute the 3th derivative."


        jerk = np.diff(joints, n=3, axis=0) * 30**3 * 1e-3
        # add padding to match original number of frames
        jerk = np.pad(jerk, ((3, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        jerk_magnitude = np.linalg.norm(jerk, ord=2, axis=-1)  # Shape: (N_frames, N_joints)
        jitter_per_frame = np.mean(jerk_magnitude, axis=-1)    # Shape: (N_frames,)
        return jitter_per_frame
    

    def _get_joint_jitter(self, ts: Timeseries, mask: np.ndarray) -> float:
        """Compute average jitter over the entire sequence as the mean of the joint position 3th derivative.
        Args:
            joints: Joint positions of shape (N_frames, N_joints, 3).
        Returns:
            jitter: Average jitter over the sequence.
        """
        jitter_per_frame = self._get_joint_jitter_per_frame(ts.values["joints"])
        jitter = np.mean(jitter_per_frame[mask])
        return jitter
    

    def _set_masks(self, ctx, intervals: list = None) -> None: 
        """ Set masks for egomc and exomc timeseries based on provided intervals. Has to be reinitialized for every movement type."""
        if ctx.exomc_exists:
            exomc_mask = self.gen_data["exomc_mask"]
            self.exomc.append("mask", exomc_mask)
            self.exomc.interpolate_mask(self.egomc)
            self.exomc.values["mask"] &= self._get_intervals_mask(self.exomc.time, intervals).astype(bool)
            self.egomc.values["mask"] &= self._get_intervals_mask(self.egomc.time, intervals).astype(bool)

        if self.josh is not None:
            for ts in self.josh:
                ts.append('mask_egomc', self.josh_data[self.josh.index(ts)]["mask"])
                ts.values["mask_egomc"] &= self._get_intervals_mask(ts.time, intervals).astype(bool)

                #mask for eval on gt
                if ctx.exomc_exists:
                    self.exomc.interpolate_mask(ts, key="mask_exomc")
                    ts.values["mask_exomc"] &= self.josh_data[self.josh.index(ts)]["mask"]
                    ts.values["mask_exomc"] &= self._get_intervals_mask(ts.time, intervals).astype(bool)

        if self.egoallo is not None:
            self.egoallo.append('mask_egomc', self.egoallo_data["mask"])
            self.egoallo.values["mask_egomc"] &= self._get_intervals_mask(self.egoallo.time, intervals).astype(bool)
            if ctx.exomc_exists:
                self.exomc.interpolate_mask(self.egoallo, key="mask_exomc")
                self.egoallo.values["mask_exomc"] &= self.egoallo_data["mask"]
                self.egoallo.values["mask_exomc"] &= self._get_intervals_mask(self.egoallo.time, intervals).astype(bool)

    def _get_intervals_mask(self, time: np.ndarray, intervals: list = None) -> np.ndarray:
        """Get a boolean mask for the given time intervals.
        
        Args:
            time: Time array of shape (N_frames,).
            intervals: List of (start, end) tuples defining intervals.
        Returns:
            mask: Boolean mask of shape (N_frames,) where True indicates the frame is within the intervals.
        """
        if intervals is None or len(intervals) == 0:
            return np.ones_like(time, dtype=bool)
        
        mask = np.zeros_like(time, dtype=bool)
        for start, end in intervals:
            mask |= (time >= start) & (time <= end)
        return mask

    def _check_ts_shape(ts1, ts2):
        """Check if two timeseries have the same shape for joint data, if not crop the larger one."""
        if ts1.values["joints"].shape != ts2.values["joints"].shape:
            min_frames = min(ts1.values["joints"].shape[0], ts2.values["joints"].shape[0])
            ts1.values["joints"] = ts1.values["joints"][:min_frames]
            ts2.values["joints"] = ts2.values["joints"][:min_frames]
