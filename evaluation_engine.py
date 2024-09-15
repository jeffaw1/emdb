"""
Copyright (C) 2023  ETH Zurich, Manuel Kaufmann
"""

import collections
import functools
import os
import pickle as pkl
import json
import numpy as np
from tabulate import tabulate

from evaluation_loaders import load_hybrik, load_scoreHMR, load_niki, load_tram, load_nlf, load_nlfs, load_pliks

from configuration import SHOT_TYPE


HYBRIK = "HybrIK"
SCOREHMR = "scoreHMR"
NIKI = "Niki"
TRAM = "Tram"
NLF = 'NLF'
NLFs = 'NLFs'
PLIKS = 'pliks'

METHOD_TO_RESULT_FOLDER = {
    HYBRIK: "hybrIK-out",
    SCOREHMR: "ScoreHMR2",
    NIKI: "Niki",
    TRAM: "Tram",
    NLF: 'NLF',
    PLIKS: 'pliks',
    NLFs: 'NLF/smoothnet_windowsize32_smoothed',
}

METHOD_TO_LOAD_FUNCTION = {
    HYBRIK: load_hybrik,
    SCOREHMR: load_scoreHMR,
    NIKI: load_niki,
    TRAM: load_tram,
    NLF: load_nlf,
    NLFs: load_nlfs,
    PLIKS: load_pliks,
}


class EvaluationEngine(object):
    def __init__(self, metrics_compute_func, force_load=False, ignore_smpl_trans=True):
        # Function to be used to compute the metrics.
        self.compute_metrics = metrics_compute_func
        # If true, it will invalidate all caches and reload the baseline results.
        self.force_load = force_load
        # If set, the SMPL translation of the predictions will be set to 0. This only affects the jitter metric because
        # we always align either by the pelvis or via Procrustes for the other metrics.
        self.ignore_smpl_trans = ignore_smpl_trans

        # New attributes for accumulating vertex errors
        self.vertex_errors = {method: [] for method in METHOD_TO_RESULT_FOLDER.keys()}
        self.vertex_counts = {method: [] for method in METHOD_TO_RESULT_FOLDER.keys()}
        self.total_frames = 0

    def get_ids_from_sequence_root(self, sequence_root):
        res = sequence_root.split(os.path.sep)
        subject_id = res[-2]
        seq_id = res[-1]
        return subject_id, seq_id
    
    @functools.lru_cache()
    def _get_emdb_data(self, sequence_root):
        subject_id, seq_id = self.get_ids_from_sequence_root(sequence_root)

        data_file = os.path.join(sequence_root, f"{subject_id}_{seq_id}_{SHOT_TYPE}_data.pkl")
        with open(os.path.join(sequence_root, data_file), "rb") as f:
            data = pkl.load(f)
        return data

    def load_emdb_gt(self, sequence_root):
        """
        Load EMDB SMPL pose parameters.
        :param sequence_root: Where the .pkl file is stored.
        :return:
          poses_gt: a np array of shape (N, 72)
          betas_gt: a np array of shape (N, 10)
          trans_gt: a np array of shape (N, 3)
        """
        data = self._get_emdb_data(sequence_root)

        poses_body = data["smpl"]["poses_body"]
        poses_root = data["smpl"]["poses_root"]
        betas = data["smpl"]["betas"]
        trans = data["smpl"]["trans"]
        vis_vertices = data["visible_vertices"]
        vis_joints = data["visible_joints"]

        poses_gt = np.concatenate([poses_root, poses_body], axis=-1)
        betas_gt = np.repeat(betas.reshape((1, -1)), repeats=data["n_frames"], axis=0)
        trans_gt = trans

        return poses_gt, betas_gt, trans_gt, vis_vertices, vis_joints

    def load_good_frames_mask(self, sequence_root):
        """Return the mask that says which frames are good and whic are not (because the human is too occluded)."""
        data = self._get_emdb_data(sequence_root)
        return data["good_frames_mask"]

    def get_gender_for_baseline(self, method):
        """Which gender to use for the baseline method."""
        if method in [HYBRIK, SCOREHMR, NIKI, TRAM, NLF, PLIKS, NLFs]:
            return "neutral"
        else:
            # This will select whatever gender the ground-truth specifies.
            return None

    def compare2method(self, poses_gt, betas_gt, trans_gt, sequence_root, result_root, method, vis_vertices, vis_joints):
        """Load this method's results and compute the metrics on them."""

        # Load the baseline results
        subject_id, seq_id = self.get_ids_from_sequence_root(sequence_root)
        method_result_dir = os.path.join(result_root, subject_id, seq_id, METHOD_TO_RESULT_FOLDER[method])
        poses_cmp, betas_cmp, trans_cmp = METHOD_TO_LOAD_FUNCTION[method](method_result_dir, self.force_load)

        if self.ignore_smpl_trans:
            trans_cmp = np.zeros_like(trans_cmp)

        # Load camera parameters.
        data = self._get_emdb_data(sequence_root)
        world2cam = data["camera"]["extrinsics"]

        gender_gt = data["gender"]
        gender_hat = self.get_gender_for_baseline(method)

        metrics, metrics_extra = self.compute_metrics(
            poses_gt,
            betas_gt,
            trans_gt,
            poses_cmp,
            betas_cmp,
            trans_cmp,
            gender_gt,
            gender_hat,
            vis_vertices,
            vis_joints,
            world2cam,
        )

        # Accumulate vertex errors and visibility counts
        vertex_errors = metrics_extra.get('vertex_errors', np.array([]))
        vertex_visibility = metrics_extra.get('vertex_visibility', np.array([]))

        if vertex_errors.size > 0 and vertex_visibility.size > 0:
            self.vertex_errors[method].append(vertex_errors)
            self.vertex_counts[method].append(vertex_visibility)

        return metrics, metrics_extra, method
    def save_accumulated_data(self, result_root):
        """Save accumulated vertex error data to a JSON file."""
        accumulated_data = {}

        for method in self.vertex_errors.keys():
            if self.vertex_errors[method] and self.vertex_counts[method]:
                # Stack errors and counts across all sequences
                all_errors = np.concatenate(self.vertex_errors[method], axis=0)  # (N_vertices, total_frames, 3)
                all_counts = np.concatenate(self.vertex_counts[method], axis=0)  # (N_vertices, total_frames)

                # Calculate mean error for each vertex
                mean_errors = np.sum(all_errors, axis=0) / np.sum(all_counts, axis=0, keepdims=True)
                mean_errors = np.nan_to_num(mean_errors, nan=0.0)  # Replace NaNs with 0

                accumulated_data[method] = {
                    "mean_vertex_errors": mean_errors.tolist(),
                    "total_vertex_counts": np.sum(all_counts, axis=0).tolist(),
                    "total_frames": all_errors.shape[0]
                }

        # Save faces data (assuming it's the same for all methods)
        data = self._get_emdb_data(self.sequence_roots[0])
        #accumulated_data["faces"] = data["smpl"]["faces"].tolist()

        # Save to JSON file
        output_file = os.path.join(result_root, "accumulated_vertex_data.json")
        with open(output_file, "w") as f:
            json.dump(accumulated_data, f)

        print(f"Accumulated vertex data saved to {output_file}")

    def evaluate_single_sequence(self, sequence_root, result_root, methods):
        """Evaluate a single sequence for all methods."""
        ms, ms_extra, ms_names = [], [], []

        poses_gt, betas_gt, trans_gt, vis_vertices, vis_joints = self.load_emdb_gt(sequence_root)

        for method in methods:
            m, m_extra, ms_name = self.compare2method(poses_gt, betas_gt, trans_gt, sequence_root, result_root, method, vis_vertices, vis_joints)

            ms.append(m)
            ms_extra.append(m_extra)
            ms_names.append(ms_name)

        return ms, ms_extra, ms_names

    def to_pretty_string(self, metrics, model_names):
        """Print the metrics onto the console, but pretty."""
        if not isinstance(metrics, list):
            metrics = [metrics]
            model_names = [model_names]
        assert len(metrics) == len(model_names)
        headers, rows = [], []
        for i in range(len(metrics)):
            values = []
            for k in metrics[i]:
                if i == 0:
                    headers.append(k)
                values.append(metrics[i][k])
            rows.append([model_names[i]] + values)
        return tabulate(rows, headers=["Model"] + headers)

    def run(self, sequence_roots, result_root, methods):
        """Run the evaluation on all sequences and all methods."""
        self.sequence_roots = sequence_roots  # Store for later use

        if not isinstance(sequence_roots, list):
            sequence_roots = [sequence_roots]

        # For every baseline, accumulate the metrics of all frames so that we can later compute statistics on them.
        ms_all = None
        ms_names = None
        n_frames = 0
        n_nans = {method: 0 for method in methods}
        total_joints = 0
        total_vertices = 0
        accumulated_vertex_errors = {method: [] for method in methods}
        accumulated_vertex_visibility = {method: [] for method in methods}

        all_results = {}
        sequence_results = {}

        for sequence_root in sequence_roots:
            ms, ms_extra, ms_names = self.evaluate_single_sequence(sequence_root, result_root, methods)

            print("Metrics for sequence {}".format(sequence_root))
            print(self.to_pretty_string(ms, ms_names))

            # Save individual sequence results
            sequence_results[sequence_root] = {
                method: {**m, **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in me.items()}}
                for method, m, me in zip(ms_names, ms, ms_extra)
            }

            n_frames += ms_extra[0]["mpjpe_all"].shape[0]

            if ms_all is None:
                ms_all = [collections.defaultdict(list) for _ in ms]

            for i, method in enumerate(methods):
                for metric in ["mpjpe_all", "mpjpe_pa_all", "mpjae_all", "mpjae_pa_all", "mve_all", "mve_pa_all"]:
                    if metric in ms_extra[i]:
                        valid_data = ms_extra[i][metric][~np.isnan(ms_extra[i][metric])]
                        if len(valid_data) > 0:
                            ms_all[i][metric].append(valid_data)
                        n_nans[method] += np.sum(np.isnan(ms_extra[i][metric]))
                
                if "jitter_pd" in ms_extra[i] and not np.isnan(ms_extra[i]["jitter_pd"]):
                    ms_all[i]["jitter_pd"].append(ms_extra[i]["jitter_pd"])
                
                # Count joints and vertices if available
                if "visible_joints" in ms_extra[i]:
                    total_joints += np.sum([len(joints) for joints in ms_extra[i]["visible_joints"]])
                if "visible_vertices" in ms_extra[i]:
                    total_vertices += np.sum([len(verts) for verts in ms_extra[i]["visible_vertices"]])
                
                # Accumulate vertex errors and visibility masks
                if 'mean_vertex_errors' in ms_extra[i] and 'vertex_visibility_mask' in ms_extra[i]:
                    accumulated_vertex_errors[method].append(ms_extra[i]['mean_vertex_errors'])
                    accumulated_vertex_visibility[method].append(ms_extra[i]['vertex_visibility_mask'])
        # Compute the mean and std over all sequences.
        ms_all_agg = []
        for i, method in enumerate(methods):
            metrics = {}
            for metric in ["mpjpe_all", "mpjpe_pa_all", "mpjae_all", "mpjae_pa_all", "mve_all", "mve_pa_all"]:
                if metric in ms_all[i] and len(ms_all[i][metric]) > 0:
                    all_data = np.concatenate(ms_all[i][metric], axis=0)
                    metrics[f"{metric[:-4]} [mm]" if "jae" not in metric else f"{metric[:-4]} [deg]"] = float(np.mean(all_data))
                    metrics[f"{metric[:-4]} std"] = float(np.std(all_data))
                else:
                    metrics[f"{metric[:-4]} [mm]" if "jae" not in metric else f"{metric[:-4]} [deg]"] = np.nan
                    metrics[f"{metric[:-4]} std"] = np.nan

            if "jitter_pd" in ms_all[i] and len(ms_all[i]["jitter_pd"]) > 0:
                jitter_all = np.array(ms_all[i]["jitter_pd"])
                metrics["Jitter [10m/s^3]"] = float(np.mean(jitter_all))
                metrics["Jitter std"] = float(np.std(jitter_all))
            else:
                metrics["Jitter [10m/s^3]"] = np.nan
                metrics["Jitter std"] = np.nan
            
            metrics["NaN count"] = n_nans[method]
            ms_all_agg.append(metrics)
            all_results[method] = metrics

        print("Metrics for all sequences")
        print(self.to_pretty_string(ms_all_agg, ms_names))
        print(" ")

        print("Total Number of Frames:", n_frames)
        if n_frames > 0:
            print("Average Number of Joints per Frame:", total_joints / n_frames)
            print("Average Number of Vertices per Frame:", total_vertices / n_frames)
        for method in methods:
            print(f"Number of NaN values for {method}:", n_nans[method])
        
        # Compute mean vertex errors and visibility across all sequences
        for method in methods:
            if accumulated_vertex_errors[method]:
                all_vertex_errors = np.stack(accumulated_vertex_errors[method], axis=0)
                all_vertex_visibility = np.stack(accumulated_vertex_visibility[method], axis=0)
                
                # Compute mean error for each vertex, considering only visible instances
                with np.errstate(invalid='ignore'):  # Suppress warnings about NaN
                    mean_vertex_errors = np.nanmean(np.where(all_vertex_visibility, all_vertex_errors, np.nan), axis=0)
                
                # Handle cases where a vertex was never visible
                mean_vertex_errors = np.where(np.isnan(mean_vertex_errors), -1, mean_vertex_errors)
                
                # Compute overall visibility mask
                overall_visibility_mask = np.any(all_vertex_visibility, axis=0)

                all_results[method]['mean_vertex_errors'] = mean_vertex_errors
                all_results[method]['vertex_visibility_mask'] = overall_visibility_mask
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Convert all_results to serializable format
        all_results_serializable = {
            method: {k: convert_to_serializable(v) for k, v in metrics.items()}
            for method, metrics in all_results.items()
        }

        # Convert sequence_results to serializable format
        sequence_results_serializable = {
            seq: {
                method: {k: convert_to_serializable(v) for k, v in metrics.items()}
                for method, metrics in methods_data.items()
            }
            for seq, methods_data in sequence_results.items()
        }

        # Save all results to a JSON file
        with open(os.path.join(result_root, 'all_results.json'), 'w') as f:
            json.dump(all_results_serializable, f, indent=2)

        # Save individual sequence results to a JSON file
        with open(os.path.join(result_root, 'sequence_results.json'), 'w') as f:
            json.dump(sequence_results_serializable, f, indent=2)

        # After processing all sequences, save accumulated data
        self.save_accumulated_data(result_root)

        return all_results, sequence_results

