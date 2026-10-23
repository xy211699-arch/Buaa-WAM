"""Expose ACT's native predicted horizon without changing the XPolicyLab submodule.

The benchmark uses receding-horizon control for every method. ACT's legacy
get_action advances an internal *call* counter and returns only one row; that
counter is not the benchmark's 30 Hz clock. Force a fresh native prediction on
each request and return the denormalized full horizon to the shared scheduler.
Temporal aggregation is deliberately disabled in this common chunk protocol.
"""
from __future__ import annotations

import numpy as np


class ACTChunkModel:
    def __init__(self, model, *, unpack=None):
        self.legacy = model
        if model.model.temporal_agg:
            raise ValueError("ACT chunk evaluation requires temporal_agg: false in deployment config")
        self._unpack = unpack

    def get_eval_capabilities(self):
        return {"adapter": "act-native-chunk-v1", "action_semantics": "joint_drive_target"}

    def reset(self):
        self.legacy.reset()

    def update_obs(self, obs):
        self.legacy.update_obs(obs)

    def get_action_chunk(self):
        core = self.legacy.model
        # Reuse the existing normalization, tensor preparation and no_grad
        # inference. t=0 forces a fresh network query instead of cached rows.
        core.t = 0
        core.get_action()
        actions = core.post_process(core.all_actions[0].detach().cpu().numpy())
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[0] != core.num_queries or not np.isfinite(actions).all():
            raise ValueError("ACT did not return a finite native action horizon")
        if self._unpack is not None:
            return self._unpack(actions)
        from XPolicyLab.utils.process_data import unpack_robot_state
        return unpack_robot_state(actions, self.legacy.action_type,
                                  self.legacy.robot_action_dim_info, source_type="obs")

    def get_action(self):
        return self.get_action_chunk()
