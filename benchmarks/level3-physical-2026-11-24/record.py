"""Advance course physics during measured controller deadline overruns.

Engineering stress evaluation; simulator ticks continue under held or already
queued commands while the CPU computes. No controller options are changed.
"""

import argparse
import csv
import hashlib
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

import gymnasium
import numpy as np
from deadline_policy import CommandQueue, missed_periods
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation

from lsy_drone_racing.utils import load_config, load_controller


def main() -> None:
    """Record a fixed seed range with optional delay, thrust loss and deadline stress."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("start", type=int)
    parser.add_argument("count", type=int)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--link", choices=("nominal", "one", "mixed"), default="nominal")
    parser.add_argument("--thrust-scale", type=float, default=1.0)
    parser.add_argument("--thrust-end", type=float)
    parser.add_argument("--check-bounds", action="store_true")
    parser.add_argument("--wall-clock", action="store_true")
    parser.add_argument("--deadline-aware", action="store_true")
    parser.add_argument("--max-wall-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    stop_at = time.time() + args.max_wall_seconds
    folder = Path(__file__).resolve().parent
    root = Path.cwd()
    config = load_config(args.config)
    config.sim.render = False
    cls = load_controller(root / "lsy_drone_racing/control" / config.controller.file)
    env_cfg = config.env
    env = JaxToNumpy(
        gymnasium.make(
            env_cfg.id,
            freq=env_cfg.freq,
            sim_config=config.sim,
            track=env_cfg.track,
            sensor_range=env_cfg.sensor_range,
            control_mode=env_cfg.control_mode,
            disturbances=env_cfg.get("disturbances"),
            randomizations=env_cfg.get("randomizations"),
            seed=1,
        )
    )
    obs, info = env.reset(seed=1)
    namespace = ()
    try:
        with TemporaryDirectory(dir=folder) as build:
            os.chdir(build)
            control = cls(obs, info, config)
            os.chdir(root)
            with args.output.open("x", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "seed",
                        "tick",
                        "before_gates",
                        "gates",
                        "compute_ms",
                        "replan",
                        "tilt_delta",
                        "accel_delta",
                        "speed",
                        "done",
                        "solver_status",
                        "command_hash",
                        "requested_tilt",
                        "requested_thrust",
                        "gate_known",
                        "gate_along",
                        "gate_cross",
                        "height",
                        "velocity_z",
                        "actual_tilt",
                        "angular_speed",
                        "solver_ms",
                        "sqp_iterations",
                        "raw_mpc_tilt",
                        "raw_roll",
                        "raw_pitch",
                        "final_roll",
                        "final_pitch",
                        "post_angle_delta",
                        "gate_angle_delta",
                        "gate_thrust_delta",
                        "post_thrust_delta",
                        "model_tilt_max",
                        "model_tilt_first",
                        "guard_alpha",
                        "reference_scale",
                        "reference_demand",
                        "reference_clock_rate",
                        "smooth_accepted",
                        "smooth_attempts",
                        "compute_call",
                        "held_for_deadline",
                        "applied_roll",
                        "applied_pitch",
                        "applied_thrust",
                        "rotor_force_min",
                        "rotor_force_max",
                        "rotor_command_force_max",
                        "motor_saturated",
                        *namespace,
                    ]
                )
                for seed in range(args.start, args.start + args.count):
                    obs, info = env.reset(seed=seed)
                    control.begin_race()
                    previous_command = prev_accel = None
                    queue = CommandQueue(control._last_command)
                    tick = 0
                    done = False
                    transport = np.random.default_rng(seed + 581909)
                    measurement = np.random.default_rng(seed + 222607)
                    start_lap = perf_counter_ns()
                    for call in range(1500):
                        if time.time() >= stop_at:
                            print(
                                "Evaluation wall-time limit reached during race", seed, flush=True
                            )
                            return
                        if args.wall_clock:
                            from time import sleep

                            wait = (start_lap - perf_counter_ns()) / 1e9 + tick / env_cfg.freq
                            if wait > 0:
                                sleep(wait)
                        old_path = getattr(control, "_path", None)
                        before = int(obs["n_gates_passed"])
                        start = perf_counter_ns()
                        measured = obs
                        if args.noise:
                            error = Rotation.from_rotvec(
                                measurement.normal(0.0, np.radians(0.30), 3)
                            )
                            measured = dict(
                                obs,
                                quat=(Rotation.from_quat(obs["quat"]) * error).as_quat(),
                                ang_vel=obs["ang_vel"] + measurement.normal(0.0, 0.06, 3),
                            )
                        action = control.compute_control(measured, info).copy()
                        elapsed = (perf_counter_ns() - start) / 1e6
                        raw = control._acados_ocp_solver.get(0, "u")
                        predicted = control._acados_ocp_solver.get_flat("x").reshape(
                            control._N + 1, control._nx
                        )
                        predicted_tilt = np.arccos(
                            np.clip(np.cos(predicted[:, 3]) * np.cos(predicted[:, 4]), -1, 1)
                        )
                        raw_tilt = np.arccos(np.clip(np.cos(raw[0]) * np.cos(raw[1]), -1, 1))
                        gate_delta = getattr(control, "_gate_delta", np.zeros(4))
                        missed = (
                            missed_periods(elapsed, 1000.0 / env_cfg.freq)
                            if args.deadline_aware
                            else 0
                        )
                        for substep in range(missed + 1):
                            if tick >= 1500 or time.time() >= stop_at:
                                print("STOPPED before complete race", seed, flush=True)
                                return
                            tick += 1
                            delivering = substep == missed
                            before = int(obs["n_gates_passed"])
                            previous = obs
                            tilt_angle = np.arccos(
                                np.clip(np.cos(action[0]) * np.cos(action[1]), -1, 1)
                            )
                            if args.check_bounds:
                                assert np.isfinite(action).all(), (seed, tick, action)
                                assert tilt_angle <= 0.600001, (seed, tick, tilt_angle)
                                assert 0.085449 - 1e-6 <= action[3] <= 0.640001, (
                                    seed,
                                    tick,
                                    action,
                                )
                            if args.link == "mixed" and (tick - 1) % 15 == 0:
                                delayed = transport.random() < 0.60
                            elif args.link != "mixed":
                                delayed = args.link == "one"
                            if delivering:
                                queue.submit(tick, action, delayed)
                            applied = queue.advance(tick)
                            scale = args.thrust_scale
                            if args.thrust_end is not None:
                                scale += min(tick / 550, 1.0) * (args.thrust_end - scale)
                            applied[3] = max(float(env.action_space.low[3]), scale * applied[3])
                            obs, reward, terminated, truncated, info = env.step(applied)
                            end = control.step_callback(
                                applied, obs, reward, terminated, truncated, info
                            )
                            accel = (obs["vel"] - previous["vel"]) * env_cfg.freq
                            delta = (
                                np.linalg.norm(accel - prev_accel) if prev_accel is not None else 0
                            )
                            tilt = (
                                np.linalg.norm(action[:2] - previous_command[:2])
                                if previous_command is not None
                                else 0
                            )
                            gates = int(obs["n_gates_passed"])
                            done = (
                                terminated
                                or truncated
                                or end
                                or gates == len(obs["gate_sequence"])
                                or tick >= 1500
                            )
                            gate = previous["gate_sequence"][
                                min(before, len(previous["gate_sequence"]) - 1)
                            ]
                            normal = Rotation.from_quat(previous["gates_quat"][gate]).apply(
                                [1.0, 0.0, 0.0]
                            )
                            normal *= previous["gate_sequence_direction"][before]
                            offset = previous["pos"] - previous["gates_pos"][gate]
                            signed = offset @ normal
                            cross = np.linalg.norm(offset - signed * normal)
                            truth = env.unwrapped.data.sim_data
                            rpm = np.asarray(truth.states.rotor_vel).reshape(-1, 4)[0]
                            command_rpm = np.asarray(truth.controls.rotor_vel).reshape(-1, 4)[0]
                            coefficients = np.asarray(truth.params.rpm2thrust).reshape(-1, 3)
                            # Log diagnostic ground truth after stepping; never send it to control.
                            rotor_force = (
                                coefficients[:, 0]
                                + coefficients[:, 1] * rpm
                                + coefficients[:, 2] * rpm**2
                            )
                            command_force = (
                                coefficients[:, 0]
                                + coefficients[:, 1] * command_rpm
                                + coefficients[:, 2] * command_rpm**2
                            )
                            writer.writerow(
                                [
                                    seed,
                                    tick,
                                    before,
                                    gates,
                                    (elapsed if substep == 0 else 0.0),
                                    int(old_path is not control._path),
                                    tilt,
                                    delta,
                                    np.linalg.norm(obs["vel"]),
                                    int(done),
                                    control.solver_status,
                                    hashlib.blake2b(action.tobytes(), digest_size=8).hexdigest(),
                                    tilt_angle,
                                    action[3],
                                    int(previous["gates_visited"][gate]),
                                    signed,
                                    cross,
                                    float(previous["pos"][2]),
                                    float(previous["vel"][2]),
                                    float(
                                        np.arccos(
                                            np.clip(
                                                Rotation.from_quat(obs["quat"]).as_matrix()[2, 2],
                                                -1.0,
                                                1.0,
                                            )
                                        )
                                    ),
                                    float(np.linalg.norm(obs["ang_vel"])),
                                    1000 * float(control._acados_ocp_solver.get_stats("time_tot")),
                                    int(control._acados_ocp_solver.get_stats("sqp_iter")),
                                    raw_tilt,
                                    float(raw[0]),
                                    float(raw[1]),
                                    float(action[0]),
                                    float(action[1]),
                                    float(np.linalg.norm(action[:2] - raw[:2])),
                                    float(np.linalg.norm(gate_delta[:2])),
                                    float(gate_delta[3]),
                                    float(action[3] - raw[3]),
                                    float(predicted_tilt.max()),
                                    float(predicted_tilt[1]),
                                    float(getattr(control, "_guard_alpha", 1.0)),
                                    float(getattr(control, "_reference_scale", 1.0)),
                                    float(getattr(control, "_reference_demand", 0.0)),
                                    float(getattr(control, "_reference_clock_rate", 1.0)),
                                    int(getattr(control, "_smooth_counts", {}).get("accepted", 0)),
                                    sum(getattr(control, "_smooth_counts", {}).values()),
                                    int(substep == 0),
                                    int(not delivering),
                                    float(applied[0]),
                                    float(applied[1]),
                                    float(applied[3]),
                                    float(np.min(rotor_force)),
                                    float(np.max(rotor_force)),
                                    float(np.max(command_force)),
                                    int(np.count_nonzero(command_force >= 0.198)),
                                    *(),
                                ]
                            )
                            previous_command, prev_accel = action, accel
                            if done:
                                break
                        if done:
                            break
                    f.flush()
                    print(seed, gates, tick / env_cfg.freq, flush=True)
                    print(
                        "SMOOTH",
                        dict(getattr(control, "_smooth_counts", {})),
                        getattr(control, "_smooth_ms", []),
                        flush=True,
                    )
                    print("ASYNC", dict(getattr(control, "_async_counts", {})), flush=True)
                    print(
                        "PHYSICAL",
                        dict(getattr(control, "_physical_counts", {})),
                        getattr(control, "_offset", None),
                        flush=True,
                    )
                    print("ADAPTIVE", dict(getattr(control, "_adaptive_counts", {})), flush=True)
    finally:
        os.chdir(root)
        env.close()


if __name__ == "__main__":
    main()
