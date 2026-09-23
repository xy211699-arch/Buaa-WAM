#!/usr/bin/env python3
"""
Parallel seed/episode launcher for Stage-5 physics-randomized planner + optional collection.

Parallelism is at the EPISODE level:
    worker 1 -> seed A -> independent SAPIEN scene -> plan -> optional collect
    worker 2 -> seed B -> independent SAPIEN scene -> plan -> optional collect
    ...

This is intentionally safer than sharing one PhysX scene across threads.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_one(
    *,
    seed,
    script_path,
    summary_path,
    log_path,
    config_path,
    collect,
    coarse_only,
    verify,
    verify_repeats,
    verify_top_k,
):
    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--seed",
        str(seed),
        "--headless",
        "--summary-path",
        str(summary_path),
    ]

    if config_path is not None:
        cmd += [
            "--config",
            str(config_path),
        ]

    if collect:
        cmd.append("--collect")

    if coarse_only:
        cmd.append("--coarse-only")

    if verify:
        cmd += [
            "--verify",
            "--verify-repeats",
            str(verify_repeats),
            "--verify-top-k",
            str(verify_top_k),
        ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"

    # Avoid CPU oversubscription when several simulator processes run together.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    t0 = time.perf_counter()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "$ "
            + " ".join(cmd)
            + "\n\n"
        )
        fp.flush()

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=fp,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.perf_counter() - t0

    result = {
        "seed": int(seed),
        "returncode": int(proc.returncode),
        "elapsed_s": float(elapsed),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
    }

    if summary_path.exists():
        try:
            summary = json.loads(
                summary_path.read_text(
                    encoding="utf-8",
                )
            )
            best = summary.get(
                "final_best",
                {},
            ) or {}
            serve = summary.get(
                "serve",
                {},
            ) or {}

            result.update(
                {
                    "serve_y": serve.get("serve_y"),
                    "serve_speed": serve.get("speed"),
                    "serve_angle_deg": serve.get("angle_deg"),
                    "predicted_event": best.get("event"),
                    "actual_event": summary.get("actual_event"),
                    "contact_time": best.get("contact_time"),
                    "contact_x": best.get("contact_x"),
                    "contact_y": best.get("contact_y"),
                    "goal_target_y": best.get("goal_target_y"),
                    "desired_out_speed": best.get("desired_out_speed"),
                    "hit_speed": best.get("hit_speed"),
                    "search_score": best.get("score"),
                    "collection_dir": summary.get("collection_dir"),
                }
            )
        except Exception as exc:
            result["summary_parse_error"] = repr(exc)

    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="parallel independent SAPIEN processes; start with 2, then try 4",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="record selected expert trajectory for every seed",
    )
    parser.add_argument(
        "--coarse-only",
        action="store_true",
        help="skip fine search for faster throughput",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="enable repeated verification (off by default)",
    )
    parser.add_argument(
        "--verify-repeats",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--verify-top-k",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--planner-script",
        type=Path,
        default=(
            ROOT
            / "scripts"
            / "plan_direct_counterattack_stage5_v1_5.py"
        ),
    )

    args = parser.parse_args()

    if args.num_seeds < 1:
        parser.error("--num-seeds must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    script_path = args.planner_script
    if not script_path.is_absolute():
        script_path = ROOT / script_path

    if not script_path.exists():
        raise FileNotFoundError(
            f"planner script not found: {script_path}"
        )

    config_path = args.config
    if (
        config_path is not None
        and not config_path.is_absolute()
    ):
        config_path = ROOT / config_path

    run_stamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = (
        ROOT
        / "outputs"
        / "direct_planner_stage5_v1_5_parallel"
        / f"run_{run_stamp}"
    )
    logs_dir = run_dir / "logs"
    summaries_dir = run_dir / "summaries"

    logs_dir.mkdir(
        parents=True,
        exist_ok=False,
    )
    summaries_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    seeds = list(
        range(
            args.seed_start,
            args.seed_start + args.num_seeds,
        )
    )

    print("")
    print("======================================================")
    print("STAGE-5 v1.5 PARALLEL EPISODE RUNNER")
    print("======================================================")
    print(f"seeds       : {seeds[0]} .. {seeds[-1]}")
    print(f"episodes    : {len(seeds)}")
    print(f"workers     : {args.workers}")
    print(f"collect     : {args.collect}")
    print(f"coarse only : {args.coarse_only}")
    print(f"verify      : {args.verify}")
    print(f"run dir     : {run_dir}")
    print("parallelism : independent process / independent SAPIEN scene")
    print("======================================================")
    print("")
    print("Live child output is written to per-seed logs.")
    print("In another terminal you can monitor with:")
    print(f"  tail -f {logs_dir}/seed_*.log")
    print("")

    jobs = {}

    with ThreadPoolExecutor(
        max_workers=args.workers,
    ) as pool:
        for seed in seeds:
            summary_path = (
                summaries_dir
                / f"seed_{seed:06d}.json"
            )
            log_path = (
                logs_dir
                / f"seed_{seed:06d}.log"
            )

            print(
                f"[START seed={seed:06d}] "
                f"log={log_path}",
                flush=True,
            )

            fut = pool.submit(
                run_one,
                seed=seed,
                script_path=script_path,
                summary_path=summary_path,
                log_path=log_path,
                config_path=config_path,
                collect=args.collect,
                coarse_only=args.coarse_only,
                verify=args.verify,
                verify_repeats=args.verify_repeats,
                verify_top_k=args.verify_top_k,
            )
            jobs[fut] = seed

        results = []

        for fut in as_completed(jobs):
            seed = jobs[fut]

            try:
                result = fut.result()
            except Exception as exc:
                result = {
                    "seed": int(seed),
                    "returncode": -999,
                    "error": repr(exc),
                }

            results.append(result)

            print(
                f"[DONE seed={seed:06d}] "
                f"rc={result.get('returncode')} "
                f"time={result.get('elapsed_s', float('nan')):.1f}s "
                f"pred={result.get('predicted_event')} "
                f"actual={result.get('actual_event')}"
            )

    results.sort(
        key=lambda r: r["seed"]
    )

    csv_path = run_dir / "results.csv"

    fields = [
        "seed",
        "returncode",
        "elapsed_s",
        "serve_y",
        "serve_speed",
        "serve_angle_deg",
        "predicted_event",
        "actual_event",
        "contact_time",
        "contact_x",
        "contact_y",
        "goal_target_y",
        "desired_out_speed",
        "hit_speed",
        "search_score",
        "collection_dir",
        "summary_path",
        "log_path",
        "summary_parse_error",
        "error",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)

    ok = sum(
        1
        for r in results
        if r.get("returncode") == 0
    )
    goals = sum(
        1
        for r in results
        if r.get("actual_event") == "GOAL"
    )

    print("")
    print("======================================================")
    print("PARALLEL RUN FINISHED")
    print("======================================================")
    print(f"successful processes : {ok}/{len(results)}")
    print(f"actual GOAL episodes : {goals}/{len(results)}")
    print(f"results CSV          : {csv_path}")
    print(f"per-seed logs        : {logs_dir}")
    print("======================================================")


if __name__ == "__main__":
    main()
