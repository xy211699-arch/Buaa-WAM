"""generate -> prepare -> run -> report, with a model-free check command."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .schema import BenchmarkConfig
from .suite import generate_manifest, read_manifest, write_manifest


def read_mapping(path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        result = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("PyYAML is required to read YAML; JSON configs need no extra parser") from error
        result = yaml.safe_load(text)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: expected a mapping")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Dynamic air-hockey benchmark (models load only in run)")
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="create paired proposals without SAPIEN or a model")
    gen.add_argument("--config", required=True)
    gen.add_argument("--assets", default="env_cfg/air_hockey_assets.yml")
    gen.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare", help="freeze all proposals after policy-free PhysX preflight")
    prepare.add_argument("--suite", required=True)
    prepare.add_argument("--output", required=True)
    check = commands.add_parser("check", help="validate files only; never import a model or SAPIEN")
    check.add_argument("--suite", required=True)
    check.add_argument("--policy-config")
    run = commands.add_parser("run", help="explicitly run a trained policy against the frozen suite")
    run.add_argument("--suite", required=True)
    run.add_argument("--policy-config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--hardware-label", required=True)
    run.add_argument("--mode", choices=("latency", "zero_latency"), default="latency")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--no-trace", action="store_true")
    report = commands.add_parser("report", help="paired analysis and plots from complete journals")
    report.add_argument("--suite", required=True)
    report.add_argument("--runs", nargs="+", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--bootstrap-samples", type=int, default=2000)
    report.add_argument("--bootstrap-seed", type=int, default=20261023)
    report.add_argument("--no-plots", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        cfg = BenchmarkConfig.from_dict(read_mapping(args.config))
        manifest = generate_manifest(cfg, read_mapping(args.assets))
        write_manifest(args.output, manifest)
        print(json.dumps({"output": args.output, "cases": len(manifest["cases"]), "prepared": False,
                          "suite_sha256": manifest["suite_sha256"]}, ensure_ascii=False))
    elif args.command == "prepare":
        from .reference import prepare_manifest
        manifest = read_manifest(args.suite)
        prepared = prepare_manifest(manifest, progress=lambda i, n, c: print(f"preflight {i}/{n}: {c}", flush=True))
        write_manifest(args.output, prepared)
        print(json.dumps({"output": args.output, "cases": len(prepared["cases"]), "prepared": True,
                          "suite_sha256": prepared["suite_sha256"]}))
    elif args.command == "check":
        manifest = read_manifest(args.suite)
        if args.policy_config:
            from .policies import validate_policy_config
            policy = read_mapping(args.policy_config)
            validate_policy_config(policy)
            if policy.get("observation_history_frames", manifest["protocol"]["history_frames"]) != manifest["protocol"]["history_frames"]:
                raise ValueError("policy history declaration conflicts with suite")
        print(json.dumps({"valid": True, "prepared": manifest["prepared"], "cases": len(manifest["cases"]),
                          "models_loaded": False, "suite_sha256": manifest["suite_sha256"]}))
    elif args.command == "run":
        from .execution import execute_suite
        result = execute_suite(read_manifest(args.suite, require_prepared=True), read_mapping(args.policy_config),
                               args.output, hardware_label=args.hardware_label, mode=args.mode,
                               resume=args.resume, save_trace=not args.no_trace,
                               progress=lambda i, n, c, r: print(f"eval {i}/{n}: {c} success={r['success']}", flush=True))
        print(json.dumps(result))
    elif args.command == "report":
        from .report import aggregate_runs, write_report
        from .storage import read_run
        manifest = read_manifest(args.suite, require_prepared=True)
        result = aggregate_runs([read_run(path) for path in args.runs], manifest,
                                bootstrap_samples=args.bootstrap_samples, random_seed=args.bootstrap_seed)
        write_report(result, args.output, plots=not args.no_plots)
        print(json.dumps({"output": args.output, "complete": True, "warnings": result["warnings"]}))
    return 0


def entrypoint():
    try:
        return main()
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"dynamic-eval: {error}", file=sys.stderr)
        return 2
