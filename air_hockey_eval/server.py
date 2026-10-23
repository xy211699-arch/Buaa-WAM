"""Explicit opt-in server launcher; --help never imports learned models."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .cli import read_mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve an existing checkpoint for the common hockey action-chunk protocol")
    parser.add_argument("--config", required=True, help="actual XPolicyLab model deployment YAML/JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    config = read_mapping(args.config)
    name = config.get("policy_name")
    if name not in ("FastWAM", "DP", "ACT"):
        raise ValueError("deployment policy_name must be FastWAM, DP or ACT")
    if config.get("allow_dummy_policy"):
        raise ValueError("dummy policies cannot produce benchmark results")
    if name == "ACT" and config.get("temporal_agg") is not False:
        raise ValueError("ACT benchmark deployment must explicitly set temporal_agg: false")
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    root = Path(__file__).resolve().parents[1] / "XPolicyLab"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from .policies import load_xpolicy_model
    model = load_xpolicy_model(name, config)
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    server = PolicyServer(model, PolicyServerConfig(host=args.host, port=args.port))
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
