from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from GPT.naive_ddp import (
    _json_safe_result,
    build_naive_ddp_check_parser,
    format_naive_ddp_check_result,
    naive_ddp_check_config_from_args,
    run_naive_ddp_check,
)


def main() -> None:
    """
    Run the toy naïve DDP correctness check from the command line.
    """

    parser = build_naive_ddp_check_parser()
    args = parser.parse_args()
    config = naive_ddp_check_config_from_args(args)
    result = run_naive_ddp_check(config, spawn_processes=not args.simulate_locally)
    if args.json:
        print(json.dumps(_json_safe_result(result), indent=2, sort_keys=True))
        return
    print(format_naive_ddp_check_result(result))


if __name__ == "__main__":
    main()
