import sys

from . import gates
from .config import Config

CHECKS = gates.LEGACY_CHECKS


def main(argv=None) -> int:
    """Every gate across the Stage-A lattice. An arm that fails one is not a
    result."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv != ["gates"]:
        from .experiments import main as experiment_main
        return experiment_main(argv)
    arms = [
        Config(transition=transition, time_mixer=mixer)
        for transition in ("flow", "direct")
        for mixer in ("attention", "mamba")
    ]
    failures = 0
    for config in arms:
        name = f"{config.transition}-{config.time_mixer}"
        report = gates.preflight(config)
        for check, result in report["components"].items():
            if result["status"] == "pass":
                print(f"  {name:16s} {check:20s} ok")
            else:
                failures += 1
                print(f"  {name:16s} {check:20s} FAIL: {result['reason']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
