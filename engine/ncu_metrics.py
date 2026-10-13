"""Archive metrics available on the actual GPU, without B200-only assumptions."""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-path", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_path.resolve()))
    import ncu_report

    report = ncu_report.load_report(str(args.report))
    actions = []
    for range_index in range(report.num_ranges()):
        current_range = report.range_by_idx(range_index)
        for action_index in range(current_range.num_actions()):
            action = current_range.action_by_idx(action_index)
            metrics = {}
            for name in action.metric_names():
                metric = action[name]
                available = metric.has_value()
                value = metric.value() if available else None
                metrics[name] = {
                    "value": value, "unit": metric.unit(), "available": available,
                    "instances": metric.num_instances(),
                }
            actions.append({"kernel": action.name(), "metrics": metrics})
    if not actions:
        raise RuntimeError("NCU report contains no profiled actions")
    args.output.write_text(json.dumps(actions, indent=2, default=str) + "\n")
    summary = [{
        "kernel": action["kernel"],
        "metric_count": len(action["metrics"]),
        "unavailable": [
            name for name, metric in action["metrics"].items()
            if not metric["available"]
        ],
        "overview": {
            name: metric for name, metric in action["metrics"].items()
            if name in (
                "gpu__time_duration.sum",
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
            )
        },
    } for action in actions]
    args.output.with_name("metrics_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"Parsed {len(actions)} actions; metrics saved to {args.output}")


if __name__ == "__main__":
    main()
