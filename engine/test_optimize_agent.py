import argparse
import json
import sys
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluate_engines import promotion_decision
from optimize_agent import TOOLS, Workflow, agent_loop, engine_file, summarize_reports


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        baseline = root / "baseline"
        baseline.mkdir()
        (baseline / "model.engine").write_bytes(b"baseline engine")
        (baseline / "layers.json").write_text(json.dumps({"Layers": [
            {"Name": "gemm", "LayerType": "gemm",
             "Inputs": [{"Name": "input"}], "Outputs": [{"Name": "hidden"}]},
            {"Name": "activation", "LayerType": "kgen",
             "Inputs": [{"Name": "hidden"}], "Outputs": [{"Name": "output"}]},
        ]}))
        (baseline / "profile.json").write_text(json.dumps([
            {"count": 5},
            {"name": "activation", "averageMs": 0.1, "percentage": 10},
            {"name": "gemm", "averageMs": 0.9, "percentage": 90},
        ]))
        onnx = root / "original.onnx"
        onnx.write_bytes(b"onnx stub")
        self.args = argparse.Namespace(
            run_dir=root / "run", baseline=baseline, onnx=onnx,
            shapes={"input": [1, 3]}, model="test-model", rtol=1e-3,
            atol=1e-4, min_speedup=1.03, build_arg=[], max_candidates=2,
            allow_local_execution=False, command_timeout=2, trtexec="trtexec",
            ncu="ncu", ncu_python=None, max_turns=3,
        )
        self.workflow = Workflow(self.args)

    def new_candidate(self) -> Path:
        self.workflow.baseline_validated = True
        result = self.workflow.dispatch("new_candidate", {
            "name": "v1", "hypothesis": "Fuse measured activation",
        })
        return Path(result["directory"])

    def test_profile_join_ranking_and_edges(self) -> None:
        result = summarize_reports(self.args.baseline)
        self.assertEqual(result["hotspots"][0]["name"], "gemm")
        self.assertEqual(result["edges"], [
            {"producer": "gemm", "consumer": "activation", "tensor": "hidden"}
        ])

    def test_classified_baseline_matches_legacy_reports(self) -> None:
        expected = summarize_reports(self.args.baseline)
        (self.args.baseline / "engines").mkdir()
        (self.args.baseline / "profiles").mkdir()
        for name in ("layers.json", "profile.json"):
            (self.args.baseline / name).rename(self.args.baseline / "profiles" / name)
        (self.args.baseline / "model.engine").rename(self.args.baseline / "engines/model.engine")
        self.assertEqual(summarize_reports(self.args.baseline), expected)
        self.assertEqual(engine_file(self.args.baseline), self.args.baseline / "engines/model.engine")
        self.args.run_dir = self.args.run_dir.with_name("classified-run")
        workflow = Workflow(self.args)
        self.assertEqual(workflow.summary, expected)
        with patch.object(workflow, "run", return_value={"exit_code": 1, "log": "test.log"}) as run:
            with self.assertRaisesRegex(RuntimeError, "produced no result"):
                workflow.dispatch("validate_baseline", {})
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--baseline") + 1], str(engine_file(self.args.baseline)))

    def test_profile_missing_layer_and_invalid_timing(self) -> None:
        path = self.args.baseline / "profile.json"
        for row in (
            {"name": "missing", "averageMs": 1},
            {"name": "gemm", "averageMs": -1},
            {"name": "gemm", "averageMs": float("nan")},
        ):
            path.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):
                summarize_reports(self.args.baseline)

    def test_baseline_correctness_is_prerequisite(self) -> None:
        with self.assertRaises(ValueError):
            self.workflow.dispatch("new_candidate", {"name": "v1", "hypothesis": "test"})
        with self.assertRaises(PermissionError):
            self.workflow.dispatch("validate_baseline", {})

    def test_candidate_path_and_plan_guards(self) -> None:
        directory = self.new_candidate()
        with self.assertRaises(ValueError):
            self.workflow.dispatch("write_file", {
                "candidate": "v1", "path": "../escape.py", "content": "pass",
            })
        with self.assertRaises(ValueError):
            self.workflow.dispatch("run_command", {
                "candidate": "v1", "argv": [sys.executable, "-c", "pass"],
            })
        self.workflow.dispatch("write_file", {
            "candidate": "v1", "path": "plan.txt", "content": "Evidence-based plan",
        })
        self.assertTrue((directory / "plan.txt").is_file())
        with self.assertRaises(PermissionError):
            self.workflow.dispatch("run_command", {
                "candidate": "v1", "argv": [sys.executable, "-c", "pass"],
            })
        self.workflow.records["v1"]["status"] = "built"
        with self.assertRaises(ValueError):
            self.workflow.dispatch("write_file", {
                "candidate": "v1", "path": "plan.txt", "content": "change",
            })

    def test_input_mutation_is_detected(self) -> None:
        self.args.onnx.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Original workflow input changed"):
            self.workflow.dispatch("validate_baseline", {})

    def test_reusing_run_directory_fails(self) -> None:
        with self.assertRaises(FileExistsError):
            Workflow(self.args)

    def test_command_exit_and_timeout_are_explicit(self) -> None:
        self.args.allow_local_execution = True
        result = self.workflow.run([sys.executable, "-c", "print('failure'); exit(7)"], self.workflow.run_dir)
        self.assertEqual(result["exit_code"], 7)
        self.assertIn("failure", result["tail"])
        self.args.command_timeout = 0.01
        with self.assertRaises(TimeoutError):
            self.workflow.run([sys.executable, "-c", "import time; time.sleep(10)"], self.workflow.run_dir)

    def test_tools_have_complete_parameter_contracts(self) -> None:
        self.assertEqual(len({tool["name"] for tool in TOOLS}), len(TOOLS))
        for tool in TOOLS:
            self.assertEqual(set(tool["parameters"]["required"]), set(tool["parameters"]["properties"]))
            self.assertFalse(tool["parameters"]["additionalProperties"])

    def test_openai_loop_preserves_tool_calls_and_errors(self) -> None:
        call = SimpleNamespace(
            type="function_call", name="new_candidate", call_id="call_1",
            arguments=json.dumps({"name": "v1", "hypothesis": "test"}),
            model_dump=lambda **kwargs: {
                "type": "function_call", "name": "new_candidate",
                "call_id": "call_1",
                "arguments": json.dumps({"name": "v1", "hypothesis": "test"}),
            },
        )
        responses = Mock()
        responses.create.side_effect = [
            SimpleNamespace(output=[call], output_text="", status="completed"),
            SimpleNamespace(output=[], output_text="Baseline validation required.", status="completed"),
        ]
        text = agent_loop(SimpleNamespace(responses=responses), self.workflow)
        self.assertIn("Baseline", text)
        request = responses.create.call_args.kwargs
        self.assertEqual(request["model"], "test-model")
        self.assertFalse(request["store"])
        self.assertFalse(request["parallel_tool_calls"])
        output = request["input"][-1]
        self.assertEqual(output["call_id"], "call_1")
        self.assertIn("Validate baseline", output["output"])
        self.assertTrue((self.workflow.run_dir / "final.txt").is_file())

    def test_incomplete_response_and_budget_abort(self) -> None:
        client = SimpleNamespace(responses=Mock())
        client.responses.create.return_value = SimpleNamespace(status="incomplete")
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            agent_loop(client, self.workflow)
        self.args.max_turns = 0
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            agent_loop(client, self.workflow)

    def test_api_model_progress_and_failure_events(self) -> None:
        from openai import APIConnectionError
        import httpx

        self.args.model = "gpt-6.1-sol"
        responses = Mock()
        responses.create.side_effect = APIConnectionError(
            request=httpx.Request("POST", "https://example.invalid/v1/responses")
        )
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(APIConnectionError):
            agent_loop(SimpleNamespace(responses=responses), self.workflow)
        self.assertEqual(responses.create.call_args.kwargs["model"], "gpt-6.1-sol")
        self.assertIn("waiting for Responses API", output.getvalue())
        self.assertIn("API failed: APIConnectionError", output.getvalue())
        events = [json.loads(line) for line in (self.workflow.run_dir / "events.jsonl").read_text().splitlines()]
        self.assertEqual(
            [row["kind"] for row in events if row["kind"].startswith("api_request_")],
            ["api_request_started", "api_request_failed", "api_request_finished"],
        )

    def test_promotion_comes_from_independent_evaluator(self) -> None:
        import hashlib

        directory = self.new_candidate()
        engine = directory / "build" / "engines" / "model.engine"
        engine.parent.mkdir(parents=True)
        engine.write_bytes(b"candidate engine")
        self.workflow.records["v1"].update({
            "status": "built", "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        })

        def evaluate(argv, cwd):
            (cwd / "evaluation.json").write_text(json.dumps({
                "promoted": True, "correctness_passed": True, "speedup": 1.1,
            }))
            return {"exit_code": 0, "log": "test log"}

        with patch.object(self.workflow, "run", side_effect=evaluate):
            result = self.workflow.dispatch("evaluate_candidate", {"candidate": "v1", "plugins": []})
        self.assertTrue(result["evaluation"]["promoted"])
        self.assertEqual(self.workflow.records["v1"]["status"], "promoted")
        self.assertTrue((self.workflow.run_dir / "promoted.json").is_file())


class PromotionTests(unittest.TestCase):
    def test_measured_threshold_and_paired_wins(self) -> None:
        self.assertTrue(promotion_decision([10] * 5, [9] * 5, 1.03)["promoted"])
        self.assertFalse(promotion_decision([10] * 5, [9.9] * 5, 1.03)["promoted"])
        self.assertFalse(promotion_decision([10] * 5, [8, 8, 8, 11, 11], 1.03)["promoted"])

    def test_invalid_benchmark_values_are_rejected(self) -> None:
        for samples in ([], [0], [float("nan")], [float("inf")], [-1]):
            with self.assertRaises(ValueError):
                promotion_decision(samples, samples, 1.03)
        with self.assertRaises(ValueError):
            promotion_decision([1], [1, 2], 1.03)


if __name__ == "__main__":
    unittest.main()
