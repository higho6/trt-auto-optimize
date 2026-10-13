import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import onnx
from onnx import TensorProto, helper

from optimize_agent import ROOT, Workflow, agent_loop, make_skill_registry
from skill_registry import SkillRegistry


spec = importlib.util.spec_from_file_location(
    "fusion_analysis", ROOT / ".github/skills/transformer-layer-fusion/scripts/analyze_graph.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load Transformer fusion analysis helper")
fusion_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion_analysis)


def graph(nodes, outputs, weights=()) -> onnx.ModelProto:
    return helper.make_model(helper.make_graph(
        nodes, "test", [], [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4])
            for name in outputs
        ], initializer=weights,
    ), opset_imports=[helper.make_opsetid("", 17)])


class RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_skill(self, name: str, content: str | None = None) -> Path:
        directory = self.root / name
        directory.mkdir()
        path = directory / "SKILL.md"
        path.write_text(content or f'---\nname: {name}\ndescription: "Use for tests"\n---\n# Test workflow\n')
        return directory

    def test_full_recursive_discovery_and_future_skills(self) -> None:
        self.write_skill("first")
        self.write_skill("second")
        registry = SkillRegistry([(self.root, "test")])
        self.assertEqual(set(registry.skills), {"first", "second"})
        self.assertIn("Test workflow", registry.get("first").instructions)
        self.assertTrue(registry.catalog()[0]["sha256"])

    def test_invalid_metadata_is_explicit(self) -> None:
        self.write_skill("invalid", '---\nname: wrong\ndescription: text\n---\nbody\n')
        with self.assertRaises(ValueError):
            SkillRegistry([(self.root, "test")])

    def test_empty_roots_and_duplicates_fail(self) -> None:
        with self.assertRaises(ValueError):
            SkillRegistry([(self.root, "test")])
        self.write_skill("first")
        with self.assertRaises(ValueError):
            SkillRegistry([(self.root, "one"), (self.root, "two")])

    def test_resources_cannot_escape_bundle(self) -> None:
        self.write_skill("first")
        registry = SkillRegistry([(self.root, "test")])
        with self.assertRaises(ValueError):
            registry.resource("first", "../../outside")
        with self.assertRaises(ValueError):
            registry.resource("first", "/etc/passwd")
        self.assertEqual(registry.list_resources("first", ".")[0]["path"], "SKILL.md")

    def test_every_real_kda_skill_is_registered(self) -> None:
        registry = make_skill_registry()
        actual = {
            path.parent.name for path in (ROOT / "kda/skills").rglob("SKILL.md")
        }
        self.assertEqual(actual, {
            skill.name for skill in registry.skills.values() if skill.source == "kda"
        })
        self.assertIn("transformer-layer-fusion", registry.skills)
        self.assertIn("tensorrt-plugin-authoring", registry.skills)


class FusionTests(unittest.TestCase):
    def analyze(self, model):
        return fusion_analysis.analyze(model, [], [])["candidates"]

    def test_gemm_bias_through_reshape(self) -> None:
        model = graph([
            helper.make_node("Gemm", ["x", "w"], ["linear"], name="linear"),
            helper.make_node("Reshape", ["linear", "shape"], ["reshaped"], name="reshape"),
            helper.make_node("Add", ["reshaped", "bias"], ["y"], name="bias"),
        ], ["y"], [helper.make_tensor("bias", TensorProto.FLOAT, [4], [0] * 4)])
        candidate = self.analyze(model)[0]
        self.assertEqual(candidate["kind"], "gemm_bias_epilogue")
        self.assertEqual(len(candidate["nodes"]), 3)

    def test_live_residual_is_detected_not_discarded(self) -> None:
        model = graph([
            helper.make_node("Add", ["x", "residual"], ["added"], name="residual"),
            helper.make_node("LayerNormalization", ["added", "gamma", "beta"], ["y"], name="norm", epsilon=1e-5),
            helper.make_node("Identity", ["added"], ["side"], name="side"),
        ], ["y", "side"])
        candidate = self.analyze(model)[0]
        self.assertEqual(candidate["kind"], "residual_layernorm")
        self.assertIn("Live intermediate", candidate["blockers"][0])
        self.assertTrue(any(item["name"] == "added" for item in candidate["outputs"]))
        self.assertAlmostEqual(candidate["nodes"][1]["attributes"]["epsilon"], 1e-5)

    def test_attention_semantics_remain_review_only(self) -> None:
        model = graph([
            helper.make_node("MatMul", ["q", "k"], ["scores"], name="scores"),
            helper.make_node("Mul", ["scores", "scale"], ["scaled"], name="scale"),
            helper.make_node("Softmax", ["scaled"], ["prob"], name="softmax", axis=-1),
            helper.make_node("MatMul", ["prob", "v"], ["y"], name="value"),
        ], ["y"])
        candidate = self.analyze(model)[0]
        self.assertEqual(candidate["kind"], "attention_score_softmax_value")
        self.assertEqual(candidate["status"], "review_only")
        self.assertIsNone(candidate["mapped_profile_ms"])

    def test_packed_qkv_is_not_proposed_for_repacking(self) -> None:
        model = graph([
            helper.make_node("MatMul", ["x", "weight"], ["packed"], name="packed"),
            helper.make_node("Split", ["packed"], ["q", "k", "v"], name="split", axis=-1),
        ], ["q", "k", "v"], [helper.make_tensor("weight", TensorProto.FLOAT, [4, 12], [0] * 48)])
        candidates = self.analyze(model)
        self.assertEqual(candidates[0]["kind"], "packed_qkv_epilogue_review")
        self.assertIn("already packed", candidates[0]["semantic_checks"][0])

    def test_custom_domain_ops_are_not_mistaken_for_onnx(self) -> None:
        model = graph([
            helper.make_node("Add", ["x", "bias"], ["added"], name="add"),
            helper.make_node("LayerNormalization", ["added"], ["y"], name="custom", domain="custom"),
        ], ["y"])
        self.assertEqual(self.analyze(model), [])

    def test_unknown_dimensions_are_not_replaced_by_zero(self) -> None:
        model = graph([
            helper.make_node("Add", ["x", "r"], ["added"], name="add"),
            helper.make_node("LayerNormalization", ["added", "g"], ["y"], name="ln"),
        ], ["y"])
        model.graph.output[0].type.tensor_type.shape.dim[0].Clear()
        candidate = self.analyze(model)[0]
        output = next(item for item in candidate["outputs"] if item["name"] == "y")
        self.assertIsNone(output["shape"][0])


class SkillToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        baseline = root / "baseline"
        baseline.mkdir()
        (baseline / "model.engine").write_bytes(b"engine")
        (baseline / "layers.json").write_text(json.dumps({
            "Layers": [{"Name": "layer", "LayerType": "gemm", "Inputs": [], "Outputs": []}]
        }))
        (baseline / "profile.json").write_text(json.dumps([{"name": "layer", "averageMs": 1.0}]))
        onnx_path = root / "model.onnx"
        onnx_path.write_bytes(b"stub")
        args = argparse.Namespace(
            run_dir=root / "run", baseline=baseline, onnx=onnx_path,
            shapes={"x": [1, 4]}, model="offline-test", rtol=1e-3, atol=1e-4,
            min_speedup=1.03, build_arg=[], max_candidates=2,
            allow_local_execution=False, command_timeout=2, trtexec="trtexec",
            ncu="ncu", ncu_python=root / "ncu-python", max_turns=1,
        )
        self.workflow = Workflow(args)

    def test_all_skills_enter_first_openai_request(self) -> None:
        responses = Mock()
        responses.create.return_value = SimpleNamespace(
            output=[], output_text="No candidate evaluated.", status="completed"
        )
        agent_loop(SimpleNamespace(responses=responses), self.workflow)
        instruction = responses.create.call_args.kwargs["instructions"]
        for skill in self.workflow.registry.skills.values():
            self.assertIn(skill.instructions[:80], instruction)
            self.assertIn(skill.name, self.workflow.loaded_skills)
        manifest = json.loads((self.workflow.run_dir / "skills_manifest.json").read_text())
        self.assertEqual(len(manifest), len(self.workflow.registry.skills))

    def test_skill_resource_loading_and_pagination(self) -> None:
        result = self.workflow.dispatch("read_skill_resource", {
            "skill": "transformer-layer-fusion",
            "path": "references/architecture-transfer.md", "start": 1, "lines": 50,
        })
        self.assertIn("Portable principles", result["text"])
        listing = self.workflow.dispatch("list_skill_resources", {
            "skill": "ncu-report-skill", "directory": "reference", "offset": 0, "limit": 1,
        })
        self.assertEqual(len(listing["resources"]), 1)
        self.assertGreater(listing["total"], 1)

    def test_wiki_tools_use_actual_kda_scripts(self) -> None:
        with patch.object(self.workflow, "run", return_value={"exit_code": 0}) as run:
            self.workflow.dispatch("query_kernel_wiki", {"query": "epilogue fusion"})
            self.assertIn("KernelWiki/scripts/query.py", run.call_args.args[0][1])
            self.workflow.dispatch("get_kernel_wiki_page", {"lookup": "technique-kernel-fusion"})
            self.assertIn("--follow-sources", run.call_args.args[0])
        with self.assertRaises(ValueError):
            self.workflow.dispatch("get_kernel_wiki_page", {"lookup": "../../outside.md"})

    def test_unknown_profile_report_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.workflow.dispatch("write_profile_report", {
                "profile_directory": str(self.workflow.run_dir), "content": "report",
            })

    def plugin_candidate(self) -> Path:
        self.workflow.baseline_validated = True
        directory = Path(self.workflow.dispatch("new_candidate", {
            "name": "plugin_v1", "hypothesis": "Plugin integration test",
        })["directory"])
        (directory / "plan.txt").write_text("Test integration, not speedup.")
        (directory / "kernel.so").write_bytes(b"test library")
        return directory

    def test_plugin_registration_and_hash_guard(self) -> None:
        directory = self.plugin_candidate()
        arguments = {
            "candidate": "plugin_v1", "library": "kernel.so",
            "name": "TestKernel", "version": "1", "namespace": "",
        }
        with patch.object(self.workflow, "run", return_value={"exit_code": 0}) as run:
            self.workflow.dispatch("register_plugin", arguments)
            self.assertIn("check_plugin.py", run.call_args.args[0][1])
        record = self.workflow.records["plugin_v1"]
        self.assertEqual(self.workflow.plugin_paths(record), [directory / "kernel.so"])
        with self.assertRaises(ValueError):
            self.workflow.dispatch("register_plugin", arguments)
        (directory / "kernel.so").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed after registration"):
            self.workflow.plugin_paths(record)

    def test_failed_plugin_registration_is_not_success(self) -> None:
        self.plugin_candidate()
        with patch.object(self.workflow, "run", return_value={"exit_code": 1}):
            self.workflow.dispatch("register_plugin", {
                "candidate": "plugin_v1", "library": "kernel.so",
                "name": "Missing", "version": "1", "namespace": "",
            })
        self.assertNotIn("plugins", self.workflow.records["plugin_v1"])

    def test_registered_plugins_load_during_build_profile_and_evaluation(self) -> None:
        directory = self.plugin_candidate()
        with patch.object(self.workflow, "run", return_value={"exit_code": 0}):
            self.workflow.dispatch("register_plugin", {
                "candidate": "plugin_v1", "library": "kernel.so",
                "name": "TestKernel", "version": "1", "namespace": "",
            })
        observed = []

        def build(argv, cwd):
            observed.extend(argv)
            engine_dir = directory / "build" / "engines"
            engine_dir.mkdir(parents=True)
            (engine_dir / "model.engine").write_bytes(b"candidate")
            return {"exit_code": 0}

        with patch.object(self.workflow, "run", side_effect=build), patch(
            "optimize_agent.summarize_reports", return_value={}
        ):
            self.workflow.dispatch("build_candidate", {
                "candidate": "plugin_v1", "onnx": str(self.workflow.args.onnx),
                "extra_args": [],
            })
        self.assertIn(f"--dynamicPlugins={directory / 'kernel.so'}", observed)
        with patch.object(self.workflow, "collect_profile", return_value={}) as collect:
            self.workflow.dispatch("profile_kernel", {
                "candidate": "plugin_v1", "kernel_regex": "test", "stage": "full",
            })
        self.assertIn(f"--dynamicPlugins={directory / 'kernel.so'}", collect.call_args.args[0])
        self.assertIn(f"--loadEngine={directory / 'build/engines/model.engine'}", collect.call_args.args[0])

        def evaluate(argv, cwd):
            self.assertIn("--plugin", argv)
            self.assertIn(str(directory / "kernel.so"), argv)
            (directory / "evaluation.json").write_text(json.dumps({
                "promoted": False, "correctness_passed": True,
            }))
            return {"exit_code": 0}

        with patch.object(self.workflow, "run", side_effect=evaluate):
            self.workflow.dispatch("evaluate_candidate", {"candidate": "plugin_v1", "plugins": []})

    def test_ncu_skill_helpers_are_executed_and_artifacts_checked(self) -> None:
        def run(argv, directory, env=None):
            if "--set" in argv:
                prefix = Path(argv[argv.index("-o") + 1])
                prefix.with_suffix(".ncu-rep").write_bytes(b"report")
            elif argv[1].endswith("ncu_metrics.py"):
                output = Path(argv[argv.index("--output") + 1])
                output.write_text(json.dumps([{"kernel": "kernel", "metrics": {
                    "pmsampling:example": {"available": True, "instances": 10, "value": 1, "unit": "%"}
                }}]))
            elif argv[1].endswith("analyze_reports.py"):
                run_dir = Path(argv[argv.index("--run-dir") + 1])
                (run_dir / "analysis/metrics_key_full.json").write_text("{}")
            elif argv[1].endswith("plot_timeline.py"):
                run_dir = Path(argv[argv.index("--run-dir") + 1])
                (run_dir / "analysis/pm_timeline_plots.txt").write_text("timeline")
            log = self.workflow.run_dir / "fake.log"
            log.write_text("details")
            return {"exit_code": 0, "log": str(log), "tail": "PmSampling section"}

        with patch.object(self.workflow, "run", side_effect=run):
            result = self.workflow.collect_profile(["harness"], self.workflow.run_dir, "kernel", "full")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual([item["helper"] for item in result["helpers"]], ["analyze_reports.py", "plot_timeline.py"])
        self.assertTrue((Path(result["profile_directory"]) / "REPORT.md").is_file())
        self.assertIn("ncu-report-skill", self.workflow.loaded_skills)


if __name__ == "__main__":
    unittest.main()
