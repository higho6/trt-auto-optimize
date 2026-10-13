import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


LAUNCHER = Path(__file__).with_name("run_profile.sh")


class LauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = self.root / "calls.jsonl"
        self.onnx = self.root / "input with spaces.onnx"
        self.onnx.write_bytes(b"test ONNX")
        self.python = self.root / "fake-python"
        self.python.write_text("""#!/usr/bin/python
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-u': args = args[1:]
with open(os.environ['LAUNCHER_TEST_CALLS'], 'a') as f:
    f.write(json.dumps({'args':args, 'has_key':bool(os.environ.get('OPENAI_API_KEY')), 'base_url':os.environ.get('OPENAI_BASE_URL')})+'\\n')
if args[0] == '-c':
    if 'config = json.load' in args[1]:
        print(json.dumps({'pixel_values':[int(args[3]),3,224,224]}))
    sys.exit(0)
script = Path(args[0]).name
def value(flag): return args[args.index(flag)+1]
if script == 'export_siglip.py':
    Path(value('--output')).write_bytes(b'exported')
elif script == 'evaluate_engines.py':
    if os.environ.get('LAUNCHER_TEST_FAIL_VALIDATION'): sys.exit(9)
    Path(value('--output')).write_text(json.dumps({'correctness_passed':True,'promoted':False}))
elif script == 'optimize_agent.py':
    if '--list-skills' in args: print('test skill catalog')
    else:
        Path(value('--run-dir')).mkdir()
        print('test agent finished')
""")
        self.python.chmod(0o700)
        self.trtexec = self.root / "fake-trtexec"
        self.trtexec.write_text("""#!/usr/bin/python
import sys
from pathlib import Path
for arg in sys.argv[1:]:
    if arg.split('=')[0] in ('--saveEngine','--exportProfile','--exportLayerInfo','--exportTimes'):
        Path(arg.split('=',1)[1]).write_text('artifact')
print('test trtexec finished')
""")
        self.trtexec.chmod(0o700)
        self.env = {
            **os.environ, "LAUNCHER_TEST_CALLS": str(self.calls),
        }
        self.env.pop("OPENAI_API_KEY", None)
        self.env.pop("OPENAI_BASE_URL", None)
        self.output = self.root / "output with spaces"
        self.common = [
            "bash", str(LAUNCHER), "--onnx", str(self.onnx),
            "--shapes", '{"pixel_values":[3,3,224,224]}',
            "--python", str(self.python), "--trtexec", str(self.trtexec),
            "--output-dir", str(self.output),
            "--artifacts-dir", str(self.root / "artifacts"),
        ]

    def run_launcher(self, extra, env=None):
        return subprocess.run(
            [*self.common, *extra], env=self.env if env is None else env,
            capture_output=True, text=True,
        )

    def test_prepare_pipeline_and_build_argument_propagation(self) -> None:
        result = self.run_launcher(["--prepare-only", "--build-arg", "--duration=1"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output / "baseline/engines/model.engine").is_file())
        self.assertTrue((self.root / "artifacts/engines" / self.output.name / "model.engine").is_file())
        self.assertTrue((self.root / "artifacts/profiles" / self.output.name / "profile.json").is_file())
        self.assertTrue((self.output / "baseline/engines").is_symlink())
        self.assertFalse(os.path.isabs(os.readlink(self.output / "baseline/engines")))
        self.assertTrue((self.output / "baseline_validation.json").is_file())
        self.assertTrue((self.output / "logs/agent.log").is_file())
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        agent = calls[-1]["args"]
        self.assertIn("--offline", agent)
        self.assertIn("--build-arg=--duration=1", agent)
        self.assertNotIn("--build-arg=--duration=5", agent)
        self.assertFalse(any(call["has_key"] for call in calls))

    def test_api_key_file_and_backend_are_environment_only(self) -> None:
        key = "test-only-placeholder-not-a-real-key"
        key_file = self.root / "key"
        key_file.write_text(key + "\n")
        ncu_dir = self.root / "ncu-python"
        ncu_dir.mkdir()
        (ncu_dir / "ncu_report.py").write_text("")
        result = self.run_launcher([
            "--api-key-file", str(key_file), "--base-url", "http://localhost:9999/v1",
            "--model", "test-model", "--ncu", "true", "--ncu-python", str(ncu_dir),
            "--allow-api-data", "--allow-local-execution",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertTrue(calls[-1]["has_key"])
        self.assertEqual(calls[-1]["base_url"], "http://localhost:9999/v1")
        self.assertIn("--allow-local-execution", calls[-1]["args"])
        self.assertNotIn(key, result.stdout + result.stderr + self.calls.read_text())
        for path in self.output.rglob("*.log"):
            self.assertNotIn(key, path.read_text())

    def test_missing_consent_and_invalid_parameters_stop_early(self) -> None:
        for extra in (
            [], ["--prepare-only", "--batch-size", "0"],
            ["--prepare-only", "--unknown"], ["--api-key-file", "a", "--api-key", "b"],
        ):
            result = self.run_launcher(extra)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.output.exists())

    def test_failure_is_preserved_and_agent_not_started(self) -> None:
        env = {**self.env, "LAUNCHER_TEST_FAIL_VALIDATION": "1"}
        result = self.run_launcher(["--prepare-only"], env)
        self.assertEqual(result.returncode, 9)
        self.assertIn("Evidence retained", result.stderr)
        self.assertTrue((self.output / "logs/validation.log").is_file())
        self.assertFalse((self.output / "agent").exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("existing work")
        result = self.run_launcher(["--prepare-only"])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "existing work")

    def test_build_bundle_and_existing_directory_guard(self) -> None:
        bundle = self.root / "build bundle"
        argv = [
            "bash", str(LAUNCHER.with_name("build_engine.sh")), str(self.onnx),
            "--output-dir", str(bundle), "--trtexec", str(self.trtexec),
        ]
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((bundle / "engines/model.engine").is_file())
        self.assertTrue((bundle / "profiles/profile.json").is_file())
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)

    def test_fixed_root_preserves_separate_runs_from_another_cwd(self) -> None:
        argv = self.common.copy()
        index = argv.index("--output-dir")
        del argv[index:index + 2]
        for _ in range(2):
            result = subprocess.run(
                [*argv, "--prepare-only"], cwd=self.root, env=self.env,
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        artifacts = self.root / "artifacts"
        runs = list((artifacts / "runs").iterdir())
        self.assertEqual(len(runs), 2)
        for run in runs:
            self.assertTrue((artifacts / "engines" / run.name / "model.engine").is_file())
            self.assertTrue((artifacts / "profiles" / run.name / "profile.json").is_file())
            self.assertTrue((run / "logs/agent.log").is_file())

    def test_reusing_classified_and_legacy_baselines(self) -> None:
        for classified in (False, True):
            with self.subTest(classified=classified):
                baseline = self.root / f"reuse-{classified}"
                baseline.mkdir()
                engines = baseline / "engines" if classified else baseline
                profiles = baseline / "profiles" if classified else baseline
                engines.mkdir(exist_ok=True)
                profiles.mkdir(exist_ok=True)
                (engines / "model.engine").write_text("engine")
                for name in ("layers.json", "profile.json", "times.json"):
                    (profiles / name).write_text("report")
                output = self.root / f"run-{classified}"
                result = self.run_launcher([
                    "--prepare-only", "--baseline", str(baseline), "--output-dir", str(output),
                ])
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
                evaluation = next(call["args"] for call in reversed(calls) if call["args"][0].endswith("evaluate_engines.py"))
                self.assertEqual(evaluation[evaluation.index("--baseline") + 1], str(engines / "model.engine"))

    def test_default_export_is_wired(self) -> None:
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        args = [
            "bash", str(LAUNCHER), "--prepare-only",
            "--model-path", str(checkpoint), "--batch-size", "4",
            "--python", str(self.python), "--trtexec", str(self.trtexec),
            "--output-dir", str(self.output),
            "--artifacts-dir", str(self.root / "artifacts"),
        ]
        result = subprocess.run(args, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "artifacts/onnx" / self.output.name / "model.onnx").is_file())
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        export = next(call["args"] for call in calls if call["args"][0].endswith("export_siglip.py"))
        self.assertEqual(export[export.index("--batch-size") + 1], "4")
        self.assertIn("--verify", export)


if __name__ == "__main__":
    unittest.main()
