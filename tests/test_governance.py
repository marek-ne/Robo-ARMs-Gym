"""Diff guard, GPU admit, quota, and certificate tests. No network and no GPU."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from governance.admit import admit
from governance.certificate import build_certificate, gate_passes, render_comment
from governance.guard import decide, review
from governance.quota import within_quota
from governance.redact import public_mapping
from twin.paths import repo_root
from twin.rewards import shaped_reward


class GuardTests(unittest.TestCase):
    def test_skill_yaml_is_scripted(self) -> None:
        decision = review(["skills/barrier/skill.yaml"])
        self.assertTrue(decision["ok"])
        self.assertEqual(decision["mode"], "scripted")
        self.assertEqual(decision["skill"], "barrier")

    def test_reward_requests_training(self) -> None:
        decision = review(["skills/place/reward.py", "skills/place/skill.yaml"])
        self.assertTrue(decision["ok"])
        self.assertEqual(decision["mode"], "train")

    def test_workflow_edit_is_refused(self) -> None:
        decision = review([".github/workflows/train.yml", "skills/reach/skill.yaml"])
        self.assertFalse(decision["ok"])
        self.assertIn("train.yml", decision["reason"])

    def test_trainer_edit_is_refused(self) -> None:
        decision = review(["twin/train.py"])
        self.assertFalse(decision["ok"])

    def test_two_skills_are_refused(self) -> None:
        decision = review(["skills/reach/skill.yaml", "skills/place/skill.yaml"])
        self.assertFalse(decision["ok"])

    def test_gym_field_selects_the_twin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "skills" / "lift"
            skill.mkdir(parents=True)
            (skill / "skill.yaml").write_text("skill: lift\ngym: barrier\n", encoding="utf-8")
            decision = decide(root, ["skills/lift/skill.yaml"])
        self.assertTrue(decision["ok"], decision)
        self.assertEqual(decision["gym"], "barrier")
        self.assertEqual(decision["skill"], "lift")


class AdmitTests(unittest.TestCase):
    def test_cloud_fallback_is_refused_and_key_is_absent(self) -> None:
        def probe(base: str, key: str, model: str) -> dict:
            return {"model": "gemini-flash", "header_model": "", "request_id": "req-1", "probe_latency_ms": 12}

        receipt = admit(key="sk-super-secret", probe=probe, gpu=lambda: {"free_mib": 20000, "processes": []})
        self.assertFalse(receipt["admitted"])
        self.assertEqual(receipt["reason"], "cloud-fallback")
        self.assertNotIn("sk-super-secret", json.dumps(receipt))

    def test_vllm_holds_the_card(self) -> None:
        def probe(base: str, key: str, model: str) -> dict:
            return {"model": "robo-arm-gpu-probe", "header_model": "", "request_id": "req-2", "probe_latency_ms": 8}

        receipt = admit(
            key="sk-super-secret",
            probe=probe,
            gpu=lambda: {"free_mib": 20000, "processes": ["VLLM::EngineCore"]},
        )
        self.assertFalse(receipt["admitted"])
        self.assertEqual(receipt["reason"], "vllm-holding-gpu")

    def test_local_probe_and_free_card_admits(self) -> None:
        def probe(base: str, key: str, model: str) -> dict:
            return {
                "model": "robo-arm-gpu-probe",
                "header_model": "robo-arm-gpu-probe",
                "request_id": "req-3",
                "probe_latency_ms": 9.5,
            }

        receipt = admit(key="sk-super-secret", probe=probe, gpu=lambda: {"free_mib": 12000, "processes": []})
        self.assertTrue(receipt["admitted"])
        self.assertEqual(receipt["worker"], "node-d")
        self.assertEqual(receipt["request_id"], "req-3")
        self.assertNotIn("sk-super-secret", json.dumps(receipt))

    def test_missing_key_does_not_call_the_probe(self) -> None:
        called = {"n": 0}

        def probe(base: str, key: str, model: str) -> dict:
            called["n"] += 1
            return {}

        receipt = admit(key="", probe=probe)
        self.assertEqual(called["n"], 0)
        self.assertEqual(receipt["reason"], "gateway-key-missing")


class QuotaTests(unittest.TestCase):
    def test_one_success_this_week_blocks_the_next(self) -> None:
        now = datetime(2026, 11, 10, tzinfo=timezone.utc)
        runs = [
            {"actor": {"login": "ada"}, "conclusion": "success", "created_at": "2026-11-09T12:00:00Z"},
            {"actor": {"login": "ada"}, "conclusion": "failure", "created_at": "2026-11-10T12:00:00Z"},
        ]
        self.assertFalse(within_quota("ada", runs, now=now))
        self.assertTrue(within_quota("grace", runs, now=now))


class CertificateTests(unittest.TestCase):
    def test_gate_requires_snyk_gateway_and_score(self) -> None:
        self.assertFalse(gate_passes(success_rate=1.0, n_episodes=50, snyk={"status": "passed"}, gateway={"admitted": False}))
        self.assertFalse(gate_passes(success_rate=0.5, n_episodes=50, snyk={"status": "passed"}, gateway={"admitted": True}))
        self.assertTrue(
            gate_passes(
                success_rate=0.8,
                n_episodes=50,
                snyk={"status": "passed"},
                gateway={"admitted": True},
            )
        )

    def test_comment_hides_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            certificate = build_certificate(
                skill="barrier",
                gym="barrier",
                success_rate=1.0,
                n_episodes=50,
                mean_distance_cm=1.2,
                checkpoint=None,
                snyk={"status": "passed", "token": "snyk-secret"},
                gateway={"admitted": True, "reason": "ok", "model": "robo-arm-gpu-probe", "worker": "node-d", "api_key": "sk-secret"},
                repo=Path(tmp),
                policy="scripted",
            )
        comment = render_comment(certificate)
        self.assertNotIn("sk-secret", comment)
        self.assertNotIn("snyk-secret", comment)
        self.assertIn("pass", comment)
        self.assertEqual(public_mapping({"api_key": "x"}), {"status": "absent"})


class RewardTests(unittest.TestCase):
    def test_missing_reward_keeps_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            value = shaped_reward("place", [0.0], [0.0], {}, 1.5, root=Path(tmp))
        self.assertEqual(value, 1.5)

    def test_reward_module_replaces_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "skills" / "place"
            folder.mkdir(parents=True)
            (folder / "reward.py").write_text(
                "def reward(obs, action, info, default):\n    return default + 3\n",
                encoding="utf-8",
            )
            value = shaped_reward("place", [0.0], [0.0], {}, 1.5, root=root)
        self.assertEqual(value, 4.5)

    def test_repo_root_is_derived_from_the_file(self) -> None:
        source = (repo_root() / "twin" / "paths.py").read_text(encoding="utf-8")
        self.assertIn("Path(__file__).resolve().parents[1]", source)
        self.assertTrue((repo_root() / "skills" / "schema.json").is_file())


if __name__ == "__main__":
    unittest.main()
