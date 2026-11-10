"""WI-0.1 skill manifest + registry tests (stdlib unittest, no extra pip)."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from platform.catalog import search
from platform.paths import SKILLS_DIR
from skills.registry import (
    SCHEMA_PATH,
    create_skill,
    list_skills,
    load,
    register_in_catalog,
    validate,
    write_certificate,
)

REACH_YAML = SKILLS_DIR / "reach" / "skill.yaml"


class SkillRegistryTests(unittest.TestCase):
    def test_yaml_loads_and_validates_against_schema(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text())
        self.assertEqual(schema.get("title"), "Robo-arm skill manifest")
        manifest = yaml.safe_load(REACH_YAML.read_text())
        self.assertIsInstance(manifest, dict)
        validate(manifest)
        self.assertEqual(manifest["skill"], "reach")
        self.assertEqual(manifest["version"], "0.1.0")
        self.assertEqual(manifest["preconditions"]["dof"], 6)
        self.assertIsNone(manifest["eval_certificate"]["eval_id"])
        self.assertIsNone(manifest["policy"]["checkpoint_uri"])

    def test_list_skills_includes_reach_and_place(self) -> None:
        skills = list_skills()
        pairs = {(s.get("skill"), s.get("version")) for s in skills}
        self.assertIn(("reach", "0.1.0"), pairs)
        self.assertIn(("place", "0.1.0"), pairs)
        self.assertIn(("barrier", "0.1.0"), pairs)
        loaded = load("place", "0.1.0")
        self.assertEqual(loaded["skill"], "place")
        self.assertEqual(loaded["task_spec"]["action"], "joint_position_delta(6)")
        self.assertIsNone(loaded["eval_certificate"]["eval_id"])
        validate(loaded)

    def test_invalid_dof_fails_validate(self) -> None:
        manifest = load("reach", "0.1.0")
        bad = copy.deepcopy(manifest)
        bad["preconditions"]["dof"] = 0
        with self.assertRaises(ValueError) as ctx:
            validate(bad)
        self.assertIn("dof", str(ctx.exception))

        bad_type = copy.deepcopy(manifest)
        bad_type["preconditions"]["dof"] = "not-an-int"
        with self.assertRaises(ValueError) as ctx:
            validate(bad_type)
        self.assertIn("dof", str(ctx.exception))

    def test_write_certificate_updates_eval_id_and_checkpoint_uri(self) -> None:
        original = REACH_YAML.read_text()
        try:
            updated = write_certificate(
                "reach",
                "0.1.0",
                eval_id="eval-wi01-test",
                checkpoint_uri="robo-arm/checkpoints/ppo/reach-test.zip",
            )
            self.assertEqual(updated["eval_certificate"]["eval_id"], "eval-wi01-test")
            self.assertEqual(
                updated["policy"]["checkpoint_uri"],
                "robo-arm/checkpoints/ppo/reach-test.zip",
            )
            on_disk = yaml.safe_load(REACH_YAML.read_text())
            self.assertEqual(on_disk["eval_certificate"]["eval_id"], "eval-wi01-test")
            self.assertEqual(
                on_disk["policy"]["checkpoint_uri"],
                "robo-arm/checkpoints/ppo/reach-test.zip",
            )
            validate(on_disk)
        finally:
            REACH_YAML.write_text(original)
        restored = yaml.safe_load(REACH_YAML.read_text())
        self.assertIsNone(restored["eval_certificate"]["eval_id"])
        self.assertIsNone(restored["policy"]["checkpoint_uri"])

    def test_register_in_catalog_records_skill_version(self) -> None:
        fd, tmp = tempfile.mkstemp(prefix="roboarm-catalog-", suffix=".json")
        os.close(fd)
        Path(tmp).unlink()
        old = os.environ.get("FIMPO_ARTIFACT_CATALOG")
        os.environ["FIMPO_ARTIFACT_CATALOG"] = tmp
        try:
            rec = register_in_catalog(load("reach", "0.1.0"))
            self.assertEqual(rec["stage"], "skills")
            self.assertEqual(rec["tenant"], "robo-arm")
            self.assertIn("reach/0.1.0", rec["uri"])
            hits = search(stage="skills", q="reach")
            self.assertTrue(any(h.get("uri") == rec["uri"] for h in hits))
        finally:
            if old is None:
                os.environ.pop("FIMPO_ARTIFACT_CATALOG", None)
            else:
                os.environ["FIMPO_ARTIFACT_CATALOG"] = old
            Path(tmp).unlink(missing_ok=True)

    def test_create_skill_copies_template_without_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            manifest = create_skill(
                "stack-cup",
                template="place",
                summary="Move a cup. Twin-only until certified.",
                skills_dir=dest,
                register=False,
            )
            self.assertEqual(manifest["skill"], "stack-cup")
            self.assertIsNone(manifest["eval_certificate"]["eval_id"])
            self.assertIsNone(manifest["policy"]["checkpoint_uri"])
            on_disk = yaml.safe_load((dest / "stack-cup" / "skill.yaml").read_text())
            self.assertEqual(on_disk["skill"], "stack-cup")
            validate(on_disk)
            with self.assertRaises(FileExistsError):
                create_skill("stack-cup", skills_dir=dest, register=False)


if __name__ == "__main__":
    unittest.main()
