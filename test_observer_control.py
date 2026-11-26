import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import lab
import observer_plan


class ObserverControl(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.state = Path(self.folder.name)
        (self.state / "attempts").mkdir()
        self.plan = lab.read_json(lab.ROOT / "plans/observer-coverage.json")

    def attempt(self, task, outcome, age=100, duration=50):
        directory = self.state / "attempts"
        started = datetime.now(timezone.utc) - timedelta(seconds=age)
        (directory / f"{len(list(directory.iterdir()))}.json").write_text(json.dumps({
            "task_id": task, "started_utc": started.isoformat(), "outcome": outcome,
            "controller_elapsed_s": duration}))

    def test_plan_and_reruns(self):
        lab.validate_plan(self.plan)
        units = list(lab.tasks(self.plan))
        self.assertEqual(len(units), 19)
        self.assertEqual(units[:4], [("preflight", ("preflight",)), ("s51-A", (51,"A")),
                                    ("s51-B",(51,"B")), ("s51-C",(51,"C"))])
        for task in ("s51-A", "s51-A", "s51-B", "s51-B"):
            self.attempt(task,"failed")
        with self.assertRaisesRegex(ValueError,"extra"):
            observer_plan.launch_limit(self.state, "s51-B", self.plan)
        self.assertEqual(observer_plan.launch_limit(self.state,"s51-C",self.plan),90)

    def test_budget_and_uncertain_exits(self):
        self.attempt("preflight","started",age=0)
        with self.assertRaisesRegex(ValueError,"watchdog"):
            observer_plan.launch_limit(self.state,"preflight",self.plan)
        for p in (self.state / "attempts").iterdir():
            p.unlink()
        for i in range(20):
            self.attempt(str(i),"checkpoint_saved",duration=89)
        with self.assertRaisesRegex(ValueError,"allocation"):
            observer_plan.launch_limit(self.state,"s56-B",self.plan)

    def test_child_timeout(self):
        with (self.state/".lock").open("w") as lock:
            with self.assertRaises(subprocess.TimeoutExpired):
                observer_plan.bounded_run([sys.executable,"-c","import time; time.sleep(4)"],
                                          self.state,os.environ, lock.fileno(),1.3)
