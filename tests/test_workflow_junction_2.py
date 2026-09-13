"""
Pytest integration test for Workflow Junction 2 (Phase 2).
"""

import tempfile
from pathlib import Path
from scripts.validate_wj2 import run_workflow_junction_2_validation

def test_workflow_junction_2():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert run_workflow_junction_2_validation(data_dir=Path(tmpdir)) is True
