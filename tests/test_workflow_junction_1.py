"""
Pytest integration test for Workflow Junction 1 (Phase 1).
"""

from scripts.validate_wj1 import run_workflow_junction_1_validation

def test_workflow_junction_1():
    assert run_workflow_junction_1_validation() is True
