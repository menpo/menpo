import sys
from pathlib import Path

import black
import pytest
from click.testing import CliRunner

menpo_root = Path(__file__).parent.parent


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 7),
    reason="Only run on one Python version to save testing time",
)
def test_black():
    runner = CliRunner()
    result = runner.invoke(
        black.main, [str(menpo_root), "--check"]
    )
    assert result.exit_code == 0, result.output
