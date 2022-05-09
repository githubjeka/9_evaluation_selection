from click.testing import CliRunner
import pytest

from eval_sel.train import train


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_error_for_invalid_test_split_ratio(
    runner: CliRunner
) -> None:
    result = runner.invoke(
        train,
        [
            "--logreg-c",
            -1,
        ],
    )
    assert result.exit_code == 2
    assert "logreg_c format must (0,1]" in result.output
