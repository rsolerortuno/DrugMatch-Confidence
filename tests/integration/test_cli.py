from typer.testing import CliRunner

from drugmatch.cli import app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "DrugMatch-Confidence" in result.stdout
