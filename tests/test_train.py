from click.testing import CliRunner
import pytest
from faker import Faker
from eval_sel.train import train
import csv
import os


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def generate_data(id):
    fake = Faker()
    return [id,
            fake.pyint(min_value=2500, max_value=2800),
            fake.pyint(min_value=30, max_value=200),
            fake.pyint(min_value=10, max_value=30),
            fake.pyint(min_value=200, max_value=300),
            fake.pyint(min_value=200, max_value=300),
            fake.pyint(min_value=-10, max_value=10),
            fake.pyint(min_value=1, max_value=7),
            ]


def test_run_train(runner: CliRunner) -> None:
    result = runner.invoke(train, [], )
    assert result.exit_code == 0


def test_run_train_fake_data(runner: CliRunner) -> None:
    FAKE_DATA_PATH = 'data/fake.csv'
    csvfile = open(FAKE_DATA_PATH, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(
        ['Id', 'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
         'Horizontal_Distance_To_Roadways', 'Cover_Type'])
    for n in range(1, 100):
        writer.writerow(generate_data(n))
    csvfile.close()

    result = runner.invoke(train, ["--dataset-path", FAKE_DATA_PATH, ], )
    assert result.exit_code == 0


def test_error_for_invalid_logreg(runner: CliRunner) -> None:
    result = runner.invoke(train, ["--logreg-c", -1, ], )
    assert result.exit_code == 2
    assert "logreg_c format must (0,1]" in result.output
