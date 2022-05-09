from faker import Faker

from eval_sel.pipeline import create_pipeline


def test_std_scaler() -> None:
    random = Faker().pyint()
    log_reg_c = Faker().pyfloat(positive=True, min_value=0.000_01, max_value=1)
    max_iter = Faker().pyint(10, 9999)

    pipeline = create_pipeline(True, max_iter, log_reg_c, random)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'scaler'
    assert pipeline.steps[1][0] == 'classifier'

    pipeline = create_pipeline(False, max_iter, log_reg_c, random)
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0][0] == 'classifier'
