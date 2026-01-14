import logging

from loader import load_clean
from descriptives import describe_by_group
from correlations import point_biserial
from tests import welch_test, mann_whitney
from regression import run_regression, regression_diagnostics
from logistic import logistic_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_all():
    df = load_clean()

    describe_by_group(df)
    point_biserial(df)
    welch_test(df)
    mann_whitney(df)

    model = run_regression(df)
    regression_diagnostics(model)

    logistic_check(df)


    logger.info("All statistical analysis completed successfully")

if __name__ == "__main__":
    run_all()

import loader
print(loader.__file__)

