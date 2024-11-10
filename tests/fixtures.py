import os
from pathlib import Path
import pytest

@pytest.fixture(scope='session', autouse=True)
def the_veredict():
    dirname = os.path.dirname(__file__)
    return Path(dirname + "/fixtures/the-veredict.txt").read_text()
