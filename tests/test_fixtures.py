from .fixtures import the_veredict

def test_get_the_veredict(the_veredict):
    assert len(the_veredict) == 20480
