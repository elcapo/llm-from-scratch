from .fixtures import the_veredict

def test_get_the_veredict(the_veredict):
    # Prepare, act and assert
    assert len(the_veredict) == 20480
