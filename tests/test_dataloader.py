from torch import equal, tensor
from scratch.dataloader import create_dataloader
from .fixtures import the_veredict

def test_create_dataloader(the_veredict):
    dataloader = create_dataloader(
        the_veredict,
        batch_size=1,
        max_length=4,
        stride=1,
        shuffle=False)
    
    data_iterator = iter(dataloader)
    first_batch = next(data_iterator)

    assert equal(first_batch[0], tensor([[40, 367, 2885, 1464]]))
    assert equal(first_batch[1], tensor([[367, 2885, 1464, 1807]]))
