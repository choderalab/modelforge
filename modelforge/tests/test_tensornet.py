from modelforge.potential.tensornet import TensorNet


def test_tensornet_init():
    net = TensorNet()
    assert net is not None
