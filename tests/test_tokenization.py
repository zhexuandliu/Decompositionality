import torch

from decompositionality import tokenization as T


def test_layout_and_ids():
    assert T.PAD_ID != 0 and T.PAD_ID not in range(10)
    assert len(set([T.PAD_ID, T.BOS_ID, T.OPS_ID, T.GRID_ID, T.OUTPUT_ID, T.EOS_ID, 10, 11, 12])) == 9
    ops = torch.tensor([[0, 1, 3]])
    x = torch.arange(36).remainder(10).view(1, 6, 6)
    y = (torch.arange(36) * 7).remainder(10).view(1, 6, 6)
    seq = T.ar_serialize(ops, x, y)[0]
    assert seq.shape[0] == 80
    assert seq[0] == T.BOS_ID and seq[1] == T.OPS_ID and seq[5] == T.GRID_ID
    assert seq[2:5].tolist() == [10, 11, T.PAD_ID]
    assert seq[42] == T.OUTPUT_ID and seq[79] == T.EOS_ID
    assert seq[6:42].tolist() == x.flatten().tolist() and seq[43:79].tolist() == y.flatten().tolist()
    pre = T.ar_prefix(ops, x)
    assert pre.shape[1] == 43 and torch.equal(pre[0], seq[:43])
    meta = T.ar_layout_metadata()
    assert meta["row"][43].item() == 0 and meta["col"][48].item() == 5 and meta["row"][78].item() == 5
    assert meta["row"][42].item() == T.NO_COORD
