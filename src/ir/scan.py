import torch
from torch import Tensor


def to_line_scanable_sequence(x: Tensor, smooth: bool, style: str):
    B, C, H, W = x.shape
    x = x.clone()

    """
    Imagine H, W as y, x axes here.

    0, 1, 2
    3, 4, 5
    6, 7, 8
    """
    if style in ("h_forward", "h_backward"):
        # h_forward: "...scanning from the top left to the bottom right"
        if smooth:
            """
            Along the sequence dimension L we would get
            => 0, 1, 2, 5, 4, 3, 6, 7, 8
            """
            x[:, :, 1::2, :] = x[:, :, 1::2, :].flip(3)
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BxLxC
        else:
            """
            Along the sequence dimension L we would get
            => 0, 1, 2, 3, 4, 5, 6, 7, 8
            """
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BxLxC
        
        if style == "h_backward":
            """
            Along the sequence dimension L we would get either
            => 8, 7, 6, 3, 4, 5, 2, 1, 0
            or
            => 8, 7, 6, 5, 4, 3, 2, 1, 0
            """
            x = x.flip(1)  # BxLxC
    
    elif style in ("w_forward", "w_backward"):
        # w_forward: "...scanning from the bottom left to the top right"
        if smooth:
            """
            Starting from
            0, 1, 2
            3, 4, 5
            6, 7, 8
            and after permute
            0, 3, 6
            1, 4, 7
            2, 5, 8

            Along the sequence dimension L we would get
            => 6, 3, 0, 1, 4, 7, 8, 5, 2
            """
            x = x.permute(0, 1, 3, 2)  # BxCxWxH
            x[:, :, 0::2, :] = x[:, :, 0::2, :].flip(3)
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BxLxC
            
        else:
            """
            Starting from
            0, 1, 2
            3, 4, 5
            6, 7, 8
            and after permute
            0, 3, 6
            1, 4, 7
            2, 5, 8
            and after reverse in H dimension
            6, 3, 0
            7, 4, 1
            8, 5, 2

            Along the sequence dimension L we would get
            => 6, 3, 0, 7, 4, 1, 8, 5, 2
            """
            x = x.permute(0, 1, 3, 2)  # BxCxWxH
            x = x.flip(3)  # Reverse in H dimension
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BxLxC

        if style == "w_backward":
            """
            Along the sequence dimension L we would get either
            => 2, 5, 8, 7, 4, 1, 0, 3, 6
            or
            => 2, 5, 8, 1, 4, 7, 0, 3, 6
            """
            x = x.flip(1)  # BxLxC

    else:
        raise ValueError(f"Unknown {style=}?")

    return x


if __name__ == "__main__":
    # x = torch.tensor([
    #     [0, 1, 2],
    #     [3, 4, 5],
    #     [6, 7, 8],
    # ]).reshape(1, 1, 3, 3)
    # seq = to_line_scanable_sequence(x, True, style="h_forward")

    import unittest

    class TestFunction(unittest.TestCase):

        def setUp(self):
            self.x = torch.tensor([
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ]).reshape(1, 1, 3, 3)  # Add dummy batch and channel dimensions.

        def test_h_forward(self):
            seq = to_line_scanable_sequence(self.x, smooth=False, style="h_forward")
            tgt = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(1, -1, 1)  # BxLxC
            self.assertTrue(torch.allclose(seq, tgt))

        def test_smooth_h_forward(self):
            seq = to_line_scanable_sequence(self.x, smooth=True, style="h_forward")
            tgt = torch.tensor([0, 1, 2, 5, 4, 3, 6, 7, 8]).reshape(1, -1, 1)  # BxLxC
            self.assertTrue(torch.allclose(seq, tgt))
        
        def test_h_backward(self):
            seq = to_line_scanable_sequence(self.x, smooth=False, style="h_backward")
            tgt = torch.tensor([8, 7, 6, 5, 4, 3, 2, 1, 0]).reshape(1, -1, 1)  # BxLxC
            self.assertTrue(torch.allclose(seq, tgt))

        def test_smooth_h_backward(self):
            seq = to_line_scanable_sequence(self.x, smooth=True, style="h_backward")
            tgt = torch.tensor([8, 7, 6, 3, 4, 5, 2, 1, 0]).reshape(1, -1, 1)  # BxLxC
            self.assertTrue(torch.allclose(seq, tgt))

        def test_w_forward(self):
            seq = to_line_scanable_sequence(self.x, smooth=False, style="w_forward")
            tgt = torch.tensor([6, 3, 0, 7, 4, 1, 8, 5, 2]).reshape(1, -1, 1)  # BxLxC
            self.assertTrue(torch.allclose(seq, tgt))
        
        def test_smooth_w_forward(self):
            seq = to_line_scanable_sequence(self.x, smooth=True, style="w_forward")
            tgt = torch.tensor([6, 3, 0, 1, 4, 7, 8, 5, 2]).reshape(1, -1, 1)  # BxLxC
            self.assertTrue(torch.allclose(seq, tgt))

        def test_w_backward(self):
            seq = to_line_scanable_sequence(self.x, smooth=False, style="w_backward")
            tgt = torch.tensor([2, 5, 8, 1, 4, 7, 0, 3, 6]).reshape(1, -1, 1)  # BxLxC
            self.assertTrue(torch.allclose(seq, tgt))
        
        def test_smooth_w_backward(self):
            seq = to_line_scanable_sequence(self.x, smooth=True, style="w_backward")
            tgt = torch.tensor([2, 5, 8, 7, 4, 1, 0, 3, 6]).reshape(1, -1, 1)  # BxLxC
            self.assertTrue(torch.allclose(seq, tgt))

    unittest.main()
    