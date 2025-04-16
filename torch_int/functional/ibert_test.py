import unittest
import torch
from ibert import IBertComputation, IBertQuant


class IBertTests(unittest.TestCase):
    def test_softmax(self):
        logits = torch.tensor([2.0, 1.0, 0.1, 0.0])
        quantized_logits = IBertQuant.quant_to_sint8(logits)

        quantized_res = IBertComputation.softmax(quantized_logits)
        q, S = quantized_res.q, quantized_res.S

        print(q, S)
        print(q*S)

