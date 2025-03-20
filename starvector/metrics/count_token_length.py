import torch
from torch.utils.data import DataLoader
from starvector.metrics.base_metric import BaseMetric
from tqdm import tqdm
from starvector.metrics.util import AverageMeter

from transformers import AutoTokenizer

class CountTokenLength(BaseMetric): 
    def __init__(self, config=None, device='cuda'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b")
        self.metric = self.calculate_token_length
        self.meter_gt_tokens = AverageMeter()
        self.meter_gen_tokens = AverageMeter()
        self.meter_diff = AverageMeter()

    def calculate_token_length(self, **kwargs):
        svg = kwargs.get('gt_svg')
        tokens = self.tokenizer.encode(svg)
        gen_svg = kwargs.get('gen_svg')
        gen_tokens = self.tokenizer.encode(gen_svg)
        diff = len(gen_tokens) - len(tokens)
        return len(tokens), len(gen_tokens), diff

    def calculate_score(self, batch, update=None):
        gt_svgs = batch['gt_svg']
        gen_svgs = batch['gen_svg']
        values = []
        for gt_svg, gen_svg in tqdm(zip(gt_svgs, gen_svgs), total=len(gt_svgs), desc="Processing SVGs"):
            gt_tokens, gen_tokens, diff = self.calculate_token_length(gt_svg=gt_svg, gen_svg=gen_svg)
            self.meter_gt_tokens.update(gt_tokens, 1)
            self.meter_gen_tokens.update(gen_tokens, 1)
            self.meter_diff.update(diff, 1)
            values.append({
                'gt_tokens': gt_tokens,
                'gen_tokens': gen_tokens,
                'diff': diff
            })
        avg_score = {
            'gt_tokens': self.meter_gt_tokens.avg,
            'gen_tokens': self.meter_gen_tokens.avg,
            'diff': self.meter_diff.avg
        }
        if not values:
            print("No valid values found for metric calculation.")
            return float("nan")

        return avg_score, values

    def reset(self):
        self.meter_gt_tokens.reset()
        self.meter_gen_tokens.reset()
        self.meter_diff.reset()
