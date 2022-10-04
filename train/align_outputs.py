from argparse import ArgumentParser
from dkm import (
    DKMv2,
)
import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--state", type=str, required=True)
    args, _ = parser.parse_known_args()

    state_dict = torch.load(args.state)
    model = DKMv2(pretrained=True, version="outdoor", resolution="high")
    model.load_state_dict(state_dict['model'].state_dict(), strict=True)
    model.train()

    with torch.no_grad():
        outputs = state_dict['model'](state_dict['train_batch'])
        diff = (outputs[16]['dense_flow'] - state_dict['out'][16]['dense_flow']).abs()
        print("Max Diff: %f, Mean Diff: %f" % (diff.max().item(), diff.mean().item()))

        outputs = model(state_dict['train_batch'])
        diff = (outputs[16]['dense_flow'] - state_dict['out'][16]['dense_flow']).abs()
        print("Max Diff: %f, Mean Diff: %f" % (diff.max().item(), diff.mean().item()))
