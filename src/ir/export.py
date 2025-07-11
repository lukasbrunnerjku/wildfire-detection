import torch
from pathlib import Path

from .mambaout import MambaOutIR


def load_model(checkpoint: Path):
    model = MambaOutIR(
        1,
        (64, 128, 256),
        (2, 2, 3),
        num_class_embeds=31,
        oss_refine_blocks=2,
        local_embeds=False,
        drop_path=0.2,
        with_stem=True,
        scriptable=True,  # !!
    )
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    return model  # On CPU by default.


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    
    model = load_model(args.checkpoint).cuda()
    
    ### EXPORT ###
    scripted_model = torch.jit.script(model)
    f = args.checkpoint.parent / "scripted_model.pt"
    scripted_model.save(f)
    print(f"Exported to: {f}")
    
    ### TEST ###
    loaded_model = torch.jit.load(f)

    x = torch.randn(1, 1, 512, 512).cuda()
    emb = torch.randint(0, 31, (1,), dtype=torch.int64).cuda()
    sample_input = (x, emb, None, True)
    
    output = loaded_model(*sample_input)
    print(f"{output.shape=}")
    