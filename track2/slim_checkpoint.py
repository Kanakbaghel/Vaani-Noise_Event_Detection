"""slim_checkpoint.py - strip the frozen SraVaani encoder out of a Track 2 checkpoint.

MaskNet holds the frozen SraVaani-1.0 encoder as a submodule, so model.state_dict()
(and therefore t2_model_real.pt) also contains its ~866 MB of weights. The encoder is
downloaded from Hugging Face at load time anyway, so only the trained mask-network
weights need to be shipped (about 10-15 MB).

Usage:
    python track2/slim_checkpoint.py t2_model_real.pt t2_model_real_slim.pt
"""
import os
import sys

import torch


def main(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("enc.")}
    out = {k: v for k, v in ck.items() if k != "model"}
    out["model"] = sd
    torch.save(out, dst)
    print(f"kept {len(sd)} of {len(ck['model'])} tensors | "
          f"{os.path.getsize(src)/1e6:.1f} MB -> {os.path.getsize(dst)/1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
