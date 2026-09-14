#!/usr/bin/env python3
"""Synthetic regression test for LTX23AudioRecover (no GPU, no files).

Run with the ComfyUI python (needs torch + torchaudio):
    python tests_audio_recover.py

Checks: retimed length is sample-exact against the world clock across mixed
hold runs; reference_mix=1.0 returns the reference bit-for-bit; the
unwired-reference warning path executes; empty-hold-map guard fires.
"""
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "ltx23_motion", os.path.join(HERE, "ltx23_motion.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SR, FPS = 32000, 25

def main():
    holds = [1] * 20 + [4] * 30 + [1] * 20 + [2] * 17 + [1] * 20
    spf = SR / FPS
    n_src = int(round(sum(holds) * spf))
    wav = torch.zeros(1, 2, n_src)
    for f in range(sum(holds)):
        i = int(f * spf)
        wav[:, :, i:i + 40] = 1.0

    node = m.LTX23AudioRecover()
    hm = json.dumps({"holds": holds, "world_len": sum(holds), "units": "frames"})

    runs = []
    for h in holds:
        if runs and runs[-1][0] == h:
            runs[-1][1] += 1
        else:
            runs.append([h, 1])
    expect = sum(int(round(c * spf)) for _, c in runs)

    out, = node.recover({"waveform": wav, "sample_rate": SR}, hm, fps=FPS,
                        reference=None, reference_mix=0.0)
    got = out["waveform"].shape[-1]
    assert got == expect, f"retimed length {got} != world target {expect}"
    print(f"PASS length: retimed {got} samples == world clock exactly")

    ref = torch.randn(1, 2, int(round(len(holds) * spf)))
    out2, = node.recover({"waveform": wav, "sample_rate": SR}, hm, fps=FPS,
                         reference={"waveform": ref, "sample_rate": SR},
                         reference_mix=1.0)
    n_out = out2["waveform"].shape[-1]
    assert torch.equal(out2["waveform"], ref[:, :, :n_out]), \
        "mix=1.0 output is not the reference bit-for-bit"
    print(f"PASS identity: mix=1.0 == reference[:{n_out}] bit-for-bit")

    node.recover({"waveform": wav, "sample_rate": SR}, hm, fps=FPS,
                 reference=None, reference_mix=1.0)
    print("PASS warning path executed (see console line above)")

    try:
        node.recover({"waveform": wav, "sample_rate": SR}, "")
    except ValueError as e:
        print(f"PASS empty-hold-map guard: {e}")
    else:
        raise AssertionError("empty hold_map should raise")

    # registration sanity
    assert "LTX23AudioRecover" in m.NODE_CLASS_MAPPINGS
    assert m.NODE_DISPLAY_NAME_MAPPINGS["LTX23AudioRecover"] == "LTX 2.3 Audio Recover"
    print("PASS registration: node is in NODE_CLASS_MAPPINGS")

    print("ALL PASS")

if __name__ == "__main__":
    main()
