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
                        fps_mode="manual (use fps below)",
                        reference=None, reference_mix=0.0)
    got = out["waveform"].shape[-1]
    assert got == expect, f"retimed length {got} != world target {expect}"
    print(f"PASS length (manual): retimed {got} samples == world clock exactly")

    # auto mode: audio decoded on a slightly different clock (padded to a
    # 1024-sample latent multiple) must still retime to the world target and
    # detect the implied fps.
    n_pad = ((n_src // 1024) + 1) * 1024
    wav_pad = torch.zeros(1, 2, n_pad)
    for f in range(sum(holds)):
        i = int(f * (n_pad / sum(holds)))
        wav_pad[:, :, i:i + 40] = 1.0
    out_a, = node.recover({"waveform": wav_pad, "sample_rate": SR}, hm,
                          fps=25, fps_mode="auto (match audio to hold map)",
                          reference=None, reference_mix=0.0)
    got_a = out_a["waveform"].shape[-1]
    # the decoder pad is distributed proportionally across the map, so the
    # retimed length may exceed the world target by the pad's share; keep it
    # within 1% (a drift beyond that would be audible as progressive lag).
    assert abs(got_a - expect) <= expect // 100, (
        f"auto-mode retime {got_a} != world target {expect}")
    print(f"PASS auto clock: padded audio {n_pad} -> retimed {got_a} "
          f"(target {expect}, within 1%)")

    # the strong invariant: audio whose length matches the dilated clock
    # exactly must retime to the world target exactly.
    out_e, = node.recover({"waveform": wav, "sample_rate": SR}, hm,
                          fps=25, fps_mode="auto (match audio to hold map)",
                          reference=None, reference_mix=0.0)
    assert out_e["waveform"].shape[-1] == expect, (
        f"auto retime of exact-length audio gave "
        f"{out_e['waveform'].shape[-1]} != {expect}")
    print(f"PASS auto clock exact: retimed {out_e['waveform'].shape[-1]} "
          f"== world target {expect}")

    # manual mode with a wrong fps must fire the drift warning
    node.recover({"waveform": wav, "sample_rate": SR}, hm, fps=50,
                 fps_mode="manual (use fps below)",
                 reference=None, reference_mix=0.0)
    print("PASS manual fps-mismatch warning executed (see console above)")

    # legacy guard: a workflow saved with the older node layout can deliver
    # a bare int for fps_mode (positional widget mapping). It must fall
    # back to the safe auto clock instead of crashing.
    out_l, = node.recover({"waveform": wav, "sample_rate": SR}, hm, fps=1,
                          fps_mode=25, reference=None, reference_mix=0.0)
    assert out_l["waveform"].shape[-1] == expect, \
        f"legacy fps_mode fallback gave {out_l['waveform'].shape[-1]} != {expect}"
    print("PASS legacy guard: int fps_mode falls back to auto clock, "
          f"retime {out_l['waveform'].shape[-1]} == target {expect}")

    ref = torch.randn(1, 2, int(round(len(holds) * spf)))
    out2, = node.recover({"waveform": wav, "sample_rate": SR}, hm, fps=FPS,
                         fps_mode="manual (use fps below)",
                         reference={"waveform": ref, "sample_rate": SR},
                         reference_mix=1.0)
    n_out = out2["waveform"].shape[-1]
    assert torch.equal(out2["waveform"], ref[:, :, :n_out]), \
        "mix=1.0 output is not the reference bit-for-bit"
    print(f"PASS identity: mix=1.0 == reference[:{n_out}] bit-for-bit")

    node.recover({"waveform": wav, "sample_rate": SR}, hm, fps=FPS,
                 fps_mode="manual (use fps below)",
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
