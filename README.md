# MAINodes for LTX2.3/2.5

Experimental custom nodes for LTX 2.3/2.5. Dramatically reduce smearing in LTX 2.3/2.5 outputs. All credits goes to matlowai [for the initial implementation](https://github.com/matlowai/ComfyUI-MAINodes), I just converted it to work with LTX 2.3/2.5.


# Installation


```
cd custom_nodes/

git clone https://github.com/sillylilithhh/ComfyUI-LTX23-MAINodes
```

# Nodes

- **LTX 2.3 Jerk Oracle** — builds the adaptive hold map from the video latent.
- **LTX 2.3 Time Smear** — holds input IMAGE frames on the hold map and snaps the sequence to LTX's legal 8k+1 frame grid.
- **LTX 2.3 Exact Recover** — inverts the time smear, keeping the first generated frame of every hold group.
- **LTX 2.3 Audio Smear** — the audio counterpart to `LTX 2.3 Time Smear`. Stretches the baseline clip's audio onto the **same dilated timeline** the smeared video init lives on using a phase vocoder (preserving pitch). Wire baseline audio and `hold_map_used` from `LTX23TimeSmear`, then pass the output to `LTXVAudioVAEEncode` and feed that dilated `audio_latent` into Pass 2's `LTXVConcatAVLatent`. This ensures Pass 2 renders speech at the slowed rate in lockstep with the video, preserving lip sync when recovered.
- **LTX 2.3 Audio Recover** — LTX adaptation of MAINodes' H3 Audio Recover. Retimes the regenerated clip's audio back to the real-time world clock using the same `hold_map_used`. Wiring: audio from `LTXVAudioVAEDecode` of the regenerated latent, and `hold_map_used` from `LTX23TimeSmear`. `fps_mode` defaults to **auto (detect audio clock)**. When Pass 2 is seeded with `LTX23AudioSmear`, the decoded audio is dilated (`fits_dilated=True`) and will be retimed back to the original length, perfectly matching the video recovered by `LTX23ExactRecover`. If Pass 2 was NOT seeded with smeared audio, the decoded audio remains at world duration and is passed through untouched to prevent sound truncation.

There is a synthetic regression test for the audio recover node (needs torch + torchaudio, no GPU):

```
python test_audio_recover.py
```

# Working lipsync configuration (proven)

`workflows/LTX-2.5_new - AIO (working lipsync config).json` is a full working example. The recipe that made lipsync work:

1. **De-smear samplers use the canonical heavy schedule**: `0.725, 0.4219, 0.0` with `euler_ancestral` (matching the original `LTX 2.5-SmearRemoval-MSR.json`). A light refiner (e.g. `0.30, 0.18, 0.08, 0.0`) makes pass 2 half-preserve / half-re-time the performance, which breaks lipsync against the original audio.
2. **Seed pass 2's audio rows with LTX23AudioSmear**: decode pass 1's audio rows, stretch them on the SAME `hold_map_used` as the video, `LTXVAudioVAEEncode` -> `LTXVConcatAVLatent` for pass 2. Then audio and video rows agree on the dilated clock.
3. **LTX23AudioRecover** with `fps_mode = auto (detect audio clock)` retimes the regenerated dilated audio back to the world clock; `reference_mix = 1.0` returns the original track. With the heavy schedule the recovered video tracks the original performance, so the original audio syncs.
4. The `sync check` console line is informational: comparing two different takes (regenerated re-render vs the original recording) typically scores 0.2-0.4 envelope correlation even with correct timing, so a `POOR sync` label there does not mean broken lipsync - judge by eye/ear. Use it to detect regressions instead (corr collapsing toward 0 or the lag swinging by seconds).

# Comfy UI Workflow

My workflow that I use, which is an Image to Video workflow for LTX 2.5, uses the MSR lora and its accompanying custom nodes. It is located in the `workflows` directory. Beware, there is quite a bit of spaghetti, and a lot of sampling passes. This was only really meant for myself and it shows. Tried to neat it up a bit. Adjust the workflow as you see fit.
