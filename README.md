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
- **LTX 2.3 Audio Recover** — LTX adaptation of MAINodes' H3 Audio Recover. Wiring: audio from `VAEDecodeAudio` of the **regenerated** latent, `hold_map_used` from the same `LTX23TimeSmear` the video used (not the raw oracle `hold_map` — the smear aligns it to the image batch and snaps the tail to the legal 8k+1 grid). `fps_mode` defaults to **auto (detect audio clock)**: it measures the decoded audio against both clocks the map implies — the DILATED duration (`sum(holds)/fps`) and the WORLD duration (`len(holds)/fps`) — and retimes only when the audio really is on the dilated clock; world-rate audio (what you get when pass 2's audio rows were seeded from pass 1's own output, since `LTXVConcatAVLatent` does **not** stretch audio tokens to the dilated timeline) is passed through untouched. Compressing world-rate audio used to cut the sound short after a few seconds. The explicit modes force a behaviour, `manual` assumes a dilated clock at the fps widget. The node prints a clock diagnostic per run — check the console if lipsync drifts. If pass 2's video mouths drift from the source performance, blend the original clip's audio back in via `reference` + `reference_mix=1`.

There is a synthetic regression test for the audio recover node (needs torch + torchaudio, no GPU):

```
python test_audio_recover.py
```

# Comfy UI Workflow

My workflow that I use, which is an Image to Video workflow for LTX 2.5, uses the MSR lora and its accompanying custom nodes. It is located in the `workflows` directory. Beware, there is quite a bit of spaghetti, and a lot of sampling passes. This was only really meant for myself and it shows. Tried to neat it up a bit. Adjust the workflow as you see fit.
