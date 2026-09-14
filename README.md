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
- **LTX 2.3 Audio Recover** — LTX adaptation of MAINodes' H3 Audio Recover. Retimes the regenerated clip's audio back to the original clock using the same hold map as the video (phase vocoder, so pitch is kept). Wire the audio from `VAEDecodeAudio` of the regenerated latent, `hold_map_used` from the same `LTX23TimeSmear`, and set `fps` to the frame rate you generated at (e.g. 25 or 50). If pass 2's audio rows were not seeded, blend the original clip's audio back in via `reference` + `reference_mix=1` to fix rushed lipsync in held regions.

There is a synthetic regression test for the audio recover node (needs torch + torchaudio, no GPU):

```
python test_audio_recover.py
```

# Comfy UI Workflow

My workflow that I use, which is an Image to Video workflow for LTX 2.5, uses the MSR lora and its accompanying custom nodes. It is located in the `workflows` directory. Beware, there is quite a bit of spaghetti, and a lot of sampling passes. This was only really meant for myself and it shows. Tried to neat it up a bit. Adjust the workflow as you see fit.
