"""LTX-2.3 adaptation of the three MAINodes motion nodes.

The original implementation targets MiniMax/H3's 17k+5 temporal grid and
24-channel latent layout.  LTX-2.3 instead uses a 128-channel video latent
with temporal VAE compression of 8x, so legal video lengths are 8k+1.

Only the nodes requested for LTX-2.3 are included here:
    LTX23JerkOracle
    LTX23TimeSmear
    LTX23ExactRecover
    LTX23AudioSmear
    LTX23AudioRecover

The algorithm itself is intentionally kept close to the original MAINodes
implementation: jerk/trajectory profile -> quantile threshold -> ramped hold
map -> integer frame smear -> exact first-frame recovery.
"""

import json
import math

import numpy as np
import torch

LTX23_LATENT_CHANNELS = 128
LTX23_TEMPORAL_SCALE = 8
LTX23_MIN_LEGAL_FRAMES = 9

# Cost model is retained from the original nodes because it is only a report
# and does not affect the actual node outputs.
COST_EXP = 1.7
OVERHEAD_S = 6.7

PROFILE_MODES = {
    "value |d3| (default)": ("value", 3),
    "value |d1| (energy baseline)": ("value", 1),
    "trajectory centroid |d3|": ("traj", 3),
}


def _video_component(samples):
    """Return a plain LTX video latent tensor shaped (B, 128, T, H, W).

    LTX-2.3 video latents are standard ComfyUI LATENT dictionaries with
    128 channels and 8x temporal compression.  We accept a nested tensor if
    one is supplied, matching the original node's tolerance for nested data.
    """
    if not isinstance(samples, dict) or "samples" not in samples:
        raise ValueError("Expected a LATENT containing a 'samples' tensor")
    z = samples["samples"]
    if hasattr(z, "is_nested") and z.is_nested:
        z = z.tensors[0]
    if not torch.is_tensor(z) or z.ndim != 5:
        raise ValueError(
            f"Expected a 5D LTX video latent [B, C, T, H, W], got {type(z).__name__} "
            f"with shape {getattr(z, 'shape', None)}"
        )
    if z.shape[1] != LTX23_LATENT_CHANNELS:
        raise ValueError(
            f"This node is for LTX-2.3 video latents: expected 128 channels, "
            f"got {z.shape[1]}."
        )
    return z


def _value_profile(v, order):
    """Per-token |Δ^order| of latent values, averaged over C/H/W."""
    j = np.abs(np.diff(v, n=order, axis=2)).mean(axis=(0, 1, 3, 4))
    lead = order // 2
    return np.pad(j, (lead, v.shape[2] - len(j) - lead), mode="edge")


def _trajectory_profile(v, order=3):
    """Per-token derivative magnitude of the latent energy centroid path."""
    e = np.abs(v).mean(axis=(0, 1))  # (T, H, W)
    e = e - e.min(axis=(1, 2), keepdims=True)
    tot = e.sum(axis=(1, 2)) + 1e-8
    ys = np.arange(e.shape[1], dtype=np.float64)[None, :, None]
    xs = np.arange(e.shape[2], dtype=np.float64)[None, None, :]
    cy = (e * ys).sum(axis=(1, 2)) / tot
    cx = (e * xs).sum(axis=(1, 2)) / tot
    path = np.stack([cy, cx], axis=1)
    j = np.linalg.norm(np.diff(path, n=order, axis=0), axis=1)
    lead = order // 2
    return np.pad(j, (lead, path.shape[0] - len(j) - lead), mode="edge")


def _phase_norm(prof):
    # Kept verbatim in spirit from the original implementation.  LTX has a
    # uniform temporal latent clock, so this normalization is no longer
    # compensating for H3's (1,4,4,4,4) temporal phase structure.  It is
    # therefore optional here and disabled by default.
    return prof


def _jerk_profile(z, mode="value |d3| (default)", phase_norm=False):
    v = z.detach().float().cpu().numpy()
    kind, order = PROFILE_MODES.get(mode, ("value", 3))
    prof = _value_profile(v, order) if kind == "value" else _trajectory_profile(v, order)
    return _phase_norm(prof) if phase_norm else prof


def _legal_ceil(n):
    """Ceil a pixel-frame count to LTX's legal 8k+1 temporal grid."""
    n = max(1, int(n))
    if n <= 1:
        return 1
    if n <= LTX23_MIN_LEGAL_FRAMES:
        return LTX23_MIN_LEGAL_FRAMES
    return 1 + 8 * math.ceil((n - 1) / 8)


def _grid_token_count(frames):
    """Number of LTX video latent time positions for a legal frame count."""
    frames = _legal_ceil(frames)
    return ((frames - 1) // 8) + 1


def _frame_token(f, t_lat):
    """Map a pixel frame to its LTX latent temporal cell.

    LTX has one anchor frame followed by 8-frame intervals:
      token 0 -> frame 0
      token 1 -> frames 1..8
      token 2 -> frames 9..16
      ...
    """
    if t_lat <= 1:
        return 0
    return min((int(f) + 7) // 8, t_lat - 1)


def _latent_holds_to_frame_holds(latent_holds, n_frames):
    """Expand LTX latent-time hold values onto decoded IMAGE frames.

    LTX's temporal VAE clock is 1 + 8k: latent token 0 owns frame 0, and
    each following latent token owns the next 8 decoded frames.  The oracle
    naturally works on latent positions; Time Smear works on actual IMAGE
    frames.  This bridges the two clocks and crops the final block to the
    exact image-batch length when a downstream node is using a shorter clip.
    """
    latent_holds = [int(h) for h in latent_holds]
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    if not latent_holds or min(latent_holds) < 1:
        raise ValueError("latent hold counts must be positive")

    out = []
    for token, hold in enumerate(latent_holds):
        block_len = 1 if token == 0 else 8
        remaining = n_frames - len(out)
        if remaining <= 0:
            break
        out.extend([hold] * min(block_len, remaining))

    if len(out) < n_frames:
        # A short map can occur when a workflow crops after the oracle.
        # Continue the final known latent rate rather than failing.
        out.extend([latent_holds[-1]] * (n_frames - len(out)))

    return out


def _profile_to_plan(prof, length, q, d_max, ramp, bridge):
    prof = np.asarray(prof, dtype=np.float64)
    t_lat = len(prof)
    if t_lat == 0:
        raise ValueError("The LTX latent has no temporal positions")

    thr = np.quantile(prof, q)
    tok_d = np.where(prof >= thr, d_max, 1).astype(int)

    if bridge:
        hot = np.where(tok_d == d_max)[0]
        for a, b in zip(hot[:-1], hot[1:]):
            if 1 < b - a <= bridge:
                tok_d[a:b + 1] = d_max

    if ramp:
        for _ in range(d_max - 1):
            left = np.concatenate([[1], tok_d[:-1]])
            right = np.concatenate([tok_d[1:], [1]])
            tok_d = np.maximum(tok_d, np.maximum(left, right) - 1)

    holds = [int(tok_d[_frame_token(f, t_lat)]) for f in range(length)]

    # Segment output stays useful for inspection.  Segment units are LTX
    # latent-time positions rather than H3 positions.
    segs = []
    t0 = 0
    for t in range(1, t_lat + 1):
        if t == t_lat or tok_d[t] != tok_d[t0]:
            if tok_d[t0] > 1:
                segs.append(f"{t0}:{t}:{int(tok_d[t0])}")
            t0 = t

    hot = np.where(tok_d > 1)[0]
    if len(hot):
        w0 = _frame_token(8 * int(hot.min()), t_lat) * 8
        end_token = min(int(hot.max()) + 1, t_lat - 1)
        w1 = min(length, 8 * end_token + 1)
        wlen = max(0, w1 - w0)
    else:
        w0, wlen = 0, 0

    return holds, ",".join(segs), w0, wlen, tok_d


def _cost_report(world_len, dilated, fps=24, s_per_step=0.0, est_steps=18,
                 overhead_s=OVERHEAD_S, tail=""):
    world_len = max(1, int(world_len))
    dilated = max(world_len, int(dilated))
    fps = max(1, int(fps))
    t_world = _grid_token_count(world_len)
    t_dil = _grid_token_count(dilated)
    time_x = (t_dil / t_world) ** COST_EXP
    report = (
        f"{world_len}f ({world_len / fps:.1f}s) -> {dilated}f "
        f"({dilated / fps:.1f}s) effective regen, "
        f"{dilated / world_len:.2f}x frames / {time_x:.1f}x time per "
        f"step; LTX latent frames {t_world} -> {t_dil}"
    )
    if tail:
        report += f"; {tail}"
    if s_per_step > 0:
        secs = time_x * (overhead_s + s_per_step * max(1, int(est_steps)))
        report += (
            f"; roughly {secs / 60:.0f} min at {s_per_step:g} s/step x "
            f"{int(est_steps)} steps (+{overhead_s:g}s encode/decode, "
            f"all x{time_x:.1f})"
        )
    return report


def _cost_widgets(with_fps=False):
    w = {}
    if with_fps:
        w["fps"] = ("INT", {
            "default": 24,
            "min": 1,
            "max": 120,
            "tooltip": "Only used to phrase the report in seconds.",
        })
    w["s_per_step"] = ("FLOAT", {
        "default": 0.0,
        "min": 0.0,
        "max": 3600.0,
        "step": 0.1,
        "tooltip": "Optional baseline sampler seconds/step used only for the estimate.",
    })
    w["est_steps"] = ("INT", {
        "default": 18,
        "min": 1,
        "max": 200,
        "tooltip": "Optional number of sampler steps used only for the estimate.",
    })
    w["overhead_s"] = ("FLOAT", {
        "default": OVERHEAD_S,
        "min": 0.0,
        "max": 600.0,
        "step": 0.1,
        "tooltip": "Optional encode/decode overhead used only for the estimate.",
    })
    return w


def expand_hold_map_to_end(holds):
    """Same end-jump fix as the original, adapted to the LTX 8-frame clock."""
    holds = [int(h) for h in holds]
    if not holds or min(holds) < 1:
        raise ValueError("hold counts must be positive")

    # For LTX, a tail up to one temporal block (8 frames) is treated as the
    # possible end jump. Longer tails are left as intentional quiet/rest.
    max_end_tail = LTX23_TEMPORAL_SCALE
    n = len(holds)
    tail = 0
    while tail < n and holds[n - 1 - tail] == 1:
        tail += 1
    if tail == 0 or tail == n or tail > max_end_tail:
        return holds, None

    start = n - tail - 1
    rate = holds[start]
    while start > 0 and holds[start - 1] == rate:
        start -= 1

    if rate <= 1:
        return holds, None

    out = holds[:]
    for i in range(start, n):
        if out[i] == 1:
            out[i] = rate

    target = _legal_ceil(sum(out))
    deficit = target - sum(out)
    if deficit > 0:
        # Add the legal-grid padding inside the lifted end region.
        i = n - 1
        while deficit > 0:
            out[i] += 1
            deficit -= 1
            i -= 1
            if i < start:
                i = n - 1

    note = (
        f"LTX2.3 end-tail expansion: lifted a {tail}-frame rate-1 tail "
        f"to x{rate} and snapped the result to the 8k+1 grid"
    )
    return out, note


class LTX23JerkOracle:
    DESCRIPTION = (
        "LTX-2.3 adaptation of H3 Jerk Oracle. Reads the LTX video latent "
        "(128 channels, 8x temporal VAE compression) and emits a source-frame "
        "hold map by projecting the latent motion profile back onto LTX's "
        "8k+1 video-frame clock. The actual profile and hold compiler remain "
        "the original algorithm."
    )

    PRESETS = {
        "balanced (default)": {"q": 0.75, "d_max": 4, "ramp": True},
        "max quality (wide plateau)": {"q": 0.70, "d_max": 4, "ramp": True},
        "economy (tight spans)": {"q": 0.85, "d_max": 3, "ramp": True},
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "length": ("INT", {
                    "default": 97,
                    "min": 1,
                    "max": 3600,
                    "step": 8,
                    "tooltip": "Pixel-frame count in the source video; LTX legal lengths are 8k+1.",
                }),
                "q": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.5,
                    "max": 0.99,
                    "step": 0.01,
                    "tooltip": "Jerk quantile that counts as hot.",
                }),
                "d_max": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 8,
                    "tooltip": "Peak hold count on the hottest LTX latent positions.",
                }),
                "ramp": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "C1 ramp shoulders instead of hard steps.",
                }),
            },
            "optional": {
                "preset": (["custom"] + list(cls.PRESETS), {
                    "default": "balanced (default)",
                    "tooltip": "Any preset overrides q, d_max, and ramp.",
                }),
                "bridge": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 20,
                    "tooltip": "Fill short valleys between hot latent positions.",
                }),
                "profile_mode": (list(PROFILE_MODES), {
                    "default": "value |d3| (default)",
                }),
                "abstain_below": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Absolute peak-to-mean contrast gate; 0 disables it.",
                }),
                **_cost_widgets(with_fps=True),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("hold_map", "segments", "window_start", "window_len", "profile", "report")
    FUNCTION = "read"
    CATEGORY = "latent/ltx23/motion"

    def read(self, samples, length, q, d_max, ramp, preset="custom", bridge=8,
             profile_mode="value |d3| (default)", abstain_below=0.0,
             fps=24, s_per_step=0.0, est_steps=18, overhead_s=OVERHEAD_S):
        if preset in self.PRESETS:
            p = self.PRESETS[preset]
            q, d_max, ramp = p["q"], p["d_max"], p["ramp"]

        z = _video_component(samples)
        t_lat = z.shape[2]
        expected_min = _grid_token_count(length)
        if t_lat != expected_min:
            # The oracle can still run with the supplied latent, but the user
            # is very likely wiring the wrong frame count. Fail early with a
            # useful LTX-specific message rather than silently misaligning the
            # generated hold map.
            raise ValueError(
                f"LTX23JerkOracle length={length} expects {expected_min} temporal "
                f"latent positions, but the connected latent has {t_lat}. "
                f"Use the source video's actual frame count (legal LTX lengths are 8k+1)."
            )

        length = int(length)
        prof = _jerk_profile(z, profile_mode, phase_norm=False)
        contrast = float(prof.max() / max(prof.mean(), 1e-8))

        if abstain_below > 0.0 and contrast < abstain_below:
            flat = json.dumps({"holds": [1] * length, "world_len": int(length), "units": "frames"})
            return (
                flat,
                "",
                0,
                length,
                " ".join(f"{v:.2f}" for v in prof),
                _cost_report(
                    length,
                    _legal_ceil(length),
                    fps,
                    s_per_step,
                    est_steps,
                    overhead_s,
                    tail=f"abstained, profile contrast {contrast:.2f} < {abstain_below:g}",
                ),
            )

        holds, segs, w0, wlen, tok_d = _profile_to_plan(
            prof, length, q, d_max, ramp, bridge
        )
        # Project the latent-time profile back onto source/video frames.
        # This keeps the same contract as the original H3 oracle: one hold
        # value per source frame, which Time Smear can apply directly.
        hold_map = json.dumps({"holds": holds, "world_len": int(length), "units": "frames"})
        profile = " ".join(f"{v:.2f}" for v in prof)
        n_held = sum(1 for h in holds if h > 1)
        report = _cost_report(
            length,
            _legal_ceil(sum(holds)),
            fps,
            s_per_step,
            est_steps,
            overhead_s,
            tail=f"{n_held} of {length} frames held, peak x{int(tok_d.max())}",
        )
        return (hold_map, segs, int(w0), int(wlen), profile, report)


class LTX23TimeSmear:
    DESCRIPTION = (
        "LTX-2.3 adaptation of H3 Time Smear. Holds input IMAGE frames on an "
        "integer retime map and snaps the resulting sequence to LTX's legal "
        "8k+1 frame grid. Wire LTX23JerkOracle.hold_map for adaptive mode. "
        "Shorter/longer frame maps are automatically aligned to the incoming "
        "IMAGE batch; uncovered frames remain at native rate."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "dilation": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "tooltip": "Uniform hold count; ignored when hold_map is connected.",
                }),
            },
            "optional": {
                "hold_map": ("STRING", {
                    "default": "",
                    "tooltip": "hold_map from LTX23JerkOracle.",
                }),
                "expand_to_end": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Lift a short rate-1 tail behind an expansion span to reduce the end jump.",
                }),
                **_cost_widgets(with_fps=True),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "hold_map_used", "length", "report")
    FUNCTION = "smear"
    CATEGORY = "image/ltx23/motion"

    def smear(self, images, dilation, hold_map="", expand_to_end=True, fps=24,
              s_per_step=0.0, est_steps=18, overhead_s=OVERHEAD_S):
        images = images.detach().cpu()
        n = images.shape[0]
        if n < 1:
            raise ValueError("LTX23TimeSmear requires at least one image frame")

        map_units = "frames"
        alignment_note = None
        if hold_map.strip():
            data = json.loads(hold_map)
            holds = [int(h) for h in data["holds"]]
            map_units = data.get("units", "frames")

            if map_units in ("ltx_latent", "latent", "tokens"):
                # Explicit latent-time maps are supported for compatibility
                # with the early LTX23 build. Expand them to IMAGE frames.
                holds = _latent_holds_to_frame_holds(holds, n)
                alignment_note = "expanded explicit LTX latent hold map to image frames"
            elif len(holds) != n:
                # A hold map is fundamentally a source-frame map. If a
                # workflow crops/extends the IMAGE batch after the oracle,
                # keep the authored portion and leave newly uncovered frames
                # at native rate (hold=1) instead of crashing. This preserves
                # exact recovery for the map emitted by this smear node.
                original_len = len(holds)
                if original_len < n:
                    holds = holds + [1] * (n - original_len)
                    alignment_note = (
                        f"hold map covered {original_len} frames; appended "
                        f"{n - original_len} native-rate frames to match image batch"
                    )
                else:
                    holds = holds[:n]
                    alignment_note = (
                        f"hold map covered {original_len} frames; cropped it to "
                        f"the {n}-frame image batch"
                    )
        else:
            holds = [int(dilation)] * n

        if len(holds) != n:
            raise ValueError(f"hold map covers {len(holds)} frames after alignment, image batch has {n}")
        if any(int(h) < 1 for h in holds):
            raise ValueError("hold map entries must be positive integers")

        note = None
        if expand_to_end:
            holds, note = expand_hold_map_to_end(holds)

        target = _legal_ceil(sum(holds))
        n_held = sum(1 for h in holds if h > 1)
        holds = list(holds)
        holds[-1] += target - sum(holds)

        idx = torch.tensor(
            [i for i, h in enumerate(holds) for _ in range(int(h))],
            dtype=torch.long,
        )
        used = json.dumps({"holds": holds, "world_len": n, "units": "frames"})
        mode = (
            f"uniform x{dilation}"
            if not hold_map.strip()
            else (
                f"adaptive, {n_held} of {n} frames held"
                + (" (expanded from LTX latent map)" if map_units == "ltx_latent" else "")
            )
        )
        report = _cost_report(
            n, target, fps, s_per_step, est_steps, overhead_s, tail=mode
        )
        if note:
            report += "\n" + note
        if alignment_note:
            report += "\n" + alignment_note

        return (images[idx], used, int(target), report)


class LTX23ExactRecover:
    DESCRIPTION = (
        "LTX-2.3 adaptation of H3 Exact Recover. Keeps the first generated "
        "frame from every integer hold group, exactly inverting LTX23TimeSmear "
        "without interpolation or resampling."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "hold_map": ("STRING", {
                    "default": "",
                    "tooltip": "hold_map_used from the same LTX23TimeSmear node.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "recover"
    CATEGORY = "image/ltx23/motion"

    def recover(self, images, hold_map):
        if not hold_map.strip():
            raise ValueError("LTX23ExactRecover requires hold_map_used from LTX23TimeSmear")
        data = json.loads(hold_map)
        holds = [int(h) for h in data["holds"]]
        starts = []
        cur = 0
        for h in holds:
            if h < 1:
                raise ValueError("hold map entries must be positive integers")
            starts.append(cur)
            cur += h
        if cur != images.shape[0]:
            raise ValueError(
                f"hold map expands to {cur} image frames, but received {images.shape[0]}"
            )
        return (images[torch.tensor(starts, dtype=torch.long)].cpu(),)


def _torchaudio(feature):
    """Import torchaudio lazily with a helpful message (mirrors MAINodes)."""
    try:
        import torchaudio
        return torchaudio
    except Exception as e:
        raise ImportError(
            f"LTX23 audio nodes need torchaudio for the {feature} op "
            f"(pip install torchaudio): {e}"
        )


def _vocoder_rate_for(spec, rate, hop, need):
    """Clamp a phase-vocoder rate so the ISTFT can cover `need` samples.

    Same guard as the original MAINodes implementation: torch.istft(center=True)
    yields n_frames*hop valid samples, and phase_vocoder emits ceil(n_in/rate)
    frames. LTX audio latents decode on their own clock, so the last stretched
    run of a map can also end up with fewer source samples than its frame
    count implies; without this clamp the ISTFT trips "window overlap add min".
    """
    n_in = spec.shape[-1]
    if n_in <= 0 or need <= 0:
        return rate
    return min(float(rate), (n_in * hop) / float(need))


class LTX23AudioSmear:
    """Expand a world-clock track onto the dilated clock: Audio Recover run
    backwards, off the same hold map.

    LTX-2.3/2.5 adaptation of H3 Audio Smear.
    The audio twin of LTX23TimeSmear: stretches the baseline clip's audio
    onto the SAME dilated timeline the smeared video init lives on, so
    the second pass can be seeded with a slowed performance instead of
    inventing speech or having desynchronized audio-video tokens.
    """

    DESCRIPTION = (
        "The audio twin of LTX23TimeSmear: stretches the baseline clip's audio "
        "onto the SAME dilated timeline the smeared video init lives on, so "
        "the second pass can be seeded with a slowed performance instead of "
        "inventing speech at natural rate.\n\n"
        "Why this exists: LTX23TimeSmear tells pass 2 to move slowly through the "
        "smeared video init, and the picture obeys, but if audio has no init "
        "or is fed un-smeared world-rate audio, the video generation cross-attends "
        "to mismatched audio tokens and the model writes speech at natural rate. "
        "LTX23ExactRecover then compresses the video back down by the hold factor, "
        "causing held regions to come back rushed and breaking lip sync.\n\n"
        "Seeding pass 2's audio rows with this node's output ensures the slowed "
        "performance is what pass 2 renders. Exact Recover and Audio Recover are "
        "then both compressing something that really was slow, keeping lip sync intact.\n\n"
        "Wiring: Feed baseline (pass 1) audio and hold_map_used from LTX23TimeSmear. "
        "Send the smeared audio output into LTXVAudioVAEEncode -> LTXVConcatAVLatent for pass 2."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO", {"tooltip": "baseline (pass-1) audio, on the world clock"}),
            "hold_map": ("STRING", {
                "default": "",
                "tooltip": "hold_map_used from the same LTX23TimeSmear node (NOT the raw oracle map).",
            }),
            "fps": ("INT", {"default": 25, "min": 1, "max": 120,
                            "tooltip": "Must equal the frame rate of the LTX generation (e.g. 24 or 25)."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "smear"
    CATEGORY = "audio/ltx23/motion"

    def smear(self, audio, hold_map, fps=25):
        import math

        if not hold_map.strip():
            raise ValueError("LTX23AudioSmear requires hold_map_used from LTX23TimeSmear")

        try:
            fps = max(1, min(120, int(fps)))
        except (TypeError, ValueError):
            fps = 25

        torchaudio = _torchaudio("phase_vocoder")

        data = json.loads(hold_map)
        holds = [int(h) for h in data["holds"]]
        if not holds or any(h < 1 for h in holds):
            raise ValueError("hold map entries must be positive integers")

        wav = audio["waveform"].detach().float().cpu()
        sr = audio["sample_rate"]
        b, c, n = wav.shape
        x = wav.reshape(b * c, n)

        runs = []                                        # consecutive equal holds
        for h in holds:
            if runs and runs[-1][0] == h:
                runs[-1][1] += 1
            else:
                runs.append([h, 1])

        n_fft, hop = 2048, 512
        window = torch.hann_window(n_fft)
        phase_adv = torch.linspace(0, math.pi * hop, n_fft // 2 + 1)[..., None]
        spf = sr / float(fps)
        xfade = max(1, int(round(0.005 * sr)))
        segs, joins, cursor = [], [], 0.0
        prev_tgt = 0
        for h, count in runs:
            src = count * spf
            tgt = int(round(h * count * spf))
            s0, s1 = int(round(cursor)), int(round(cursor + src))
            cursor += src
            f = 0 if not segs else min(xfade, tgt, prev_tgt, s0 * max(h, 1))
            prev_tgt = tgt
            pre = max(1, f // max(h, 1)) if f else 0
            seg = x[:, max(0, s0 - pre):min(s1, n)] if f else x[:, s0:min(s1, n)]
            if seg.shape[1] == 0:
                segs.append(torch.zeros(x.shape[0], tgt))
                joins.append(0)
                continue
            if h > 1:
                spec = torch.stft(seg, n_fft, hop, window=window,
                                  return_complex=True)
                spec = torchaudio.functional.phase_vocoder(
                    spec, _vocoder_rate_for(spec, 1.0 / float(h), hop, f + tgt), phase_adv)
                seg = torch.istft(spec, n_fft, hop, window=window,
                                  length=f + tgt)
            if seg.shape[1] < f + tgt:
                seg = torch.nn.functional.pad(seg, (0, f + tgt - seg.shape[1]))
            segs.append(seg[:, :f + tgt])
            joins.append(f)
        parts = []
        for seg, f in zip(segs, joins):
            if f:
                t = torch.arange(1, f + 1, dtype=seg.dtype) / f
                prev = parts[-1].clone()
                prev[:, -f:] = (prev[:, -f:] * torch.cos(t * (math.pi / 2))
                                + seg[:, :f] * torch.sin(t * (math.pi / 2)))
                parts[-1] = prev
            parts.append(seg[:, f:])
        y = torch.cat(parts, dim=1)
        print(f"[LTX23AudioSmear] map: {len(holds)} world frames -> {sum(holds)} dilated frames "
              f"(avg x{sum(holds) / len(holds):.2f}); "
              f"audio: {n} -> {y.shape[-1]} samples @ {sr} Hz "
              f"({n / sr:.3f} s -> {y.shape[-1] / sr:.3f} s)")
        return ({"waveform": y.reshape(b, c, -1).contiguous(), "sample_rate": sr},)


class LTX23AudioRecover:
    """Retime the regenerated LTX clip's jointly-generated audio back to the
    original clock, using the same hold map as the video.

    LTX-2.3 adaptation of H3 Audio Recover. The retiming math is clock-
    agnostic (frames x samples-per-frame), so it carries over unchanged;
    what changes is the contract: hold_map here is the hold_map_used string
    emitted by LTX23TimeSmear (same JSON the smear snapped to the 8k+1
    grid), and the node makes no assumption about the 17k+5 grid.
    """

    DESCRIPTION = (
        "LTX-2.3 adaptation of H3 Audio Recover. Retimes the regenerated "
        "clip's own audio back to the original clock, using the same hold "
        "map as the video. Each hold segment is compressed with a phase "
        "vocoder, so pitch is preserved while duration shrinks. Wire audio "
        "from VAEDecodeAudio of the regenerated LTX latent and hold_map_used "
        "from the same LTX23TimeSmear that built the init; the result lines "
        "up with LTX23ExactRecover's video frame for frame. fps is the video "
        "frame rate the holds count in (LTX 2.3/2.5 renders are commonly "
        "25 or 50 fps - set it to whatever you generated at).\n\n"
        "Lipsync note: if pass 2's audio rows were NOT seeded, the model "
        "invented speech at natural rate inside the dilated timeline; this "
        "node then compresses it by the hold factor and held regions come "
        "back rushed. In that case blend the original clip's audio back in "
        "via reference + reference_mix=1.\n\n"
        "Thickness dial: the regenerated audio is scored for the slowed "
        "performance, so it comes back leaner than a native-speed mix. Wire "
        "the baseline clip's audio into reference and raise reference_mix to "
        "blend its full-speed track back in: 0 keeps the regenerated audio, "
        "1 is the baseline track alone."
    )

    SOURCES = {
        "keep the original performance (safe default)": 1.0,
        "use pass 2's audio - ONLY IF the audio rows were seeded": 0.0,
        "blend, favour the original": 0.75,
        "blend, favour pass 2": 0.25,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "hold_map": ("STRING", {
                "default": "",
                "tooltip": "hold_map_used from the same LTX23TimeSmear node (NOT the raw LTX23JerkOracle.hold_map - the smear aligns it to the image batch and snaps the tail to the legal 8k+1 grid, so the audio must retime off the SAME map the video used).",
            }),
            "fps_mode": ([
                "auto (detect audio clock)",
                "audio is dilated (retime by hold map)",
                "audio is world-rate (passthrough)",
                "manual fps (assume dilated)",
            ],
                {"default": "auto (detect audio clock)",
                 "tooltip": "auto measures the decoded audio against BOTH clock the map implies - the DILATED duration (sum(holds)/fps, what a regenerated AV latent should produce when pass 2's audio rows were stretched) and the WORLD duration (len(holds)/fps, what you get when pass 2's audio rows were seeded from pass 1's own world-rate audio, as LTXVConcatAVLatent does NOT stretch audio tokens) - and retimes only when the audio really is dilated. If the audio is already world-rate it is passed through untouched; compressing world-rate audio is what used to cut the sound short after a few seconds. The explicit modes force a behaviour; manual assumes a dilated clock at the fps given below."}),
            "fps": ("INT", {"default": 25, "min": 1, "max": 120,
                            "tooltip": "Used only in manual fps_mode. Must equal the frame rate of the LTX generation (25 for 25fps renders, 50 for 50fps renders)."}),
        }, "optional": {
            "reference": ("AUDIO", {"tooltip": "baseline clip audio (already real-time)"}),
            "reference_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                        "step": 0.05,
                                        "tooltip": "1 = the baseline track intact (default: regenerated "
                                                   "audio quality varies), 0 = regenerated audio only. "
                                                   "Mid values blend two takes and are happiest near the ends"}),
            "audio_source": (["custom (use reference_mix)"] + list(cls.SOURCES),
                {"default": "custom (use reference_mix)",
                 "tooltip": "plain-language presets; anything but 'custom' overrides reference_mix. "
                            "WHICH ONE IS RIGHT DEPENDS ON WHETHER PASS 2's AUDIO ROWS WERE SEEDED "
                            "(LTX23AudioSmear -> VAE Encode Audio -> audio latent init). Unseeded, "
                            "pass 2 invents speech at natural rate and this node compresses it, so "
                            "held regions come back rushed - keep the original. Seeded, pass 2 really "
                            "did perform slowly, so the retime is valid"}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "recover"
    CATEGORY = "audio/ltx23/motion"

    def recover(self, audio, hold_map, fps_mode="auto (detect audio clock)",
                fps=25, reference=None, reference_mix=1.0,
                audio_source="custom (use reference_mix)"):
        import math

        if not hold_map.strip():
            raise ValueError(
                "LTX23AudioRecover requires hold_map_used from LTX23TimeSmear"
            )
        if audio_source in self.SOURCES:          # words win over the raw dial
            reference_mix = self.SOURCES[audio_source]

        torchaudio = _torchaudio("phase_vocoder")

        data = json.loads(hold_map)
        holds = [int(h) for h in data["holds"]]
        if not holds or any(h < 1 for h in holds):
            raise ValueError("hold map entries must be positive integers")
        total_holds = sum(holds)

        wav = audio["waveform"].detach().float().cpu()   # [B, C, N]
        sr = audio["sample_rate"]
        b, c, n = wav.shape
        x = wav.reshape(b * c, n)

        # Clock discipline. The retime is only valid if the decoded audio
        # actually lives on the DILATED clock (sum(holds) frames): pass 2's
        # audio rows must have been stretched when the AV latent was built.
        # In pipelines where pass 2's audio rows come from pass 1's own
        # world-rate output (LTXVConcatAVLatent does NOT stretch audio
        # tokens), the decoded audio is already on the WORLD clock
        # (len(holds) frames) and compressing it by the hold factor cuts it
        # short - the historical "sound gone after a few seconds" failure.
        # So: measure the audio against both implied durations and decide.
        # Legacy guard: a workflow saved with an older node layout can hand
        # this method mangled widget values (e.g. a bare int landed on
        # fps_mode by positional mapping); anything unrecognized falls back
        # to the safe auto-detect.
        try:
            fps = max(1, min(120, int(fps)))
        except (TypeError, ValueError):
            fps = 25
        mode = fps_mode if isinstance(fps_mode, str) else ""
        world_samples = len(holds) * sr / float(fps)
        dilated_samples = total_holds * sr / float(fps)
        tol = max(0.02 * n, int(0.05 * sr))          # 2% of track or 50 ms
        fits_world = abs(n - world_samples) <= tol
        fits_dilated = abs(n - dilated_samples) <= tol

        if "world-rate" in mode:                     # explicit passthrough
            clock, retime = "forced world-rate (passthrough)", False
            if fits_dilated and not fits_world:
                print(f"[LTX23AudioRecover] WARNING: forced passthrough but "
                      f"the audio ({n / sr:.3f} s) matches the DILATED "
                      f"duration ({dilated_samples / sr:.3f} s), not the "
                      f"world duration ({world_samples / sr:.3f} s) - the "
                      f"output will lag the recovered video by the hold "
                      f"factor. Use auto or the dilated mode")
        elif "dilated" in mode and "manual" not in mode:   # force retime
            clock, retime = "forced dilated (retime)", True
            if fits_world and not fits_dilated:
                print(f"[LTX23AudioRecover] WARNING: forced retime but the "
                      f"audio ({n / sr:.3f} s) matches the WORLD duration "
                      f"({world_samples / sr:.3f} s), not the dilated one "
                      f"({dilated_samples / sr:.3f} s) - compressing "
                      f"world-rate audio cuts it short. Use auto or the "
                      f"passthrough mode")
        elif "manual" in mode:
            clock, retime = f"manual (assumed dilated @ fps={fps})", True
            spf = sr / float(fps)
        else:                                        # auto-detect
            if fits_world and fits_dilated:
                # avg hold ~1: the two clocks coincide, either is identity
                clock, retime = "auto: world==dilated (passthrough)", False
            elif fits_world:
                clock, retime = (
                    f"auto: audio is WORLD-rate ({n / sr:.3f} s ~ "
                    f"{world_samples / sr:.3f} s, dilated would be "
                    f"{dilated_samples / sr:.3f} s) - passthrough, no retime",
                    False)
                print(f"[LTX23AudioRecover] NOTE: audio duration ({n / sr:.3f} s) matches "
                      f"the WORLD clock, not the dilated clock ({dilated_samples / sr:.3f} s). "
                      f"If pass 2 lip sync doesn't match the video, pass 2 was rendered without smeared audio. "
                      f"To preserve lip sync, wire the baseline audio into LTX23AudioSmear -> LTXVAudioVAEEncode -> pass 2.")
            elif fits_dilated:
                clock, retime = (
                    f"auto: audio is DILATED ({n / sr:.3f} s ~ "
                    f"{dilated_samples / sr:.3f} s, world would be "
                    f"{world_samples / sr:.3f} s) - retime by hold map",
                    True)
            else:
                # neither fits: pick the closer clock and say so loudly
                d_world, d_dil = abs(n - world_samples), abs(n - dilated_samples)
                retime = d_dil < d_world
                clock = (f"auto: audio {n / sr:.3f} s matches NEITHER clock "
                         f"(world {world_samples / sr:.3f} s, dilated "
                         f"{dilated_samples / sr:.3f} s) - chose the closer "
                         f"({'dilated, retime' if retime else 'world, passthrough'})")
                print(f"[LTX23AudioRecover] WARNING: {clock}. Check that "
                      f"fps={fps} is the render rate and that this audio "
                      f"comes from VAEDecodeAudio of the REGENERATED latent")
        if retime and "manual" in mode:
            src_spf = sr / float(fps)      # manual: assume dilated at same fps
            tgt_spf = sr / float(fps)
        elif retime:
            src_spf = n / float(total_holds)  # audio's own dilated clock
            tgt_spf = world_samples / float(len(holds))  # world clock (sr/fps corrected)
        print(f"[LTX23AudioRecover] map: {len(holds)} world frames, "
              f"{total_holds} dilated frames (avg x{total_holds / len(holds):.2f}); "
              f"audio: {b * c} ch, {n} samples @ {sr} Hz ({n / sr:.3f} s); "
              f"clock: {clock}"
              + (f"; src_spf={src_spf:.3f} tgt_spf={tgt_spf:.3f}" if retime else ""))

        if not retime:
            # The audio is already on the world clock: hands off. Retiming
            # it would compress it by the hold factor and cut it short
            # (sound "gone" after a few seconds of a 15 s video).
            # NOTE: the decoded pass-2 audio can be a few hundred milliseconds
            # shorter than the reference because the latent VAE quantises length
            # to a fixed hop grid. When reference_mix=1.0 we are going to
            # return the reference anyway - let the reference determine the
            # output length so we do NOT silently truncate the last ~100 ms.
            y = x
        else:
            runs = []                                    # consecutive equal holds
            for h in holds:
                if runs and runs[-1][0] == h:
                    runs[-1][1] += 1
                else:
                    runs.append([h, 1])

            n_fft, hop = 2048, 512
            window = torch.hann_window(n_fft)
            phase_adv = torch.linspace(0, math.pi * hop, n_fft // 2 + 1)[..., None]
            # src_spf (dilated) is used to read from the source audio;
            # tgt_spf (world) is used to compute the output segment length.
            xfade = max(1, int(round(0.005 * sr)))       # 5 ms crossfade
            segs, joins, cursor = [], [], 0.0
            prev_tgt = 0
            for h, count in runs:
                src = h * count * src_spf     # dilated audio samples for this run
                # target: world-clock samples this run must occupy. Use the true
                # world rate (tgt_spf = sr/fps or world_samples/len(holds)) so
                # the retimed audio duration exactly matches the video.
                tgt = int(round(count * tgt_spf))
                s0, s1 = int(round(cursor)), int(round(cursor + src))
                cursor += src  # advance cursor in dilated source space
                # pre-roll for the crossfade into the previous run: f output
                # samples cost f*h source samples, taken from BEFORE s0, so both
                # sides of the join carry the same material at their own rate
                # and nothing is dropped. It lands ON TOP of the previous
                # segment's tail, so output length is exactly sum(tgt).
                f = 0 if not segs else min(xfade, tgt, prev_tgt, s0 // max(h, 1))
                prev_tgt = tgt
                seg = x[:, max(0, s0 - f * h):min(s1, x.shape[1])]
                if seg.shape[1] == 0:
                    # source exhausted: hold the clock with silence instead of
                    # silently shortening every later segment's position
                    segs.append(torch.zeros(x.shape[0], tgt))
                    joins.append(0)
                    continue
                if h > 1:
                    spec = torch.stft(seg, n_fft, hop, window=window,
                                      return_complex=True)
                    spec = torchaudio.functional.phase_vocoder(
                        spec, _vocoder_rate_for(spec, float(h), hop, f + tgt), phase_adv)
                    # length= is load-bearing: without it istft returns a
                    # hop-multiple that falls short of the target and the pad
                    # below appends digital silence at every held run's tail.
                    seg = torch.istft(spec, n_fft, hop, window=window,
                                      length=f + tgt)
                if seg.shape[1] < f + tgt:
                    seg = torch.nn.functional.pad(seg, (0, f + tgt - seg.shape[1]))
                segs.append(seg[:, :f + tgt])
                joins.append(f)
            parts = []
            for seg, f in zip(segs, joins):
                if f:
                    # equal-power crossfade: the two takes of this material are
                    # retimed at different rates, so they sum incoherently
                    t = torch.arange(1, f + 1, dtype=seg.dtype) / f
                    prev = parts[-1].clone()                 # never write into x
                    prev[:, -f:] = (prev[:, -f:] * torch.cos(t * (math.pi / 2))
                                    + seg[:, :f] * torch.sin(t * (math.pi / 2)))
                    parts[-1] = prev
                parts.append(seg[:, f:])
            y = torch.cat(parts, dim=1)
        if reference is None and reference_mix > 0:
            print(f"[LTX23AudioRecover] reference_mix={reference_mix} but no "
                  "reference audio is wired: the mix does NOTHING and the "
                  "output is pure regenerated audio. Wire the baseline's "
                  "audio into 'reference' to blend it back in")
        if reference is not None:
            ref = reference["waveform"].detach().float().cpu().reshape(
                -1, reference["waveform"].shape[-1])
            if reference["sample_rate"] != sr:
                _ta = _torchaudio("resample")
                ref = _ta.functional.resample(ref, reference["sample_rate"], sr)
            if ref.shape[0] != y.shape[0]:
                ref = ref[:1].expand(y.shape[0], -1)
            _sync_report(y, ref, sr, tag="pre-mix")
            if not retime and reference_mix >= 1.0:
                # Passthrough + full reference: return reference at its own
                # length so VAE-quantised audio does not silently truncate the
                # last ~100 ms of the clip.
                print(f"[LTX23AudioRecover] passthrough + reference_mix=1.0: "
                      f"returning reference ({ref.shape[1] / sr:.3f} s) "
                      f"instead of decoded audio ({y.shape[1] / sr:.3f} s)")
                y = ref
            else:
                # Mixed mode: align to the shorter of the two so we never
                # read past either buffer's end.
                n_out = min(y.shape[1], ref.shape[1])
                y = (1 - reference_mix) * y[:, :n_out] + reference_mix * ref[:, :n_out]
        return ({"waveform": y.reshape(b, c, -1).contiguous(), "sample_rate": sr},)


def _sync_report(y, ref, sr, tag=""):
    """Objective A/V-sync metric: how well does the retimed regenerated
    track line up with the original reference?

    Compares 20 ms loudness envelopes (waveform correlation is destroyed by
    phase differences between two takes; envelope shape survives). Reports
    Pearson correlation at zero lag and the best lag within +-500 ms. Read
    it as: corr >= 0.6 and |lag| <= ~40 ms  -> timing is as good as two
    different takes get; large |lag| -> systematic offset (retime or mux);
    corr near 0 even at best lag -> the two takes are different performances
    (pass 2 re-timed the show; no clock math can fix that, see the
    audio-smear seeding / pass-2 sigma discussion).
    """
    import torch.nn.functional as F

    hop = max(1, int(0.02 * sr))                 # 20 ms envelope frames
    e_y = F.avg_pool1d(y.abs(), kernel_size=hop, stride=hop).mean(dim=0)
    e_r = F.avg_pool1d(ref.abs(), kernel_size=hop, stride=hop).mean(dim=0)
    n = min(e_y.shape[-1], e_r.shape[-1])
    if n < 25:
        return
    e_y, e_r = e_y[:n] - e_y[:n].mean(), e_r[:n] - e_r[:n].mean()
    std = (e_y.std() * e_r.std()) or 1e-9
    corr0 = float((e_y * e_r).mean() / std)
    span = min(25, n // 4)                       # +-500 ms in 20 ms steps
    e_y_c = e_y - e_y.mean()
    best_lag, best_corr = 0, corr0
    for lag in range(1, span):
        if lag >= n // 2:
            break
        a = e_y_c[lag:]
        b = e_r[:n - lag] - e_r[:n - lag].mean()
        c = float((a * b).mean() / ((a.std() * b.std()) or 1e-9))
        if c > best_corr:
            best_corr, best_lag = c, lag
    # negative lag = the recovered track leads the reference (it starts
    # early); positive = it lags (it starts late).
    sign = -1 if best_corr <= corr0 else 1
    print(f"[LTX23AudioRecover] sync check {tag}: envelope corr at 0 lag = "
          f"{corr0:.3f}; best corr = {best_corr:.3f} at "
          f"{sign * best_lag * 0.02:+.2f} s "
          f"({'recovered track is EARLY' if sign * best_lag * 0.02 < 0 else 'recovered track is LATE' if sign * best_lag * 0.02 > 0 else 'no offset'})"
          f"{'  -> GOOD sync' if best_corr >= 0.6 and abs(best_lag * 0.02) <= 0.06 else '  -> POOR sync: see tooltip/sync notes' if best_corr < 0.6 else ''}")



NODE_CLASS_MAPPINGS = {
    "LTX23JerkOracle": LTX23JerkOracle,
    "LTX23TimeSmear": LTX23TimeSmear,
    "LTX23ExactRecover": LTX23ExactRecover,
    "LTX23AudioSmear": LTX23AudioSmear,
    "LTX23AudioRecover": LTX23AudioRecover,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX23JerkOracle": "LTX 2.3 Jerk Oracle",
    "LTX23TimeSmear": "LTX 2.3 Time Smear",
    "LTX23ExactRecover": "LTX 2.3 Exact Recover",
    "LTX23AudioSmear": "LTX 2.3 Audio Smear",
    "LTX23AudioRecover": "LTX 2.3 Audio Recover",
}
