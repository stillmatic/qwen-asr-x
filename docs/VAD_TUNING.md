# VAD Tuning & Debug Viewer

The pipeline uses [Silero VAD](https://github.com/snakers4/silero-vad) to detect speech segments before sending audio to the ASR model. Tuning the VAD parameters affects which parts of the audio are transcribed -- a threshold that's too high will miss quiet speech, and one that's too low will send noise/music to the ASR model.

## VAD Parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| Threshold | `--vad-threshold` | `0.2` | Speech probability threshold (0.0-1.0). Higher = stricter. |
| Min speech duration | `--min-speech-duration-ms` | `250` | Discard speech segments shorter than this (filters noise bursts). |
| Min silence duration | `--min-silence-duration-ms` | `200` | Silence gap required to split speech segments. |
| Speech padding | `--speech-pad-ms` | `100` | Padding added before and after each detected segment. |
| Max speech duration | (code only) | `30.0s` | Force-split segments longer than this. |
| Merge gap | (code only) | `0.3s` | Merge segments separated by less than this gap. |

### Auto threshold mode

When `--vad-threshold` is set to `0` (or omitted from the server API with `null`), the pipeline enters **auto mode**. Instead of using fixed parameters, it:

1. Runs Silero VAD to get per-window speech probabilities for the entire file
2. Analyzes the probability distribution to find the noise floor (silence cluster)
3. Sets the threshold just above the silence noise floor (silence mean + 3 sigma, capped at 0.5)
4. Derives `min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms`, and `merge_gap_s` from the detected speech/silence patterns using Otsu's method

The auto-derived parameters are logged to stderr:

```
[VAD] auto params -- threshold: 0.108, min_speech: 50ms, min_silence: 100ms, pad: 80ms, merge_gap: 0.15s
```

Auto mode works well for most audio. Use a fixed threshold when you need deterministic behavior or when auto mode doesn't handle a specific recording well.

## VAD Debug Viewer

The debug viewer lets you visualize the VAD probability curve, see detected speech segments, and interactively adjust parameters to find the right settings for your audio.

### Step 1: Generate VAD debug data

Add `--visualize-vad` to your transcription command:

```bash
# CLI
uv run python main.py audio.mp3 -o output.json --visualize-vad

# Server mode -- add --visualize-vad when starting the server
uv run python server.py --visualize-vad
```

This saves a `vad.json` file to `examples/vad/<audio-stem>/` containing:
- Per-window speech probabilities from Silero VAD
- The VAD parameters that were used
- The detected speech segments
- Audio duration and metadata

### Step 2: Launch the viewer

```bash
uv run python vad_viewer.py
# or on a custom port:
uv run python vad_viewer.py --port 8888
```

Open `http://localhost:8899` in a browser.

### What the viewer shows

Each audio file gets a card with:

- **Probability curve** (blue line) -- Silero VAD's speech probability for each 32ms window
- **Detected speech segments** (orange shaded regions) -- segments that pass the current threshold and duration filters
- **Threshold line** (red dashed) -- the current speech probability threshold
- **Stats bar** -- total duration, number of segments, total speech time, and current view range

### Interactive controls

| Control | Range | What it does |
|---------|-------|-------------|
| **Threshold** | 0.05 - 0.99 | Speech probability cutoff. Frames above this are considered speech. |
| **Min speech** | 0 - 2000ms | Minimum segment duration. Shorter segments are discarded. |
| **Min silence** | 0 - 2000ms | Minimum silence gap to split segments. |
| **Pad** | 0 - 500ms | Padding added around each segment edge. |
| **Merge gap** | 0 - 2s | Segments closer than this are merged into one. |

Changes are applied immediately -- the segments and stats update in real time as you drag the sliders.

### Navigation

- **Scroll wheel** -- zoom in/out (centered on cursor position)
- **Click and drag** -- pan left/right
- **Double-click** -- reset to full view
- **Hover** -- tooltip shows exact time and probability value

### Workflow for tuning thresholds

1. Run a transcription with `--visualize-vad` and the default threshold
2. Open the viewer and look at the probability curve
3. If the transcription is **missing speech** (words cut off): lower the threshold slider until the orange regions cover all spoken parts
4. If the transcription has **extra noise** (non-speech getting transcribed): raise the threshold until the orange regions only cover actual speech
5. Adjust **min speech** to filter out short noise bursts that still sneak through
6. Adjust **min silence** to control whether pauses within a sentence split the segment or not
7. Once you find good values, pass them as CLI flags:

```bash
uv run python main.py audio.mp3 -o output.json \
  --vad-threshold 0.15 \
  --min-speech-duration-ms 200 \
  --min-silence-duration-ms 150 \
  --speech-pad-ms 80
```

### Common scenarios

**Noisy recording (background music, crowd noise):**
Raise threshold to 0.3-0.5. Increase min-speech-duration to 300-500ms to filter short noise bursts.

**Quiet speaker / low volume:**
Lower threshold to 0.1-0.15. The auto mode often handles this well.

**Interview with pauses:**
Lower min-silence-duration to avoid splitting mid-sentence pauses. Increase merge-gap to join segments that belong to the same utterance.

**Rapid back-and-forth dialogue:**
Lower min-silence-duration (100-150ms) so speaker changes create separate segments. Keep merge-gap small (0.1-0.2s).
