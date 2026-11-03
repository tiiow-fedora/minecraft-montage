# Overhaul: automatic montages

The original flow is manual: pick a replay, pick a span, render, stitch. This
adds an automatic path alongside it. Nothing in the old flow changed.

```
replay ─▶ tracks (+ events) ─▶ highlights ─▶ select ─▶ camera shots ─▶ renderQueue.json
                                                                           │ render in Minecraft (one click)
queue ◀── plan → render_wait → edit → upload ──────────────────────────────┘
edit  = clips + transitions + beat-synced cuts + title/credits cards + music
```

| Module | Job |
|---|---|
| `events.py` | Shared `Event` / `Highlight` types and the optional event sidecar |
| `highlights.py` | Detect deaths, kills, rapid movement, falls and builds |
| `campath.py` | Easing, shot builders, slow-motion ramps, render jobs |
| `transitions.py` | ffmpeg xfade filter-graph builder |
| `titlecards.py` | Title cards, credits cards, lower thirds |
| `beatsync.py` | Pure-Python beat tracking and beat-aligned cut planning |
| `jobqueue.py` | Persistent batch queue with resume |
| `queue_handlers.py` | Real handlers for the queue's job kinds |
| `auto.py` | Glue between the above, used by the CLI |

## What the detectors can and cannot see

Highlights come from entity *tracks* (positions over time, extracted by
`tools/DumpTracks.java`). Positions alone do not say what happened, so:

- **Deaths** are *observed* when an event sidecar has `death` events. Without
  one they are *inferred*: a teleport of 30+ blocks after a time gap or a long
  drop. That cannot tell a respawn from a lag spike followed by `/tp`.
- **Kills** need `kill` events. Tracks alone cannot show them, so with no sidecar
  there are none.
- **Rapid movement and falls** come from tracks only (sprinting, elytra-like
  speed, long drops). Teleports are excluded so they cannot fake speed.
- **Builds** come from `block_place` events when present. Otherwise a heuristic
  looks for the player staying put while making many small movements, labelled
  "Possible build". It can fire on fishing or AFK spinning and misses builds done
  while walking round a structure.

The event sidecar is `work/<replay>.events.json`, schema in `events.py`.
**Nothing writes it yet**: extending `DumpTracks.java` to emit death, kill and
block events is the natural next step and is not done here.

## Commands

```bash
python -m mcmontage highlights [REPLAY] [--out h.json]
python -m mcmontage autoplan   [REPLAY] --total 60 --seconds-each 8 --dry-run
python -m mcmontage assemble clip1.mp4 clip2.mp4 --music TRACK_ID --style dynamic
python -m mcmontage queue enqueue --seconds 20 --dry-run
python -m mcmontage queue run
python -m mcmontage queue status
```

`autoplan` writes one render job per chosen highlight, each with its own shot
(death: spiral-in orbit or crane with slow motion; kill: flyby or orbit with
slow motion; speed: follow or flyby; build: wide orbit or crane up; fall: crane
down or dolly). Variation is seeded from the highlight, so reruns give the same
shots. As before, Minecraft must be closed while the queue is written, and you
click Render once.

ReplayMod only has linear, cubic and catmull-rom interpolators, so easing and
speed ramps are baked into keyframe spacing rather than expressed as curves.

## Batch queue

`queue enqueue` builds a plan → render → edit → (upload) chain per play session.
The render step cannot be automated, so it parks as *waiting* until the rendered
file exists and has stopped growing; run `queue run` again after rendering.

- State lives in `work/queue.json`, written atomically. A runner that crashed
  leaves jobs `running`; the next run resets them to pending (attempts stay
  counted, so a job that kills its runner cannot loop forever).
- Keys derive from clip path, mtime, size and settings, so re-running enqueue does
  not duplicate work, and changing a setting creates a fresh chain.
- A runner lock and a short transaction lock allow `add`/`cancel` from another
  process while a runner is busy. Locks do not use `fcntl`.
- **Secrets are refused.** Params with secret-like names or token-shaped values
  raise an error, and results are redacted before they are saved. The upload
  handler loads credentials from `secrets/` itself (git-ignored).
- Uploads are private only, as in the rest of the app.

Not verified: the Windows-specific parts (pid liveness via ctypes, lock and
replace behaviour) have not been run on Windows.

## Beat sync

`beatsync.py` decodes audio through ffmpeg and finds tempo and beats with no
third-party libraries (onset envelope, autocorrelation, dynamic-programming beat
tracking). On synthetic click and kick tracks it is within 0.1% of tempo at
90/120/140 BPM. Caveats:

- Tempo is assumed roughly constant.
- Near the 60/180 BPM limits it may report an octave of the felt tempo; pass
  `min_bpm`/`max_bpm` to force one.
- Downbeats are a 4/4 accent heuristic.
- It is untested on real dense mixes with my own ears; judge the first real
  result yourself. `--no-sync` turns it off.
- MP3 decoder delay can shift beats by tens of milliseconds.

Analysis is cached next to the music file, keyed on size and mtime.

## Transitions and cards

All 58 xfade transitions in ffmpeg 6.1.1 render (checked). Older ffmpeg builds
may lack some: `transitions.available_transitions()` asks the installed build.
Cards are generated with no assets; a font is found from `$MCMONTAGE_FONT`, then
common Windows/macOS/Linux locations. The "typewriter" animation is a left-to-right
wipe, not letter by letter.

## Tests

```bash
pip install pytest
python -m pytest -q
```

Integration tests use ffmpeg on tiny synthetic clips and skip when ffmpeg, a font
or `drawtext` is missing. No test needs a real replay, the mod jar or network.
