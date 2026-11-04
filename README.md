# minecraft-montage

Semi-automatic pipeline: Minecraft replay → camera path → rendered video →
music + titles → YouTube.

One manual click per video (the render), everything either side automated.

## The desktop app

Double-click **`Montage.cmd`** (or the desktop shortcut). Three tabs:

1. **Prepare replay** — pick a `.mcpr`, choose which part of it to use, set
   the shot, write the camera path. Or use **Or do a whole session at once**
   at the bottom to prepare every clip from one play session in one go, using
   the same camera settings — with a Preview button that shows exactly what it
   would do before writing anything.
   **Minecraft must be closed** for this: ReplayMod holds recordings open, and
   the write fails with `WinError 5` otherwise. The app checks and tells you
   rather than failing obscurely. (The write is temp-file-then-rename, so a
   failure never damages the recording.)
2. **Finish video** — build a sequence from one or more rendered clips, then
   add music and a title. Add clips with **Add ↓**, reorder with **▲/▼**, drop
   one with **✕**, and check any of them with **▶ Preview**. The summary line
   shows the finished length, crossfades already deducted. Safe to run any
   time.
3. **Upload (private)** — preview the video, check the description, upload. It
   lands private and stays private; a button then opens YouTube Studio if you
   want to publish it.

Music takes **several tracks**, not just one: add them in the Music box in tab 2
and they share the video's runtime equally, crossfading into each other. Every
track's credit is appended to the description automatically.

The render sits between tabs 1 and 2 and happens inside Minecraft: Replay
Viewer → open the replay → Render Queue → Render.

Use `Montage (debug).cmd` if it ever fails to start — same app, but with a
console so you can see why.

Uploads are private-only and privacy is not a control on the form: an
unverified API project cannot publish publicly anyway, so offering the choice
would be a lie. Publishing stays a deliberate act in YouTube Studio.

## Why one click

ReplayMod-style rendering **cannot** be made headless. Verified against
`reforgedplaymod-1.21.1-0.3.jar` (sha256 `6084816…5cc0`), not from docs:

- Rendering captures live OpenGL frames (`OpenGlFrameCapturer`,
  `PboOpenGlFrameCapturer`, `ForceChunkLoadingHook`). It needs a running game
  with a GL context.
- The only `main()` entry points in the jar are `render.blend.ToJson` and
  ReplayStudio's `Launcher`. The `Launcher` CLI does packet manipulation only —
  split, squash, append, reverse, filter. It cannot render video.
- The entire mod reads exactly three system properties:
  `replaymod.skipversioncheck`, `fml.noGrab`, `replaystudio.port`. No render
  hook, no chat command.

**But** the render queue is data on disk, so the expensive part — placing camera
keyframes by hand — is automatable. `RenderJob.readQueue`/`writeQueue` persist it
as a `renderQueue.json` entry *inside the `.mcpr`*, and
`ReplayModRender.onReplayOpened` loads it automatically when you open the replay.

So: generate the path here → open the replay in game → **Render Queue → Render**.

## The `.mcpr` render-queue schema

A `.mcpr` is a plain zip. `renderQueue.json` is a Gson `List<RenderJob>`:

```json
[ { "timeline": "<escaped JSON string>", "settings": { … } } ]
```

`timeline` is a **nested JSON string** (a custom `TimelineTypeAdapter` delegates
to `TimelineSerialization`), keyed by the empty string:

```json
{ "": [ <TIME path>, <POSITION path> ] }
```

Path order is fixed by the `SPTimeline$SPPath` enum: `TIME` = 0, `POSITION` = 1.
Each path is:

```json
{ "keyframes":     [ {"time": <ms on the video timeline>, "properties": { … }} ],
  "segments":      [ <index into interpolators> | null, … ],
  "interpolators": [ {"type": <see below>, "properties": ["<group>:<id>"]} ] }
```

Property ids and value encodings:

| property id       | value                | notes |
|-------------------|----------------------|-------|
| `timestamp`       | `int`                | ms into the replay |
| `camera:position` | `[x, y, z]`          | doubles |
| `camera:rotation` | `[yaw, pitch, roll]` | floats, yaw **not** wrapped to ±180 |
| `spectate`        | `int`                | network entity id — see below |

Three things here are easy to get wrong:

1. **A property is prefixed `group:` only if it belongs to a `PropertyGroup`.**
   `CameraProperties` is a group, so its members are `camera:position` /
   `camera:rotation`. `TimestampProperty` and `SpectatorProperty` are standalone,
   so they are bare `timestamp` and `spectate` — *not* `timestamp:timestamp`.
   This one was got wrong from bytecode alone and only caught by diffing against
   a real file; the mod silently rejects the wrong key.
2. **The interpolator `"type"` is polymorphic.** `"linear"` and `"cubic-spline"`
   serialise as plain strings, but catmull-rom serialises as an *object*:
   `{"type":"catmull-rom-spline","alpha":0.5}`.
3. **`RenderSettings` has a public no-arg constructor**, so Gson leaves any field
   absent from the JSON at its built-in default. Only overrides need emitting.
   `outputFile` is a plain path string (`@JsonAdapter FileTypeAdapter`), *not* a
   reflected `File` object.

### Following the player

`--follow` orbits you *while you move*, which is usually what you want — a fixed
orbit centre is useless if you wander around the build.

It does not use `spectate` keyframes. `SpectatorProperty` is
`AbstractProperty<Integer>` keyed on the **network entity id**, a runtime value.
Instead the player's position is extracted ahead of time and baked into ordinary
position keyframes, so the render needs no entity id at all — the id only
matters at plan time.

The extraction runs ReplayStudio's own `EntityPositionTracker` (the class the mod
uses to make spectator keyframes work) out of the mod jar, via
`tools/DumpTracks.java`, rather than reimplementing packet parsing. Two things
that costs:

- **ReplayStudio is not self-contained.** gson, commons-lang3 and guava come from
  Minecraft's runtime, and opennbt and friends are jarjar'd inside the mod at
  `META-INF/jars/`. `tools/make_cp.py` gathers all three sources.
- **`EntityPositionTracker` writes a cache entry back into the replay it is
  given**, so `tracks.dump()` always hands it a temp copy. Your recording is
  never passed to it.

Build it once:

```bash
javac -cp "<profile>/mods/reforgedplaymod-1.21.1-0.3.jar" -d build tools/DumpTracks.java
python tools/make_cp.py
```

The player is auto-detected: its position is updated far more often than anything
else, so sample count dominates, with movement and an early start as tie-breaks.
On the test replay that picked entity 47020 with 2501 samples against 726 for the
next candidate. Check it with `mcmontage tracks`, override with `--follow-entity`.
A busy server with several players would make this ambiguous.

**Smoothing matters.** Raw player position judders — strafing, jumping, stepping
round a build — and a camera aimed straight at it is unwatchable. `--smooth`
applies a centred moving average, which also makes the camera trail slightly
behind, reading as an operator following you rather than a rig welded to your
back.

**The window is chosen by measurement, not by a formula.** Smoothing also cuts
corners: the smoothed point lags wherever you actually were, and how much lag a
given window causes depends entirely on how fast you were moving. Measured on an
11-minute session covering 5157 blocks:

| window | worst drift from your real position |
|--------|-------------------------------------|
| 1000 ms | 1.8 blocks |
| 2000 ms | 3.7 blocks |
| 4000 ms | 35.2 blocks |
| 8000 ms | 71.3 blocks |
| 25000 ms | 97.8 blocks |

With a 12-block orbit radius, anything past ~24 blocks means you are out of
shot. So `choose_smoothing()` tries progressively smaller windows and takes the
first whose worst-case drift fits a budget of 35% of the orbit radius.

Two heuristics were tried and discarded: a **fixed** window (fine on a slow
build, catastrophic on a fast one) and a window **scaled by the timelapse
speed-up** (worse — it grows exactly when the subject is covering the most
ground, which is precisely when lag hurts). Jitter is a sub-second physical
thing; it does not scale with playback rate. The keyframe sampling itself
already low-passes the path, so only a little extra smoothing is ever needed.

`--smooth` overrides in replay milliseconds, and warns if the drift would push
you out of frame.

`--revolutions` defaults to 1.0 but is worth dropping to ~0.5 when following: a
full spin on top of your own movement reads as restless.

### Schema validation status

**Timeline format: validated** (`replay.TIMELINE_SCHEMA_VALIDATED`). Diffed
field-for-field against a `timelines.json` the mod itself wrote, on 2026-09-08.
That file comes from the same `TimelineSerialization.serialize()` call that
produces the nested timeline string in `renderQueue.json`, so the hard half is
confirmed. It caught the `timestamp` prefix error above.

**Render-queue wrapper: validated** (`replay.RENDER_QUEUE_SCHEMA_VALIDATED`).
Confirmed on 2026-09-08 by a completed render: the mod consumed a
`renderQueue.json` written by this code and produced exactly the requested
output — 25.00 s at 1920×1080 and 60 fps, matching the timeline duration and the
`videoWidth` / `videoHeight` / `framesPerSecond` fields. A malformed envelope or
wrong field names could not have produced a to-spec render.

## Stitching clips

Several clips are joined in order, each pair overlapping by a crossfade, so the
finished runtime is `sum(durations) - (n-1) * xfade`. Clips are normalised to
the first one's resolution and frame rate first, so renders made at different
settings still join cleanly. One long build across several sessions therefore
becomes one video: render each replay separately, then stack them in tab 2.

## Timelapse without dropping frames

The TIME path maps video time onto replay time. Two keyframes spanning 30 min of
replay across 30 s of video give a ×60 speed-up — and it happens *in engine*, so
every rendered frame is a real frame rather than a survivor of frame-dropping.

### Picking a range

Untick **Use the whole replay** to take a slice, in `m:ss` or `h:mm:ss`
(`--start` / `--end` in milliseconds on the CLI). This matters more than it
sounds on long sessions: a 73-minute recording squeezed into 20 s is a ×220
timelapse that is too frantic to follow anyone through, whereas the five minutes
where you actually built something is ×15 and watchable.

The summary line under the fields shows the replay length, the slice, and the
resulting speed-up as you type, and refuses a range that runs past the end of
the replay or ends before it starts.

## Batching a whole session

The batch panel at the bottom of tab 1 does this from the app. On the CLI,
`mcmontage sessions` groups recordings into play sessions and `mcmontage batch`
renames and prepares a whole session in one pass:

```bash
python -m mcmontage sessions
python -m mcmontage batch --seconds 18 --max-speedup 30 --rename --dry-run
python -m mcmontage batch --seconds 18 --max-speedup 30 --rename
```

Then render each replay in game and stitch them in tab 2.

**The day boundary is not midnight.** A play-day runs 04:00 to 04:00
(`--cutoff`), so a session running 22:00 → 01:30 stays one sitting instead of
being split in two and numbered wrongly. `--max-gap` splits a day further when
there is a long quiet stretch, for days you play twice.

Grouping uses the recording time from each replay's own `metaData.json`, never
file mtime — writing a render queue rewrites the archive and bumps mtime, so
mtime stops reflecting when you actually played.

**`--max-speedup` is what makes batching usable on long sessions.** Whole
replays at a fixed clip length give wildly different rates — 21, 33, 38 and 73
minutes at 18 s each is ×71, ×109, ×126 and ×244. Capping at ×30 takes a centred
slice of each instead, so every clip runs at the same watchable rate. Centred,
because the start of a recording is usually spawning in and the end is usually
logging out.

Renaming happens **before** preparing, never after — see the trap below.

## A trap: rename before you prepare, not after

The output filename is baked into `renderQueue.json` when you prepare, derived
from the replay's name at that moment. Rename the `.mcpr` afterwards and the
render still writes the *old* name — so renders land under wrong names and can
silently overwrite each other. Rename first, then prepare.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
cp music/tracks.example.toml music/tracks.toml   # then add your tracks
```

Requires `ffmpeg`/`ffprobe` on PATH (the mod needs FFmpeg too).

## Usage

```bash
python -m mcmontage status                     # what is set up, what is not
python -m mcmontage inspect                    # newest replay: entries + render queue
python -m mcmontage tracks                     # list entities, spot the player
python -m mcmontage plan --follow --radius 12 --height 6 --revolutions 0.5 --keyframes 60
python -m mcmontage plan -x 100 -y 70 -z -40 --radius 28   # or a fixed centre
python -m mcmontage edit clip1.mp4 clip2.mp4 clip3.mp4 --xfade 1.0 --title "Base Build" --music atlantean_twilight
python -m mcmontage upload out/montage.mp4 --title "…" --music atlantean_twilight
```

`plan` writes into the `.mcpr` and keeps a `.bak` beside it.

## Music licensing

"Non-copyright" is not public domain. Every track in `music/tracks.toml` carries
a licence record, and the credit is injected into the video description
automatically — a CC-BY track cannot be used without one. A track with an
unrecognised licence is refused unless `attribution_required` is set explicitly,
so nothing can silently ship uncredited.

Safest first: **YouTube Audio Library** → Pixabay / CC0 → Incompetech (CC-BY,
credit mandatory).

### Several tracks per video

Tracks share the runtime equally and crossfade into each other, and **each is
loudness-normalised first**. That is not optional polish: of the four bundled
tracks, Airship Serenity sits at −13.5 dB and Atlantean Twilight at −34 dB — a
21 dB spread. Sequenced raw, the bed lurches from inaudible to blaring at every
join. Normalising to a common target narrowed the measured spread across a
three-track bed from 14.4 dB to 4.9 dB. A single track skips it, since its level
is just the gain.

Currently stocked: four Kevin MacLeod tracks (`atlantean_twilight`,
`bathed_in_the_light`, `airship_serenity`, `arcadia`) from the Internet
Archive's Incompetech collection, CC-BY-3.0. Note the version: incompetech.com
now issues most tracks under 4.0, but that archive item states 3.0, so 3.0 is
what is recorded — credit the licence you actually received the file under.

Pixabay was the first choice but returns **403 to scripted access**, same trap
as CurseForge. Its files have to be fetched by hand through a browser.

## YouTube: uploads land private, deliberately

Videos uploaded through an API project that has not passed a Google API audit are
permanently locked to private, and per Google's policy page that lock **cannot be
appealed**. So `youtube.py` sets `privacyStatus=private` and treats publishing as
a human step in YouTube Studio — which doubles as a review gate.

Quota, checked against Google's quota page on 2026-09-08: `videos.insert` costs
**1 unit** and has its **own bucket of 100 calls/day**, separate from the
10 000-unit pool. The widely repeated "1600 units per upload" figure is stale.
Quota will not be the constraint.

The mod also ships its own YouTube uploader with **ReplayMod's own OAuth client
id and secret embedded in the jar**, against a 2016-era API client. This pipeline
does not touch it: those are a third party's credentials. The in-game button is
there if you want to click it yourself.

## Layout

```
mcmontage/gui.py       the desktop app (Montage.cmd launches it)
mcmontage/pipeline.py  plan_replay() / finish_video() -- shared by GUI and CLI
mcmontage/tracks.py    entity position tracks, player detection, smoothing
mcmontage/replay.py    .mcpr read/write, render-queue round-trip
mcmontage/timeline.py  camera path + render-settings generation
mcmontage/edit.py      ffmpeg assembly: crossfades, titles, music bed
mcmontage/music.py     licence records and automatic attribution
mcmontage/youtube.py   OAuth + resumable upload (private)
mcmontage/cli.py       command line
```

## Thumbnails, generated metadata, private publishing

Three CLI commands (and the modules behind them) cover the last mile:

```
python -m mcmontage thumbnail out/montage.mp4 --title "Nether hub" --subtitle "Day 3" --template bold
python -m mcmontage meta --theme "Nether hub" --series "Survival Diaries" --day-index 3 --clips a.mp4 b.mp4 c.mp4
python -m mcmontage publish out/montage.mp4 --thumbnail out/thumbnail.png --clips a.mp4 b.mp4 c.mp4   # dry run
python -m mcmontage publish ... --yes                                                                # uploads, PRIVATE
```

- `mcmontage/thumbnail.py` samples frames with ffmpeg and scores them on
  sharpness (Laplacian variance) and contrast, penalising near-black/blown-out
  frames. Templates: `bold`, `minimal`, `banner`, `poster`. Output is 1280x720
  and kept under YouTube's 2 MB limit. Needs Pillow only.
- `mcmontage/metadata.py` builds title, tags, description and chapters.
  Chapter times account for crossfades and follow YouTube's rules (start at
  0:00, 3+ chapters, 10 s minimum); too-short clips fold into their neighbour.
- `mcmontage/upload_meta.py` is the private-by-default layer. A plan is
  private unless `allow_non_private=True` is set explicitly (nothing in the
  package does), the final request body is re-checked before sending, and
  credential-looking text or `secrets/` paths are refused. The network sits
  behind an `Uploader` interface; tests use `FakeUploader`.

Tests: `python -m pytest` (needs `pytest`, `Pillow`; ffmpeg tests skip without it).

## Tests

```bash
python -m pip install pytest
python -m pytest
```

No Minecraft, Java or network needed. `tests/synthetic.py` builds fake `.mcpr`
files (a zip with `metaData.json`, filler `recording.tmcpr`, optional
`renderQueue.json` / `timelines.json`), fake entity-track JSON, and — if
`ffmpeg` is on `PATH` — tiny test clips and tones. Tests that need ffmpeg are
marked `ffmpeg` and skip when it is missing.

Generated output is checked against golden files in `tests/golden/`: the
`renderQueue.json` for static and follow plans, the nested timeline JSON, the
full ffmpeg argument list for several clip/music/title combinations, and CLI
output. If generated output changes on purpose, review and refresh them with:

```bash
UPDATE_GOLDEN=1 python -m pytest && git diff tests/golden
```

The goldens pin *this code's* output; they are not proof the mod accepts it.
The schema was validated against the real mod on 2026-09-08 (see above), and a
golden changing is a prompt to re-check that, not to just regenerate.
