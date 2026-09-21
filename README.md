# MLOps Cat Demo

Three fixed RTSP cameras watch a living room. A YOLO model detects **which** cat
is in frame, zone polygons drawn on each camera's reference frame turn that into
a room location, and the result is recorded to SQLite around the clock. A
scheduled job turns the finished day into a short narrative and posts it to
Discord. Ask in the channel what happened today and the bot answers there too.

## Project Goal

- Detect `bagel` and `kurumi` by identity. This is **detection**, not
  classification: the model emits a box per cat and the class index is the name.
- Resolve each detection to a named zone (`on_sofa`, `table_top`, `toilet_1`, …).
  The zone anchor is the **bottom-centre of the box**, which is the surface the
  cat stands on, so "on the table" and "under the table" stay separable.
- Survive camera drift. Zones are stored in the coordinate system of a per-camera
  **reference frame**; every sample is aligned back to it with ORB features, so a
  nudged camera does not silently relabel the room.
- Record dwell visits 24/7 and report the day.
- Answer questions about the day, asked in Discord, from the same data.

## Pipeline

```
RTSP cameras
   │  collect_identity_detection_from_rtsp.py   candidate frames + auto labels
   ▼
dataset/identity_detection_candidates
   │  annotate in CVAT
   ▼
dataset/2 … dataset/5                          one directory per annotation round
   │  create_cvat_yolo_archive.py / split_detection_dataset.py
   ▼
dataset/data.yaml                              the round training reads
   │  train_cat_identity_detector.py
   ▼
models/cat_identity_detection_yolo26n_*/weights/best.pt
   │
   ├── web_preview_app.py    draw zones on a live frame (browser)
   ├── recalibration.py      re-anchor zones after the camera moved
   ▼
location_tracker.py ──► data/location_history.db
   │                          │
   ├── realtime_view.py       │  where is the cat, right now
   ├── discord_bot.py ────────┤  "报告一下今天两只猫都做什么了" (inbound)
   └── location_report.py ────┘  what happened today ──► Discord (webhook)
```

## Repository Layout

| Path | What it is |
| --- | --- |
| `src/data/` | Dataset collection and preparation |
| `src/training/` | YOLO detection training |
| `src/monitoring/` | Zones, alignment, tracking, reporting, browser UI |
| `src/notification/` | Discord webhook posting |
| `configs/config.yaml` | Cameras, model paths, training defaults |
| `configs/locations.yaml` | Tracking loop, alignment, preview, report, Discord bot |
| `configs/zones.yaml` | Zone polygons, written by the tools |
| `deploy/systemd/` | Units for the 24/7 tracker, the daily report and the Discord bot |
| `scripts/` | Shell wrappers and a demo-day generator |

### Entry points

Every module is reachable through a thin `src/execute_*.py` wrapper that fixes
`sys.path` and calls the module's `main()`. Run them from the project root.

| Wrapper | Module | Purpose |
| --- | --- | --- |
| `src/execute_location_tracker.py` | `monitoring/location_tracker.py` | 24/7 recorder |
| `src/execute_realtime_monitor.py` | `monitoring/realtime_view.py` | Live view, prints the current location |
| `src/execute_web_preview.py` | `monitoring/web_preview_app.py` | Browser preview and zone editor |
| `src/execute_recalibrate.py` | `monitoring/recalibration.py` | Re-anchor a camera's zones |
| `src/execute_location_report.py` | `monitoring/location_report.py` | Daily summary and delivery |
| `src/execute_discord_bot.py` | `monitoring/discord_bot.py` | Answers report requests in a Discord channel |

The module column is relative to `src/`.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate        # .venv/bin/... also works without activating
pip install -r requirements.txt
```

Dependencies: `ultralytics`, `torch`, `opencv-python`, `numpy`, `PyYAML`,
`discord.py` (the bot only; the tracker and the report do not import it).

### `.env`

Gitignored, and the only place the secrets live. Both the camera URLs and the
report credentials belong here because `config_loader` substitutes `${VAR}` from
the environment and systemd starts the services with an empty environment.

```
LIVING_ROOM_RTSP_URL=rtsp://<user>:<pass>@<ip>:<port>/h264/ch1/main/av_stream
SOFA_RTSP_URL=rtsp://<user>:<pass>@<ip>:<port>/h264/ch1/main/av_stream
FEEDER_RTSP_URL=rtsp://<user>:<pass>@<ip>:<port>/h264/ch1/main/av_stream
LLM_API_KEY=...
LOCATION_REPORT_WEBHOOK_URL=https://discord.com/api/webhooks/.../...
DISCORD_BOT_TOKEN=...                 # only needed for the inbound bot
```

The camera placeholders in `configs/config.yaml` are `${LIVING_ROOM_RTSP_URL:}`
and friends, so an unset variable becomes an empty string rather than an error.
`scripts/run_tracker.sh` therefore checks every configured camera before starting
and fails with the names of the ones that are missing.

## Usage

### 1. Collect candidate frames

Samples the cameras and writes frames that the detector is not already confident
about, with auto-generated labels, under
`dataset/identity_detection_candidates/{images,labels}/<camera>/`.

```bash
python src/data/collect_identity_detection_from_rtsp.py
python src/data/collect_identity_detection_from_rtsp.py --camera sofa --max-images-per-camera 50
```

### 2. Annotate

Package a candidate batch as an Ultralytics YOLO detection archive and upload it
to CVAT:

```bash
python src/data/create_cvat_yolo_archive.py \
    --images-dir dataset/identity_detection_candidates/images \
    --labels-dir dataset/identity_detection_candidates/labels \
    --output dataset/zip/round6.zip \
    --class-name bagel --class-name kurumi
```

Corrected labels come back as a new round directory (`dataset/2`, `dataset/3`,
…), each with its own `images/` and `labels/` split into `train/`, `val/`,
`test/` and `skipped/`.

### 3. Split a round

```bash
python src/data/split_detection_dataset.py --dataset dataset/6 --dry-run
python src/data/split_detection_dataset.py --dataset dataset/6
```

### 4. Train

`configs/config.yaml` points `training.identity_detection_dataset` at the round to
use. `dataset/data.yaml` is the file Ultralytics actually reads:

```yaml
path: dataset
train: [5/images/train]
val:   [5/images/val]
names: {0: bagel, 1: kurumi}
```

```bash
python src/training/train_cat_identity_detector.py --epochs 50
python src/training/train_cat_identity_detector.py --resume      # continues the last run
```

Then point `models.identity_detection_model_path` in `configs/config.yaml` at the
new `weights/best.pt`. **The class indices must stay `0: bagel`, `1: kurumi`** —
the tracker refuses to start if `model.names` differs from the configured
`cats.identity_classes`, because a silently reshuffled index would swap the two
cats in every report.

### 5. Draw zones

The zone editor is served over HTTP rather than shown in a window, because WSLg
cannot keep an OpenCV window visible once the monitor layout changes.

```bash
python src/execute_web_preview.py
# then open http://localhost:8765
```

The canvas draws polygons in the camera's reference frame and saves them to
`configs/zones.yaml` via `POST /api/zones/<camera>`. Each detection is drawn with
a crosshair on its bottom-centre anchor, which is the point zone lookup uses, so
what you see is what the tracker will match.

### 6. Re-anchor after the camera moved

If a camera is knocked out of place, or alignment quality drops to `degraded`,
project the stored zones onto a fresh frame instead of redrawing them:

```bash
python src/execute_recalibrate.py --camera sofa
```

This writes a **new** `calibration_id` (with the reference frame and an overlay
image) rather than overwriting the old one, and posts the result to Discord for
confirmation.

### 7. Run the tracker

```bash
python src/execute_location_tracker.py                        # all cameras
python src/execute_location_tracker.py --camera sofa          # one camera, for debugging
```

It samples every `tracking.sample_interval_seconds`, resolves each cat to a zone,
and writes observations plus dwell visits to `data/location_history.db`. A zone
change is only accepted after `tracking.switch_min_samples` consecutive samples,
and a visit is closed after `tracking.missing_timeout_seconds` without a
detection. When alignment quality is too poor, the sample is stored with
`zone = NULL` instead of guessing.

The database uses WAL and commits every write, so the report can read it while
the tracker is running.

### 8. Daily report

```bash
scripts/daily_report.sh                    # yesterday, picked from .env
scripts/daily_report.sh --print-only       # build it, print it, send nothing
scripts/daily_report.sh --days-ago 3
scripts/daily_report.sh --date 2026-09-19 --database /tmp/demo.db
```

`report.mode: llm` sends the day's summary to an OpenAI-compatible endpoint
(DeepSeek by default) and posts the narrative to Discord. The prompt is assembled
in `build_instruction()` from a tunable `persona` / `style` / `hints` plus fixed
code constants; see [Prompt Layers](#prompt-layers).

To try it without waiting a day, fabricate one:

```bash
python scripts/make_demo_day.py /tmp/demo.db
scripts/daily_report.sh --date 2026-09-19 --database /tmp/demo.db --dry-run
```

### 9. Ask from Discord

Posting is a webhook, but a webhook can only send. Answering a question means
holding a **Gateway** connection, which is also the only option that works from
this machine: the bot dials out over a WebSocket, so no inbound port, no public
IP and no tunnel are needed. (Slash commands over the Interactions HTTP endpoint
would need a public HTTPS URL and a reply within three seconds.)

In the [Discord Developer Portal](https://discord.com/developers/applications):

1. **New Application → Bot → Reset Token**, put it in `.env` as
   `DISCORD_BOT_TOKEN`.
2. On the same page, enable **Message Content Intent** under *Privileged Gateway
   Intents*. Without it the events still arrive but `message.content` is always
   empty, so the bot looks dead while it is connected.
3. **OAuth2 → URL Generator**: scope `bot`, then open the generated URL to invite
   it to your server. It needs *View Channel*, *Send Messages* and *Read Message
   History* in whichever channel it should answer in.

```bash
scripts/run_discord_bot.sh --check-only    # validates the config, connects nothing
scripts/run_discord_bot.sh                 # run it in the foreground
```

Then say this in the channel:

```
报告一下今天两只猫都做什么了
```

`monitoring/discord_bot.py` reads the day's visits with the same
`build_summary()` the daily report uses, hands your own words to the LLM as the
`question` block, and replies in the channel. Only messages containing a trigger
from `discord_bot.triggers` are answered, and bot accounts are ignored (the reply
quotes the question back, so answering a bot would loop).

- `昨天` / `yesterday` and `前天` ask about other days; otherwise
  `discord_bot.days_ago` (0 = today) decides.
- `allowed_channel_ids` / `allowed_user_ids` restrict who can spend an LLM call.
  Empty means *no restriction*, so set them.
- Two fallbacks keep the bot from going silent: if the data cannot be read it
  replies with the day and the reason, and if only the LLM call fails it replies
  with the plain `render_text()` summary rather than an apology.

### 10. Run it 24/7

`deploy/systemd/` holds three units: `cat-tracker.service` runs the recorder with
`Restart=always`, `cat-report.timer` fires `cat-report.service` at midnight, and
`cat-discord.service` keeps the bot connected.

```bash
scripts/install_services.sh              # symlink, enable, enable-linger, start
scripts/install_services.sh --uninstall
```

```bash
systemctl --user status cat-tracker.service
journalctl --user -u cat-tracker -f
systemctl --user list-timers cat-report.timer
systemctl --user start cat-report.service      # send one now, to test
journalctl --user -u cat-discord -f            # live bot log
```

`install_services.sh` preflights each unit before starting it, so a missing
camera URL or a missing `DISCORD_BOT_TOKEN` leaves the unit **enabled but not
started** instead of burning its restart budget. Re-run it after adding the
token.

Two details worth knowing:

- **`loginctl enable-linger` is required.** Without it the user manager stops when
  the last terminal closes and the tracker dies with it.
- **WSL only boots the user manager when the distro is entered**, so after a
  Windows reboot the tracker starts on the first WSL session. The report timer
  sets `Persistent=true`, so a missed midnight still runs on the next boot.

After editing a unit file, `systemctl --user daemon-reload` then restart. Editing
`scripts/run_tracker.sh` only needs a restart.

## Configuration

### `configs/config.yaml`

| Key | Meaning |
| --- | --- |
| `models.identity_detection_model_path` | Weights the tracker and previews load |
| `cats.identity_classes` | Expected class order; validated at startup |
| `identity_collection.cameras` | Camera names and their `${...}` RTSP variables |
| `training.device` | Shared default device (trainer and `realtime_view`) |
| `training.identity_detection_*` | Trainer defaults |

### `configs/locations.yaml`

| Key | Meaning |
| --- | --- |
| `cameras` | Which cameras to sample; names must exist in `config.yaml` |
| `preview` | Browser UI host, port, resolution and JPEG quality |
| `tracking` | Sample interval, confidence, `imgsz`, switch hysteresis, timeouts, database |
| `alignment` | ORB matching thresholds and the trust-last-good window |
| `report` | Timezone, language, delivery mode, and the LLM settings |
| `discord_bot` | Bot token, channel/user allowlists, trigger words, default day |

### Prompt Layers

`build_instruction()` composes the system prompt from blocks, and the placement
rule is: *delete this sentence — would the report lose a category of
information?* If yes it belongs in code, if it only changes how it reads it
belongs in the config.

| Block | Where | Changes when |
| --- | --- | --- |
| `persona` | config | You want a different voice |
| `TASK` | code | The report must cover something new |
| `QUESTION` | runtime | Only when a caller passes one (the Discord bot forwards the message) |
| `EVENTS` | code | A new kind of derived fact appears |
| `PRESENTATION` | code | Language or formatting rules change |
| `style` | config | You want it to read differently |
| `hints` | config | Loose notes about reading the data |
| `GROUNDING_RULES` | code | Never — it stays last for recency |

`hints` must not restate a rule that code already enforces. The toilet
use-vs-pass-by rule used to live there, with the model comparing timestamps
itself; across eleven measured runs it got this right about a third of the time.
It is now `toilet_events()` in `location_report.py`, computed from the visit list
and handed to the model as an already-decided `events` array.

## Testing

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/pre-commit run --all-files
```

`tests/` covers alignment, zone lookup, tracking, the web UI's embedded
JavaScript, the report's prompt assembly and delivery, and the Discord bot's
decision rules (which messages to answer, which day they mean, what to reply).
The bot tests run without a token or a connection: everything above `run()` in
`discord_bot.py` is free of any `discord` import on purpose. The two JavaScript
guards are worth knowing about: one checks per-line quote balance (a raw newline
inside a Python string ends a JS string literal early and the browser discards the
whole script), the other checks that every function the JS calls is defined.
