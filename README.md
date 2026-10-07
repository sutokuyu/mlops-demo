# MLOps Cat Demo

Three fixed RTSP cameras watch a living room. A YOLO model detects **which** cat
is in frame, zone polygons drawn on each camera's reference frame turn that into
a room location, and the result is recorded to SQLite around the clock. A
scheduled job turns the finished day into a short narrative and posts it to
Discord. Ask in the channel what happened today and the bot answers there too, or
ask it for a live frame with the zones drawn on it to aim a camera.

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
- Send one live frame on request, with the zones drawn where they are stored, so
  a camera can be re-aimed without opening the editor.

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
   ├── snapshot.py            │  "@bot 沙发" ──► one frame, zones drawn
   ├── location_queries.py    │  "进过水池吗" ──► counted in code, as callable tools
   └── location_report.py ────┘  what happened today ──► Discord (webhook)
```

## Repository Layout

| Path | What it is |
| --- | --- |
| `src/data/` | Dataset collection and preparation |
| `src/training/` | YOLO detection training |
| `src/monitoring/` | Zones, alignment, tracking, reporting, query tools, browser UI |
| `src/notification/` | Discord webhook posting |
| `configs/config.yaml` | Cameras, model paths, training defaults |
| `configs/locations.yaml` | Tracking loop, alignment, preview, report, Discord bot |
| `configs/zones.yaml` | Zone polygons, written by the tools |
| `deploy/systemd/` | systemd user units for the 24/7 tracker, the daily report and the Discord bot (Linux) |
| `deploy/launchd/` | launchd agent plists for the same jobs (macOS) |
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

**Every runnable entry point reads this file itself**, before it imports anything that
reads the config - so `python src/execute_web_preview.py` works in a fresh terminal
with nothing exported. This is not a convenience: the config substitutes `${VAR}` at
*import* time, so a value loaded later never reaches the camera URLs, and a shell that
exported them days ago would otherwise keep a stale address winning over the file the
tracker keeps correcting.

`.env` therefore wins over the surrounding environment. To override it deliberately for
one run, point `MLOPS_ENV_FILE` at another file:

```bash
MLOPS_ENV_FILE=/tmp/fake.env python src/execute_web_preview.py
```

The **host** part of those URLs is maintained automatically - see
[When a camera changes address](#when-a-camera-changes-address). Only the address
is ever rewritten; the credentials, port and stream path stay as you typed them.

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

Every configured camera is listed as a tab, but **only the tab being looked at gets a
stream**: the camera is opened when a browser subscribes to its MJPEG and released
`--idle-seconds` (default 15) after the last one looks away. The tracker already
records all of them through one Wi-Fi relay, and a preview that quietly held a session
per camera competes with that recording for the same link.

The canvas draws polygons in the camera's reference frame and saves them to
`configs/zones.yaml` via `POST /api/zones/<camera>`. Each detection is drawn with
a crosshair on its bottom-centre anchor, which is the point zone lookup uses, so
what you see is what the tracker will match. Moving the pointer over the picture
shows its coordinates on the **0-100 scale** (`x 62.13  y 91.40`) in the corner of
the frame - the same numbers a Discord point question takes, so a spot can be read
off the screen instead of guessed.

Editing a polygon that is already saved: click empty space to start a new one, or
**drag any drawn corner** to nudge it. Handles only respond while 显示区域描点 is
ticked, and nothing is written until 保存区域 is pressed.

### 6. Re-anchor after the camera moved

If a camera is knocked out of place, or alignment quality drops to `degraded`,
project the stored zones onto a fresh frame instead of redrawing them:

```bash
python src/execute_recalibrate.py --camera sofa
```

This writes a **new** `calibration_id` (with the reference frame and an overlay
image) rather than overwriting the old one, and posts the result to Discord for
confirmation.

The tracker re-anchors on its own when a drift is detected, which is controlled by
`alignment.reanchor_mode`:

| mode | what a detected drift does |
| --- | --- |
| `apply` | projects the zones onto a fresh frame and adopts it as the new reference (default) |
| `alert` | sends one LLM-worded warning and leaves the zones exactly as drawn |

This machine runs `alert`. A projection is only as good as the transform behind
it, and a bad one fails quietly: an automatic re-anchor whose transform reported
`scale=0.000` stored all eight of feeder's hand-drawn zones as a single point at
`(0, 0)`. The drift is still detected the same way (`alignment.reanchor_trigger`
and its thresholds), so the warning still arrives - the repair is just left to a
human. The mode gates the **automatic** path only: `execute_recalibrate.py` and
the editor's 重锚定 button keep re-anchoring whenever you ask them to.

One drift is one message. The alert is re-armed only after the camera has lined
up with its reference frame continuously for
`alignment.reanchor_alert_clear_seconds` (default 600s), not after a single clean
sample: at night these cameras flicker between "large" and "lined up" every few
samples, and clearing the flag on that one good sample sent 32 messages in 38
minutes for a drift that never changed. An unknown mode, or an unknown
trigger, raises at startup rather than quietly picking a side.

### 7. Run the tracker

```bash
python src/execute_location_tracker.py                        # all cameras
python src/execute_location_tracker.py --camera sofa          # one camera, for debugging
```

It samples every `tracking.sample_interval_seconds`, resolves each cat to a zone,
and writes observations plus dwell visits to `data/location_history.db`. Each
observation keeps the anchor twice: `norm_x`/`norm_y` in the reference frame the
zones were drawn on (for zone lookup), and `cam_x`/`cam_y` in the camera's own frame
plus `frame_width`/`frame_height` (for pointing at a spot on the live picture). A
zone change is only accepted after `tracking.switch_min_samples` consecutive samples,
and a visit is closed after `tracking.missing_timeout_seconds` without a
detection. When alignment quality is too poor, the sample is stored with
`zone = NULL` instead of guessing.

The database uses WAL and commits every write, so the report can read it while
the tracker is running.

#### When a camera changes address

The cameras hang behind a Wi-Fi relay that hands them a new address every so
often, and a stale `*_RTSP_URL` is invisible until a camera quietly stops being
recorded. Two things keep that from needing a human.

**The tracker relocates itself.** Once a camera has been unreachable for
`discovery.missing_after_seconds`, it sweeps the subnet in a background thread
and adopts the address it finds, without a restart. The owner gets exactly one
Discord message per outage either way: one when the camera cannot be found, and
one when it answers again. That second message is what re-arms the alert, so a
camera that flaps all night is news twice, not news every five minutes.

**`.env` is kept correct** for everything else that reads it - `run_tracker.sh`,
`realtime_view`, the browser preview:

```bash
scripts/sync_camera_ips.py               # report only, change nothing
scripts/sync_camera_ips.py --apply       # rewrite .env, keeping .env.bak
scripts/sync_camera_ips.py --explain     # what every reachable address answered
```

`cat-camera-ip.timer` runs the `--apply` form every five minutes, and
`run_tracker.sh` runs it once before starting. It exits `2` when a camera is still
missing, which is a warning about the camera rather than a failure of the job.

While the tracker is recording, that lookup **stands down** (and a lock keeps two
of them from running at once). An address is identified by opening a stream on the
camera, which competes with the recording for the same relay link - measured: two
lookups alongside the tracker stalled a 2560x1440 handshake past 30 seconds and the
tracker dropped its stream on another camera. In that state the tracker maintains
`.env` itself, and `--force` looks anyway for a diagnosis done on purpose.

**How a camera is identified - and why it is not by MAC.** The relay rewrites the
source MAC, so every device behind it answers ARP with the same hardware address
(measured on this network: twelve addresses in `192.168.3.0/24`, all three cameras
among them, one MAC). A MAC table therefore only says "this device is behind the
relay", and a DHCP reservation on the router cannot separate them either - the
router sees that same one address. So each candidate address is asked the one
question only a camera can answer:

1. **Its password.** Every camera has its own; the others are refused with 401.
2. **Its picture.** One frame, matched with the same ORB matcher the zones use,
   against the reference frame those zones were drawn on.

Both have to agree. A picture that clearly belongs to a *different* camera vetoes
the credentials instead of confirming them, because a wrong address is worse than
an offline camera: offline loses samples, a mixed-up camera files one room's cat
under another room's zones. When the evidence cannot separate two addresses the
camera is left alone and the ambiguity is reported.

`--explain` prints the evidence for every address, which is what makes "not
found" actionable: it names the addresses that speak RTSP but refuse this
camera's password, and the ones that answer an SDK port but no RTSP at all.

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
scripts/run_discord_bot.sh --check-only    # validate config and ask Discord two questions
scripts/run_discord_bot.sh --offline --check-only   # ...without the network
scripts/run_discord_bot.sh                 # run it in the foreground
```

`--check-only` never opens the Gateway, but it does ask Discord two things that
cannot be answered locally and that both turn into a service restarting forever:
is the token still valid (a reset token is otherwise only discovered from
`journalctl`), and is **Message Content Intent** on. The answer comes from the
application's own flags in `GET /applications/@me`, and the intent being off is
reported with the exact portal path to fix it - the Gateway would refuse the
connection with `PrivilegedIntentsRequired`.

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

#### Ask for a live frame

The same bot can send one frame from a camera with the zone polygons drawn on it,
for aiming a camera without opening the editor:

```
沙发画面
feeder 截图
@Cat monitor assistant sofa
```

Any word from `discord_bot.snapshot.triggers` asks for a picture, and a real
@mention whose whole message is a camera name counts too. Cameras can be named by
their key (`living_room`, or `living room` with a space instead of the underscore)
or by any alias under `discord_bot.snapshot.aliases` (`沙发`, `客厅`, `喂食器`). Name
no camera and the only configured camera is used, or the bot replies with the
choices.

The zones are drawn **where they are stored**, in the reference frame's
coordinates - deliberately not projected onto the live frame. A projection would
follow the camera and hide exactly the misalignment you are looking for; drawn as
stored, a polygon that no longer sits on its furniture is the signal. The
`align=` note on the image (and the sentence under it) says how far the live view
is from the reference frame, which tells a moved camera apart from changed
lighting.

The bot holds one stream per camera while you keep asking and releases it
`discord_bot.snapshot.idle_timeout_seconds` after the last request, so an aiming
session costs one extra session on the relay rather than one per message. The
camera address is re-read from `.env` per request, so a camera the tracker has
relocated is found without restarting the bot.

#### Ask about a place

A question that names a place is **counted in code** before the model sees it:

```
kurumi 有没有进过水池      # ever, over all recorded history
kurumi 今天在猫砂盆待了多久
bagel 最近3天去过哪
```

This is the third time the same lesson came back. The toilet rule was right about a
third of the time while it lived in `hints`; the meal rule was applied unevenly between
the two cats because the model invented its own criterion; and on 2026-10-02 the bot
answered "没进过水池" while the database held four `sink` stays that day. Asked with the
identifier instead, the same model on the same data answered correctly - so the failure
was that the owner's word and the stored name never met, and ``GROUNDING_RULES`` turned
that into a confident "no such record" (one run even invented support for it).

Two fixes, both in code:

* **`report.zone_aliases`** maps each zone identifier to the words that may refer to it
  (`sink: [水池, 水槽, ...]`). The same list goes into the prompt as the allowed
  vocabulary and drives the resolver, so the question and the data share one language.
  One word belongs to exactly one zone here - a word claimed by two is a typo, and a test
  fails on it.
* **`report.zone_groups`** is for a word that genuinely means several zones. The cat has
  two wet food bowls, so `湿粮碗: [wet_food_bowl_1, wet_food_bowl_2]` makes "去过湿粮碗吗"
  count both, while the answer still breaks the total down per bowl
  (`counts_by_zone`). A numbered word still wins over the group: "湿粮碗2" is only bowl 2.
  Zone names that repeat across cameras need no group - `sink` is one identifier on two
  cameras, and the answer names the camera.
* **`location_queries.answer_question()`** turns a question that names a place into an
  exact count - times, minutes, cameras - which rides to the model as a `query` field it
  must answer from instead of recounting the timeline. A question naming no place still
  goes down the old path untouched.

A question with no time word defaults to today; 昨天/前天/最近N天/这周/本月 and explicit
dates all work. "有没有进过" without a time word means *all recorded history*, because
that is what it asks, and scoping it to today is how the 10-02 record stayed invisible
on the 10-03 question. Zone names repeat across cameras (`sink` is on both living_room
and sofa), so every answer names its camera.

#### The tool layer a harness calls

The counting lives in `src/monitoring/location_queries.py`, shaped so a DeepSeek
harness (or any function-calling loop) can use it as-is:

```python
from src.monitoring.location_queries import tools_schema, call_tool

tools_schema()          # OpenAI-compatible function definitions, ready to register
call_tool("zone_stay", {"zones": ["sink"], "cat": "kurumi", "since": "2026-10-01"})
# -> {"ok": True, "tool": "zone_stay", "result": {"count": 3, "total_minutes": 3.8, ...}}
```

Everything in the module is one dictionary in, one dictionary out, JSON-serializable
both ways, deterministic, and free of any LLM, network or clock dependency (`now` is
injectable). `call_tool` never raises: an unknown tool or bad arguments come back as
`{"ok": False, "error": ...}` for the model to read. Adding a tool means writing one
function with keyword arguments and adding one `Tool` entry - the schema, the
`call_tool` dispatch and the harness integration need no changes.

The tools are `zone_stay` (how often and how long in given zones), `zone_totals` (time
per zone over a range, for cross-day questions), `daily_summary` (the report's own
summary and its meal/drink/toilet verdicts) and `point_stay` (who stayed near a
specific spot on one camera).

#### Asking about a spot, not a zone

A zone like `floor` is far too coarse to say *where* on the floor something happened.
`point_stay` takes a camera and an `x`/`y` on the **0-100 scale** the preview readout
shows (0..1 fractions and raw pixels are accepted too - the magnitudes decide when
`unit` is omitted) and returns, per cat, the minutes spent within `radius` of that
point, with the individual stays:

```python
call_tool("point_stay", {"camera": "living_room", "x": 60, "y": 50})
# -> {"ok": True, "result": {"minutes_by_cat": {"kurumi": 4.0}, "stays": [...], ...}}
```

A coordinate only means something on one camera, so `camera` is required and the same
numbers are a different place on another camera. Consecutive samples inside the radius
and no more than `report.query.point_gap_seconds` apart count as one stay, which is
what separates a cat that *stayed* there from one walking through - sort by the minutes
to tell them apart. Rows written before `cam_x`/`cam_y` existed fall back to the stored
detection box (its bottom centre is the same anchor), so the whole history is
queryable without a backfill.

The bot understands this too: a message naming a coordinate (`客厅地板上 (0.6, 0.5)`)
is answered from `point_stay`. Which words mean which camera is the *same* list the
snapshot feature uses, `discord_bot.snapshot.aliases` (客厅/沙发/喂食器...), so there is
one camera vocabulary to maintain. When a coordinate is given without a camera the bot
asks which one rather than guessing.

The answer comes with a **live frame with the spot circled** (orange circle + crosshair,
plus the radius it matched with), so an owner who meant a different place can see that
immediately and ask again. It rides on the same held-stream machinery as "@bot 沙发画面"
- one RTSP session per camera, released when idle - and taking it is best-effort: if the
camera is busy or offline the text answer still arrives, with a note instead of a
picture.

### 10. Run it 24/7

`deploy/systemd/` holds five units: `cat-tracker.service` runs the recorder with
`Restart=always`, `cat-report.timer` fires `cat-report.service` at midnight,
`cat-discord.service` keeps the bot connected, and `cat-camera-ip.timer` re-finds
the cameras and refreshes their addresses in `.env`.

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
systemctl --user list-timers cat-camera-ip.timer
```

`install_services.sh` preflights each unit before starting it, so a missing
camera URL or a bot that fails its checks is left **enabled but not started** -
with the reason printed instead of a unit that restarts forever. Re-run it after
fixing whatever it named.

Two details worth knowing:

- **`loginctl enable-linger` is required.** Without it the user manager stops when
  the last terminal closes and the tracker dies with it.
- **WSL only boots the user manager when the distro is entered**, so after a
  Windows reboot the tracker starts on the first WSL session. The report timer
  sets `Persistent=true`, so a missed midnight still runs on the next boot.

After editing a unit file, `systemctl --user daemon-reload` then restart. Editing
`scripts/run_tracker.sh` only needs a restart.

#### macOS (e.g. a Mac mini M2)

`deploy/launchd/` holds the same four jobs as launchd agents (there is no separate
timer unit on launchd; `cat-report` and `cat-camera-ip` schedule themselves in
their own plist).

```bash
scripts/install_services_macos.sh              # generate plists, bootstrap, start
scripts/install_services_macos.sh --uninstall
```

```bash
launchctl print gui/$(id -u)/com.sutokuyu.mlops-demo.cat-tracker
tail -f ~/Library/Logs/mlops-demo/cat-tracker.log
launchctl kickstart -k gui/$(id -u)/com.sutokuyu.mlops-demo.cat-report   # send one now, to test
tail -f ~/Library/Logs/mlops-demo/cat-discord.log                       # live bot log
```

Two details worth knowing:

- **These are LaunchAgents, not LaunchDaemons**: they only run while someone is
  logged in. For a Mac mini acting as a headless box, enable automatic login
  (System Settings > Users & Groups > Login Options) and stop it sleeping
  (`sudo pmset -a sleep 0`, or System Settings > Energy) - otherwise the tracker
  stops recording whenever the screen locks or the machine sleeps.
- **`training.device` and `tracking.device` both default to `auto`**, resolved at
  startup (see `resolve_device` in `src/config_loader.py`) to cuda if torch sees a
  GPU, else Apple Silicon's `mps`, else `cpu` - the same `configs/*.yaml` works
  unedited on WSL/a CUDA box and on a Mac mini. Pass `--device cpu` for any op
  Ultralytics does not yet support on `mps`, or set an explicit value in the YAML
  to pin one machine to a particular device.

After editing a plist, re-run `install_services_macos.sh` to pick it up.

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
| `discovery` | Where to look for a camera that moved: subnets, ports, scan timing, how long to wait before searching and how often to retry |
| `preview` | Browser UI host, port, resolution and JPEG quality |
| `tracking` | Sample interval, confidence, `imgsz`, switch hysteresis, timeouts, database |
| `alignment` | ORB matching thresholds and the trust-last-good window |
| `report` | Timezone, language, delivery mode, and the LLM settings |
| `report.zone_aliases` | Zone identifier → the words the owner uses for it (the shared vocabulary) |
| `report.zone_groups` | A word that means several zones at once (e.g. 湿粮碗 = bowl 1 or 2) |
| `report.query` | Limits for the code-computed answers (`max_stays`, `max_days`) |
| `discord_bot` | Bot token, channel/user allowlists, trigger words, default day |
| `discord_bot.snapshot` | Picture-request words, camera aliases, image size, how long a stream is held |

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
JavaScript, the report's prompt assembly and delivery, camera discovery, the
Discord bot's decision rules (which messages to answer, which day they mean, what
to reply), and the zone questions - which are answered without a camera, a network
or an LLM: the database is a throwaway file and the clock is injected.
Nothing in `tests/test_camera_discovery.py` touches a network or a camera: the
scan, the stream open and the picture match are injected, so the rules that cost
real debugging - which signal decides identity, and what to do when the evidence
disagrees - are tested without one.
The bot tests run without a token or a connection: everything above `run()` in
`discord_bot.py` is free of any `discord` import on purpose (the two exceptions,
`build_client` and `discord_file`, exist only to create the client and to attach an
uploaded image). The two JavaScript
guards are worth knowing about: one checks per-line quote balance (a raw newline
inside a Python string ends a JS string literal early and the browser discards the
whole script), the other checks that every function the JS calls is defined.
