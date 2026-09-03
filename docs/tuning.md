# Tuning

Everything here is a flag. `cardstream-web --help` and
`cardstream-client --help` list them all with their defaults; the defaults
themselves live in one place, `AnalyzerConfig` in `client/analyzer.py`.

The shipped defaults **are** the tuned configuration — a bare `cardstream-web`
runs the same pipeline `make dev` spells out flag by flag. Reach for this page
when your framing, lighting or card mix differs from the common case.

See also: [Locating the card](locators.md) · [Frame sources](sources.md) ·
[CLI reference](cli-reference.md)


All vision runs on your machine; one JPEG crop leaves it per **distinct** card:

| Stage | Where it runs | Cost |
|---|---|---|
| Frame capture | browser webcam, or a pulled source | free |
| Motion gate | local (mean frame-diff) | free |
| Card location | local: detection (RF-DETR / RT-DETRv2) or segmentation (RF-DETR) | free |
| Same-card identity gate | local (embedding cosine, or pHash) | free |
| Identify lookup | Ximilar `collectibles/v2/*_id` | **paid, once per distinct card** |

```bash
export XIMILAR_API_KEY=...              # the one env var the client reads

cardstream-web                          # browser UI on your webcam (http://127.0.0.1:8001)
cardstream-web --gate phash             # zero-ML fallback, no torch/onnx needed
cardstream-web --source rtsp://user:pw@cam/stream1 --debug
cardstream-web --source rtmp://0.0.0.0:1935/live --listen   # OBS pushes here
cardstream-web --source "srt://:9000" --listen              # SRT caller pushes
cardstream-client --source 0            # headless webcam
cardstream-client --source card.jpg --loop
cardstream-web --game "One Piece"       # prefill the game — faster, more precise tcg_id
cardstream-web --set-code PBL           # restrict matching to one set
cardstream-web --alphabet japanese      # japanese cards — see the warning below
cardstream-web --price-stats            # USD market prices with every match (tcg / sport / comics)
cardstream-web --camera-width 3840      # ask the webcam for more pixels to identify from
cardstream-web --store-images crops/    # keep every crop that was paid for
cardstream-web --store-images shots/ --store-images-type frame   # …or the whole frame
cardstream-web --min-card-size 0 --min-card-aspect-ratio 0   # accept every detection
cardstream-web --retry-unmatched 0      # never re-ask about a card that came back nameless
```

## Call economics: when a card costs money

A card that **matched** is asked about exactly once, however long it is held —
that is the whole point of the identity gate. A card that came back with
*nothing* is different: the gate commits its signature when the call fires, not
when it succeeds, so without help a single bad look (glare, a hand across the
art, a match the result threshold dropped) would stick for as long as the card
stays in frame. `--retry-unmatched SECONDS` (default **0.5**) gives it another
attempt after that delay, and again each time the delay passes while it is still
unnamed; `0` restores the strict never-retry policy. The cost to watch is
something that can never match — a slab back, a hand read as a card — sitting in
shot: at the default that is two calls a second for as long as it is there.
`--min-card-size` is the cheap guard against the sliver cases; raise
`--retry-unmatched` for the rest.

Two filters drop a detection before it can cost anything — a card caught half
out of shot, held edge-on mid-swap, or a corner poking past a sleeve detects at
0.9 confidence and identifies at nothing:

- **`--min-card-size FRACTION`** (default `0.1`) — too small, measured in either
  dimension against the analysed frame. A fraction rather than a pixel count, so
  it keeps its meaning when `--width` or the camera resolution changes. Lower it
  if your framing is wide and real cards are being ignored.
- **`--min-card-aspect-ratio RATIO`** (default `0.4`) — wrong proportions:
  shortest side over longest. A card is ~0.71 held portrait *or* landscape, so
  the ratio is orientation-blind, and unlike the size filter it does not move
  when someone holds a card nearer the lens. The 214×702 fragment that prompted
  this is 0.30.

`0` disables either one. `--debug` logs each rejection —
`[size] box 120x300 under 0.20 of 960x540 — ignored`, or
`[aspect] box 214x702 is 0.30 (under 0.40) — ignored`.

`--store-images FOLDER` writes one JPEG per **paid** call — the free frames
never touch the disk. Each file is the record's own `_base64` decoded, so it is
byte-for-byte what the endpoint saw (upscaling and JPEG quality included),
named `<call number>-<random>.jpg`: the number sorts the folder in the order
the cards were shown, the random suffix keeps a second run from overwriting the
first. The image is written *before* the POST, so a card that came back
unmatched is still there to look at. A write that fails is logged and skipped;
identification carries on.

`--store-images-type` picks what the file holds: `object` (the default) is that
crop, and `frame` is the whole frame it was cut from — the same one call per
image either way, but the frame answers a different question (where the card
was, what else was in shot, whether the detector boxed the right thing). The
frame is the ORIGINAL, not the `--width` downscale, and it is encoded inline in
the capture loop rather than on the identify thread, which costs a few
milliseconds per call. Passing `--store-images-type` without `--store-images` is
an error rather than a run that quietly writes nothing.

Both commands open with a banner: the wordmark, then every knob and what it is
set to, with the values that came from a flag marked `●` and showing the
default they replaced. It prints before any model loads, so it is also the
quickest way to confirm a long multi-line command actually arrived intact — a
dropped `\` silently truncates the rest of the line, and the banner shows those
flags sitting at their defaults.

Identify records carry every attribute we already know on the card object
(`_objects[0]`) — always `"Side": "front"` + `"Rotation": "rotation_ok"`, and
for tcg also `"Top Category": "Card"` + `"Category": "Card/Trading Card
Game"`; `--game` (Pokémon, Magic The Gathering, One Piece) additionally sends
the game as `Subcategory` so tcg_id narrows its search ([Ximilar
docs](https://docs.ximilar.com/collectibles/recognition#tcg-identification)).
`--set-code` (alias `--set_code`, e.g. `PBL`) rides on the **record** rather
than the card object — `{"records": [{"_base64": …, "set_code": "PBL",
"_objects": [{…}]}]}` — restricting the id endpoint to that one set. It is
passed through verbatim (codes are not all upper-case). `--no-known-attrs`
drops the Side/Rotation assertions so the endpoint classifies backs and
rotated cards itself.

**`--price-stats` asks for market prices with every match.** Off unless you
pass it; on, the POST body carries the flag at the top level beside the
records — `{"records": [{…}], "price_stats": true}` — and the best match
comes back with aggregated sale statistics per `stats_type` (`ungraded`,
`graded`, `overall`). The amounts are USD; the API names no currency. The
page shows the median, the min–max range and the latest sale with its date
on the card panel, and one line per history row —
`ungraded $24.99 (15–60) · graded $45.00 (30–80)` — which is also what
`cardstream-client` prints under its `[IDENTIFIED]` line. `overall` is shown
only when the card has neither ungraded nor graded sales. Only `tcg_id`,
`sport_id` and `comics_id` document the flag, so on slab it is not sent (the
switch stays on and applies again when you switch back). Ximilar does not
say whether the extra data costs credits, which is why it is opt-in. The ⚙
dialog toggles it live; the next paid call follows.

**`--alphabet` matters more than it looks.** It is **omitted by default** —
no `Alphabet` in the record, so the endpoint classifies the writing system
itself. But prefilling `Subcategory` (i.e. setting a game) switches that
classifier off and the endpoint then assumes `latin`: measured on a Japanese
Avalugg, which came back as its English print (`Chaos Rising (CRI) #24`,
distance 0.099) instead of the Japanese one (`Ninja Spinner (M4) #24`,
distance 0.087). **So whenever you pass `--game`, pass `--alphabet` too**
(`latin` / `japanese` / `chinese` / `korean` / `thai`), or leave the Game
unset and let the endpoint work both out. Unknown values are rejected locally,
because the API accepts them silently and then matches the wrong print.

## The settings dialog

The smart page keeps the Game dropdown at the top of the right panel with a
⚙ button next to it; the dialog behind it holds the rest:

| Setting | Effect |
| --- | --- |
| Category | Which id endpoint the crop goes to (`tcg_id` / `sport_id` / `slab_id` / `comics_id`) **and** the record's `Top Category` / `Category` pair — `Card/Trading Card Game` for tcg, `Card/Sport Card` for sport |
| Game / Sport | The record's `Subcategory`; the list follows Category (Pokémon… for tcg, Baseball… for sport) |
| Alphabet | The record's `Alphabet`: `Not Specified` (default — field omitted), `latin`, `japanese`, `chinese`, `korean`, `thai` |
| Set code | The record's `set_code` |
| Assume front side, upright | On: send `Side: front` + `Rotation: rotation_ok`. Off: let the endpoint decide |
| Market price statistics | On: send the top-level `price_stats` flag; the card panel and every history row show the USD median, range and latest sale (tcg / sport / comics only) |
| Result threshold | `--result-threshold` retuned live on every running analyzer |
| Send rate, Show detection box | Page-local: capture fps and the bbox overlay |

Everything but the last row round-trips through `GET`/`POST /settings` on the
local process, so it applies to the next identify call without a restart.
Switching Category clears a game the new endpoint doesn't know.

## Resolution: analyse small, identify big

Local analysis runs on a downscaled frame (`--width`, default 960) because
detection, motion and the identity gate gain nothing from more pixels — but the
crop sent to Ximilar is re-cut from the **original** frame, where the set code
and card number are still legible. Three knobs, in the order the pixels travel: `--camera-width` (default 1920)
is what the browser asks the webcam for, `--send-width` (default 1920,
`0` = as captured) caps what it encodes and ships, and `--width` (default 960)
is what gets analysed. The identify crop is cut at **send** resolution, so
`--send-width` is the one that decides match quality — but the page encodes
every frame on its main thread (~10 ms at 1280, ~18 ms at 1920, ~49 ms at
3840), so raising it costs UI smoothness. Pulled sources hand over native
frames and the process downscales its own copy. The `cam` and `analysed` badges show both numbers,
and `--debug` logs `crop=WxH` on every identify — that line is the proof of
what was actually sent.

The header's `N calls` badge counts the paid identify calls this session has
spent (one analyzer = one session: per browser connection in camera mode, per
process with a pulled `--source`). A call that returns no match still counts —
it was still spent.

**A card that stays away is forgotten.** The identity gate normally remembers
the last card across a dropout, so waving one in and out of frame costs
nothing. Past `--forget-after` seconds (default **2**, `0` = never) that
assumption expires: the gate signature and the cached identification are
cleared, so the next card is analysed from scratch and identified again even if
it looks identical. Short detection flickers — the lost/found cycles you see a
second apart in `--debug` — are unaffected. The extra call announces itself:

```
[gate] card away 6.3s (> 2s) — forgetting the last card, analysing fresh
```

The History list under the card panel shows how long each card stayed in
frame: the duration on the right counts up (in accent colour) while the card
is there and freezes the moment it is lost. By default a card put back in
frame **resumes its existing row** and its time keeps totalling (time out of
frame is not counted). Pass `--split-results` (alias `--split_results`) to log
every appearance as its own row instead — the same Avalugg coming back is then
a separate entry, not merged into the latest one. A card swapped in place is
always a new row.

The result panel shows the identify call's wall time next to the match
distance. The located card is NOT drawn by default — pass `--show-detection`
to see it (locating always runs). With `--detector-model` that is a bounding
box; with `--segmentor-model` it is the **four-corner outline**, which is the
exact shape the identify crop is cut from, so what you see is what gets sent.
Both are drawn in red; add `--detection-expansion` and a second green outline
appears showing the wider shape actually paid for. The ⚙ dialog toggles it live.

`--min-card-time SECONDS` (default 1.0) keeps a card out of the history list
until it has been on stream that long, so a card glimpsed mid-swap does not
clutter the log. It only affects the list — a brief card is still detected,
identified and paid for like any other. Time is the total across appearances
unless `--split-results` is set; `0` lists every card.

With a URL `--source` the browser page becomes a passive viewer (incoming
frames, results, the located card with `--show-detection`, and — with
`--debug` — a live
log panel); analysis runs even with no browser open. Pulled sources arrive at
their native resolution and are downscaled for analysis only (`--width`,
default 960, `0` = analyse full-res). Video **files** replay at their native
frame rate, like a real feed — override with `--fps` (`0` = as fast as
decoding allows).

`--listen` and `--ffmpeg` (a portability fallback when the opencv wheel's
FFmpeg lacks a protocol, e.g. libsrt) need the **system ffmpeg** binary:
`brew install ffmpeg`. Keep `cardstream-web` on localhost — that process holds
your Ximilar key.
