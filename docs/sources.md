# Frame sources

`--source` accepts a webcam index, a file, a stream URL, or `camera` (the
browser's own webcam, `cardstream-web` only). Analysis is identical whichever
one is in use — sources differ only in how frames arrive.

| `--source` | What it is |
| --- | --- |
| `camera` | the browser's webcam; frames are pushed over the WebSocket (`cardstream-web` only) |
| `0`, `1`, … | a local webcam by index |
| `card.jpg`, `clip.mp4` | a file; video replays at its native frame rate, `--loop` repeats it |
| `rtsp://…` `rtmp://…` `srt://…` | a stream this process **pulls** |
| `ws://…` | a binary-JPEG WebSocket feed |
| any of the above + `--listen` | a stream **pushed** to us by an encoder (OBS → RTMP, SRT callers) |

```bash
cardstream-web --source rtsp://user:pw@cam/stream1 --debug
cardstream-web --source rtmp://0.0.0.0:1935/live --listen   # OBS pushes here
cardstream-web --source "srt://:9000" --listen              # SRT caller pushes
cardstream-client --source 0                                # headless webcam
cardstream-client --source card.jpg --loop
```

With a URL `--source` the browser page becomes a passive viewer: incoming
frames, results, the located card with `--show-detection`, and a live log panel
with `--debug`. Analysis runs even with no browser open.

Pulled sources arrive at their native resolution and are downscaled for
analysis only (`--width`, default 960; `0` analyses full-res). Video **files**
replay at their native frame rate, like a real feed — override with `--fps`
(`0` = as fast as decoding allows).

## SRT / RTMP / RTSP

All three are consumed directly: pull via `--source rtsp://…` (also `rtmp://`,
`srt://`), or `--listen` to receive streams *pushed* by an encoder (OBS →
RTMP, SRT callers). A new protocol is a `FrameSource` subclass in
`cardstream/client/sources.py` plus a branch in `make_source`.

## The system ffmpeg binary

`--listen` and `--ffmpeg` need **system ffmpeg** on your `PATH`
(`brew install ffmpeg`, or your distribution's package). Plain pulls work with
a pip-only install. `--ffmpeg` is the portability fallback for when the OpenCV
wheel's bundled FFmpeg lacks a protocol — libsrt is the usual culprit.

Keep `cardstream-web` on localhost: that process holds your Ximilar key. See
[SECURITY.md](../SECURITY.md).
