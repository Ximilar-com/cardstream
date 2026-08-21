# cardstream documentation

The [README](../README.md) gets you running. These pages are the reference for
when you need to change something.

| Page | What is in it |
| --- | --- |
| **[CLI reference](cli-reference.md)** | every flag, its default and what it does — generated from the parser, so it cannot drift |
| **[Tuning](tuning.md)** | the call economics, the detection filters, the image archive, the settings dialog, and the three resolution knobs |
| **[Locating the card](locators.md)** | detection vs segmentation, `--detection-expansion`, the optional visual tracker, and training your own |
| **[Frame sources](sources.md)** | webcams, files, RTSP/RTMP/SRT, `--listen`, and the system ffmpeg requirement |
| **[Model weights](../model/README.md)** | what goes in `model/`, each export's I/O contract, and the model zoo |

Also worth reading: **[SECURITY.md](../SECURITY.md)** — this app holds a
credential that spends money and opens an unauthenticated local port, and both
facts have consequences for how you should run it.

Contributors want **[CONTRIBUTING.md](../CONTRIBUTING.md)** and
**[CLAUDE.md](../CLAUDE.md)**, the architecture map.

## A note on where documentation lives

These pages are the canonical reference and live beside the code on purpose:
`tests/test_docs_contract.py` reads them and checks every documented
`(default X)` against the real argparse parser. A default that changes without
its documentation failing CI is a bug this project treats like any other.

[cardstream.ai](https://cardstream.ai) carries the same material shaped for
reading rather than for reference, plus guides and the model zoo.
