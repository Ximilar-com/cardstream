# Saving a show to Ximilar

By default nothing is kept: the history on the page is gone when the tab
closes, and the identify call is the only thing that leaves your machine.
`--ximilar-stream` opts in to one more thing. The show's history, exactly as
the page lists it, the crop each card was identified from, and the number of
paid identify calls are also saved to a **session** on the Ximilar platform. You can come back after the show and see
what was shown, when, for how long, how sure each match was and what it was
worth.

```bash
cardstream-web --game Pokemon --alphabet latin --price-stats --ximilar-stream NEW
```

At startup the client prints the session it started, and where to review it:

```
[session] started Pokémon show 2026-09-29 20:15 (0b7c9a52-…) — review at https://api.ximilar.com/cardstream/v2/session/0b7c9a52-…/summary/
```

On a clean exit (Ctrl-C) it uploads whatever is still queued and closes the
session:

```
[session] 212 card(s), 212 image(s) and 230 paid call(s) saved to 0b7c9a52-… — session closed
```

## Requirements

- The same `XIMILAR_API_KEY` the identify call uses. The session belongs to
  that key's account: its default workspace, or the workspace
  `--ximilar-workspace` names.
- That account needs the **cardstream** service. Without it, the start fails
  with the API's own explanation, before the show begins.

## The flags

| Flag | What it does |
| --- | --- |
| `--ximilar-stream NEW` | start a new session |
| `--ximilar-stream ID` | resume a session, e.g. after restarting the client mid-show; a closed one is reopened |
| `--ximilar-stream-name NAME` | name a new session; otherwise it is named after the game (or card type) and the start time |
| `--ximilar-stream-platform` | where the show streams: `whatnot`, `tiktok`, `ebay`, `fanatics`, `youtube`, `twitch` or `other` |
| `--no-ximilar-stream-images` | save the rows as text only, without the crop of each card |
| `--ximilar-stream-keep-open` | leave the session live on exit, as a show that is not over yet |
| `--ximilar-workspace ID` | save the session in this workspace instead of the API key's default one; also needed to resume a session of that workspace |
| `--ximilar-stream-url URL` | the session API base URL, only for a development backend |

The name and platform describe a **new** session; passing them with an ID is
refused. Any of the other session flags without `--ximilar-stream` is refused
too, rather than silently doing nothing.

A clean exit closes the session, but it can still be continued: resuming it
with `--ximilar-stream ID` reopens it and the show goes on in the same session
(the session page on the Ximilar platform shows this command as "Continue this
show"). Only starting a session is billed, so reopening it costs nothing. With
`--ximilar-stream-keep-open` the session simply stays live between runs.

A session lives in one workspace. Without `--ximilar-workspace` that is the
API key's default workspace; pass the same `--ximilar-workspace ID` when
resuming a session saved in another one, or it is not found.

## What is sent

**One row per card shown, exactly as the page's history lists it.** The same
card identified again while it stays on stream, or brought back before a
different card appears, stays one row: its time on stream keeps adding up and
the paid calls behind it are counted. A card shown for less than
`--min-card-time` gets no row, on the page or in the session. With
`--split-results` every appearance is its own row. For each row:

- **When:** when the card appeared, and how long it was on stream in total.
- **Calls:** how many paid identify calls matched it. 0 is possible with
  `--split-results`, when a returning card is shown from memory.
- **Card:** the category (`tcg`, `sport`, `slab`, `comics`), name, full name,
  set name and code, card number, series, year and subcategory.
- **Match:** the match distance and confidence tier, the links and up to four
  alternatives, all from the row's first identification.
- **Prices:** the price statistics, when `--price-stats` is on. The platform
  derives a representative price from them.

**The crop of each row:** the cut-out card the row's first identification
was made from, the same picture the identify call received, as a JPEG no
larger than 1024 px on its long side. It is uploaded once per row, right
after the row itself is saved; the platform keeps it privately with a
thumbnail, and deleting the row or the session deletes them.
`--no-ximilar-stream-images` turns this off.

A row is sent as soon as it earns its place in the list. It is sent again,
under the same id, when the card leaves or comes back, and every ten seconds
while it stays, so the session updates the row rather than adding one.

**The number of paid identify calls**, matched or not: the page's "N calls"
badge. Each run of the client reports its own total, so a resumed session adds
the new run's calls to the earlier ones.

At the start, the session is described by its name, game, platform and the
client's settings (version, card type, set code, alphabet and whether prices
are on).

What is **not** sent: frames or video, crops of matches that did not start a
row (a card identified again is merged into its row, which keeps its first
crop), the page's own history thumbnails, anything from the platform you
stream on (buyers, sales, chat), and matches the result threshold dropped.

## When the network misbehaves

Recording never blocks the show. Rows go into a queue that a background
thread uploads in batches every few seconds.

- **Network errors, rate limits and server errors** keep the queue and retry
  with a growing pause. Every row carries a random id, and the platform treats
  a known id as an update that can only make the row longer, so a batch whose
  reply was lost is simply sent again without creating duplicates.
- **A long outage:** the queue holds the most recent 5000 rows and drops the
  oldest beyond that. A row that changes while queued is sent once, in its
  latest state.
- **A rejected batch** (malformed data) is dropped and logged.
- **Images** wait for their row to be saved, then follow the same rules: a
  transient failure is retried, and a rejected image (for instance of a row
  deleted on the platform during the show) is dropped and logged. At most 200
  wait for an unreachable API; beyond that the oldest are dropped.
- **A session that is closed, missing or forbidden** stops the uploads for the
  rest of the run. The show itself carries on.
- **On exit** the card still on stream gets its final time, then the client
  waits up to ten seconds for the last uploads and reports anything it could
  not save.

## Reviewing a session

For now the review is the session API itself (the platform UI is to come):

```bash
curl -H "Authorization: Token $XIMILAR_API_KEY" \
  https://api.ximilar.com/cardstream/v2/session/SESSION_ID/summary/
curl -H "Authorization: Token $XIMILAR_API_KEY" \
  "https://api.ximilar.com/cardstream/v2/identification/?session=SESSION_ID"
curl -H "Authorization: Token $XIMILAR_API_KEY" \
  https://api.ximilar.com/cardstream/v2/session/
```

- **The summary** has the number of cards shown and of distinct cards, the
  paid calls, the total time on stream, the confidence breakdown, the priced
  total, when the first and last card were seen, the most valuable cards and
  the most frequent sets.
- **The identification list** has every row in the order the cards were
  shown, with each card's time on stream and calls, and temporary links to
  its image and thumbnail.
- **The session list** has all your sessions.

## Limitations

- **Messages go to the terminal only.** Session messages go to standard
  output, not to the page's debug panel.
- **Every browser tab shares one session.** In camera mode every connected
  tab analyses its own frames and keeps its own history, but they all save to
  the same session. A tab that disconnects ends the row of the card it showed.
