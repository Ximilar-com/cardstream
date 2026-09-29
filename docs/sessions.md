# Saving a show to Ximilar

By default nothing is kept: the history on the page is gone when the tab
closes, and the identify call is the only thing that leaves your machine.
`--ximilar-stream` opts in to one more thing. Every identification is also
saved to a **session** on the Ximilar platform, so you can come back after the
show and see what was shown, when, how sure each match was and what it was
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
[session] 212 identification(s) saved to 0b7c9a52-… — session closed
```

## Requirements

- The same `XIMILAR_API_KEY` the identify call uses. The session belongs to
  that key's account (its default workspace).
- That account needs the **cardstream** service. Without it, the start fails
  with the API's own explanation, before the show begins.

## The flags

| Flag | What it does |
| --- | --- |
| `--ximilar-stream NEW` | start a new session |
| `--ximilar-stream ID` | resume a live session, e.g. after restarting the client mid-show |
| `--ximilar-stream-name NAME` | name a new session; otherwise it is named after the game (or card type) and the start time |
| `--ximilar-stream-platform` | where the show streams: `whatnot`, `tiktok`, `ebay`, `fanatics`, `youtube`, `twitch` or `other` |
| `--ximilar-stream-keep-open` | leave the session live on exit, so the next run can resume it |
| `--ximilar-stream-url URL` | the session API base URL, only for a development backend |

The name and platform describe a **new** session; passing them with an ID is
refused. Any of the other session flags without `--ximilar-stream` is refused
too, rather than silently doing nothing.

A closed session cannot be resumed. If you plan to restart the client during a
show (to change a flag the settings dialog does not cover), run with
`--ximilar-stream-keep-open`, then resume with the printed ID.

## What is sent

One record per identification that survives `--result-threshold`: exactly the
matches the page shows. For each record:

- **When:** the time the identify call fired.
- **Card:** the category (`tcg`, `sport`, `slab`, `comics`), name, full name,
  set name and code, card number, series, year and subcategory.
- **Match:** the match distance and confidence tier, the links and up to four
  alternatives.
- **Prices:** the price statistics, when `--price-stats` is on. The platform
  derives a representative price from them.

At the start, the session is described by its name, game, platform and the
client's settings (version, card type, set code, alphabet and whether prices
are on).

What is **not** sent: images, frames or video, anything from the platform you
stream on (buyers, sales, chat), and matches the result threshold dropped.

## When the network misbehaves

Recording never blocks the show. Identifications go into a queue that a
background thread uploads in batches every few seconds.

- **Network errors, rate limits and server errors** keep the queue and retry
  with a growing pause. Every record carries a random id, and the platform
  skips ids it already has, so a batch whose reply was lost is simply sent
  again without creating duplicates.
- **A long outage:** the queue holds the most recent 5000 identifications and
  drops the oldest beyond that.
- **A rejected batch** (malformed data) is dropped and logged.
- **A session that is closed, missing or forbidden** stops the uploads for the
  rest of the run. The show itself carries on.
- **On exit** the client waits up to ten seconds for the last uploads, then
  reports anything it could not save.

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

- **The summary** has the number of identifications and of distinct cards, the
  confidence breakdown, the priced total, when the first and last card were
  seen, the most valuable cards and the most frequent sets.
- **The identification list** has every record in the order the cards were
  seen.
- **The session list** has all your sessions.

## Limitations

- **How long a card stayed on stream is not recorded.** The history row on the
  page times it; the session has the moment it was identified.
- **Messages go to the terminal only.** Session messages go to standard
  output, not to the page's debug panel.
- **Every browser tab shares one session.** In camera mode every connected
  tab analyses its own frames, but they all save to the same session.
