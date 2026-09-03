"""The id-type registry — one struct per identification endpoint.

Everything that varies with the id type (``tcg`` / ``sport`` / ``slab`` /
``comics``) lives on one :class:`IdType` and is looked up through
:data:`ID_TYPES`: the endpoint URL, the dropdown label, the Category pair the
endpoint implies, the games/sports it accepts as ``Subcategory``, and
whether it takes the top-level ``price_stats`` request flag.

Before this module those four facts lived in four dicts across two files, all
keyed by the same strings, and adding a category meant editing every one of
them with nothing to catch a miss. Now a new category is one tuple entry.

Stdlib only, so both :mod:`cardstream.core.ximilar` (HTTP + parsing) and
:mod:`cardstream.core.identify_options` can import it without a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

# The dropdown's "no value selected" option. One spelling for the Game and
# Alphabet selects, both identify clients and the settings endpoint — mirrored
# in webui/shared/constants.js and pinned by a test.
NOT_SPECIFIED = "Not Specified"

# Writing systems the id endpoints know, as the record's "Alphabet" value.
# MEASURED, not guessed: prefilling "Subcategory" switches OFF the endpoint's
# own Alphabet classifier, which then falls back to latin — a Japanese card
# then matches its English print. So whenever we prefill a game we must say
# which alphabet it is. The API does NOT validate this value (an unknown
# string is echoed back and quietly treated as non-matching), hence the
# whitelist here.
ALPHABETS = ("latin", "japanese", "chinese", "korean", "thai")


def is_unspecified(value: str | None) -> bool:
    """True when a user-supplied value means "leave this field out".

    ``None``, empty, whitespace-only and the literal "Not Specified" the
    dropdowns send. THE definition — every optional identify field uses it, so
    the three spellings this replaces cannot drift apart again.
    """
    if value is None:
        return True
    return value.strip().casefold() in ("", NOT_SPECIFIED.casefold())


@dataclass(frozen=True)
class IdType:
    """One identification endpoint and everything implied by choosing it."""

    key: str  # "tcg" — the wire/CLI value
    label: str  # "Trading Card Game" — the UI label
    url: str  # the collectibles/v2 endpoint
    # Category attributes the endpoint implies, prefilled on the card object so
    # it skips its own classifier (docs.ximilar.com/collectibles/recognition):
    # tcg_id -> Card/Trading Card Game, sport_id -> Card/Sport Card. slab_id
    # and comics_id only get the top category — slab's Category is per grading
    # company ("Slab Label/PSA", …) and comics' is undocumented, so we don't
    # guess either.
    category_attrs: Mapping[str, str]
    # Display name (UI / --game) -> Ximilar "Subcategory" value. The id
    # endpoints narrow their search when a record carries the game/sport, and
    # the valid values depend on the endpoint. Empty for the types that take
    # none.
    subcategories: Mapping[str, str]
    # Whether the endpoint honours the top-level ``price_stats`` request flag
    # (market price statistics on the best match). Documented for tcg_id,
    # sport_id and comics_id; slab_id has no prices to aggregate.
    # IdentifyOptions.payload() sends the flag only where this is True.
    price_stats: bool

    # Tolerant index: display name OR API value, any case -> display name.
    _lookup: Mapping[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        lookup: dict[str, str] = {}
        for display, api in self.subcategories.items():
            lookup[display.casefold()] = display
            lookup[api.casefold()] = display
        # Freeze for real: a plain dict on a frozen dataclass is still mutable,
        # and these are process-global.
        object.__setattr__(
            self, "category_attrs", MappingProxyType(dict(self.category_attrs))
        )
        object.__setattr__(
            self, "subcategories", MappingProxyType(dict(self.subcategories))
        )
        object.__setattr__(self, "_lookup", MappingProxyType(lookup))

    def normalize_game(self, game: str | None) -> str | None:
        """Canonical display name for ``game``; ``None`` when not specified.

        Accepts the display name ("Pokémon") or the Ximilar value ("Pokemon")
        in any case. Raises ``ValueError`` (listing what IS valid for this
        type) on anything else — a game the endpoint doesn't know is a bad
        hint that quietly costs match quality, so it has to fail here.
        """
        if is_unspecified(game):
            return None
        assert game is not None  # is_unspecified covered None
        key = game.strip().casefold()
        if key not in self._lookup:
            valid = ", ".join(self.subcategories) or "(none for this category)"
            raise ValueError(
                f"unknown game {game!r} for {self.key}; valid: {valid} (or leave unset)"
            )
        return self._lookup[key]

    def subcategory_for(self, game: str | None) -> str | None:
        """Ximilar ``Subcategory`` value for a game; ``None`` when unspecified."""
        display = self.normalize_game(game)
        return self.subcategories[display] if display else None

    @property
    def game_choices(self) -> list[str]:
        """The Game select's options, "Not Specified" first."""
        return [NOT_SPECIFIED, *self.subcategories]


ID_TYPES: Mapping[str, IdType] = MappingProxyType(
    {
        t.key: t
        for t in (
            IdType(
                key="tcg",
                label="Trading Card Game",
                url="https://api.ximilar.com/collectibles/v2/tcg_id",
                category_attrs={
                    "Top Category": "Card",
                    "Category": "Card/Trading Card Game",
                },
                subcategories={
                    "Pokémon": "Pokemon",
                    "Magic The Gathering": "Magic The Gathering",
                    "One Piece": "One Piece",
                },
                price_stats=True,
            ),
            IdType(
                key="sport",
                label="Sport Card",
                url="https://api.ximilar.com/collectibles/v2/sport_id",
                category_attrs={"Top Category": "Card", "Category": "Card/Sport Card"},
                subcategories={
                    "Baseball": "Baseball",
                    "Basketball": "Basketball",
                    "Football": "Football",
                    "Hockey": "Hockey",
                    "Soccer": "Soccer",
                    "MMA": "MMA",
                },
                price_stats=True,
            ),
            IdType(
                key="slab",
                label="Slab Label",
                url="https://api.ximilar.com/collectibles/v2/slab_id",
                category_attrs={"Top Category": "Slab Label"},
                subcategories={},
                price_stats=False,
            ),
            IdType(
                key="comics",
                label="Comics",
                url="https://api.ximilar.com/collectibles/v2/comics_id",
                category_attrs={"Top Category": "Comics"},
                subcategories={},
                price_stats=True,
            ),
        )
    }
)


def resolve_id_type(key: str | IdType | None, *, noun: str = "id type") -> IdType:
    """Look up an id type by key; THE validation site.

    ``noun`` only picks the wording, so each surface keeps the vocabulary its
    users see — "type" for ``?type=``, "category" for the settings dialog —
    without a second implementation behind it.
    """
    if isinstance(key, IdType):
        return key
    name = (key or "").strip().lower()
    if name not in ID_TYPES:
        raise ValueError(f"unknown {noun} {key!r}; valid: {', '.join(ID_TYPES)}")
    return ID_TYPES[name]


def normalize_set_code(set_code: str | None) -> str | None:
    """Canonical record ``set_code`` (e.g. ``"PBL"``); ``None`` when unset.

    Anything specified is passed through verbatim — Ximilar set codes are not
    all upper-case (``sv3pt5``).
    """
    if is_unspecified(set_code):
        return None
    assert set_code is not None
    return set_code.strip()


def normalize_alphabet(alphabet: str | None) -> str | None:
    """Canonical ``Alphabet`` value; ``None`` when unset (endpoint detects it).

    Raises ``ValueError`` on anything outside :data:`ALPHABETS` — the API
    accepts unknown values silently and then matches the wrong print, so a
    typo has to fail here or nowhere.
    """
    if is_unspecified(alphabet):
        return None
    assert alphabet is not None
    value = alphabet.strip().lower()
    if value not in ALPHABETS:
        raise ValueError(
            f"unknown alphabet {alphabet!r}; valid: {', '.join(ALPHABETS)}"
        )
    return value
