"""What the id endpoint is asked — one value object, normalized once.

The bundle ``(id_type, game, set_code, known_attrs, alphabet, price_stats)``
used to be rebuilt by hand in several places (the identify client, the web
client's settings endpoint and the CLI wiring), each re-implementing the same
rules:
what "not specified" means, and that switching category has to drop a game the
new category doesn't know.

Now they all construct or patch one frozen :class:`IdentifyOptions`, and the
two wire forms it has to survive — the Ximilar record built by :meth:`record`
and the POST body built by :meth:`payload` — are derived from it in a single
place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cardstream.core.id_types import (
    IdType,
    normalize_alphabet,
    normalize_set_code,
    resolve_id_type,
)

_FIELDS = ("id_type", "game", "set_code", "known_attrs", "alphabet", "price_stats")


@dataclass(frozen=True)
class IdentifyOptions:
    """Every prefill one identify call carries, validated at construction."""

    # Which endpoint to call; also decides the record's Category pair and which
    # games are valid. A plain key ("tcg") is accepted and resolved.
    id_type: IdType = "tcg"  # type: ignore[assignment]
    # Canonical DISPLAY name of the game/sport ("Pokémon"), None = not sent.
    game: str | None = None
    # Restricts matching to one set; None = any set.
    set_code: str | None = None
    # False = don't assert Side/Rotation, so backs and rotated cards are
    # classified by the endpoint instead of mis-asserted by us.
    known_attrs: bool = True
    # Writing system; None = don't send the field and let the endpoint
    # classify it. Worth setting whenever a game is prefilled: the game
    # prefill switches the endpoint's own alphabet classifier off and it then
    # assumes latin, so a Japanese card matches its English print.
    alphabet: str | None = None
    # Ask the endpoint for market price statistics with every match (USD:
    # median, range, latest sale). Off by default — the extra data is not
    # documented as free. Sent only to the id types that take the flag
    # (IdType.price_stats); the preference itself survives a category
    # switch, so tcg → slab → tcg does not silently turn it off.
    price_stats: bool = False

    def __post_init__(self) -> None:
        id_type = resolve_id_type(self.id_type)
        object.__setattr__(self, "id_type", id_type)
        object.__setattr__(self, "game", id_type.normalize_game(self.game))
        object.__setattr__(self, "set_code", normalize_set_code(self.set_code))
        object.__setattr__(self, "known_attrs", bool(self.known_attrs))
        object.__setattr__(self, "alphabet", normalize_alphabet(self.alphabet))
        object.__setattr__(self, "price_stats", bool(self.price_stats))

    @property
    def subcategory(self) -> str | None:
        """The record's ``Subcategory`` value for :attr:`game`."""
        return self.id_type.subcategories[self.game] if self.game else None

    def with_(self, **patch: Any) -> IdentifyOptions:
        """Return a copy with the given fields replaced; absent keys unchanged.

        THE place the cross-field rule lives: changing ``id_type`` drops a game
        the new category doesn't know (Pokémon on sport_id) rather than sending
        it as a bad hint — but an explicitly patched game is validated against
        the NEW category and raises, because that one is the user's mistake,
        not a leftover.
        """
        if unknown := set(patch) - set(_FIELDS):
            raise ValueError(f"unknown option {', '.join(sorted(unknown))!r}")
        id_type = resolve_id_type(patch.get("id_type", self.id_type))
        if "game" in patch:
            game = patch["game"]
        else:
            try:
                game = id_type.normalize_game(self.game)
            except ValueError:
                game = None
        return IdentifyOptions(
            id_type=id_type,
            game=game,
            set_code=patch.get("set_code", self.set_code),
            known_attrs=patch.get("known_attrs", self.known_attrs),
            alphabet=patch.get("alphabet", self.alphabet),
            price_stats=patch.get("price_stats", self.price_stats),
        )

    def record(self, b64: str, objects: list[dict[str, Any]]) -> dict[str, Any]:
        """Build an id-endpoint record: image + objects plus every attribute we
        already know — the category pair implied by the id type, the game as
        ``Subcategory`` and the writing system as ``Alphabet`` (when set), and,
        with ``known_attrs``, the assertion that the card is front-side and
        upright. Known attributes let the endpoint skip its own classifiers and
        narrow the search (docs.ximilar.com/collectibles/recognition).

        Identification runs per detected object, so the attributes ride on
        ``objects[0]`` (the card being identified), not on the record itself —
        except ``set_code``, which sits on the RECORD and restricts matching to
        that one set.
        """
        attrs: dict[str, Any] = dict(self.id_type.category_attrs)
        if self.known_attrs:
            attrs["Side"] = "front"
            attrs["Rotation"] = "rotation_ok"
        if self.subcategory:
            attrs["Subcategory"] = self.subcategory
        # Verified equivalent on the live endpoint whether it rides on the
        # record (as the docs show) or on the object with the rest — kept
        # together with the other attributes.
        if self.alphabet:
            attrs["Alphabet"] = self.alphabet
        # Copy, don't mutate the caller's objects.
        merged = [{**objects[0], **attrs}, *objects[1:]] if objects else []
        record: dict[str, Any] = {"_base64": b64, "_objects": merged}
        if self.set_code:
            record["set_code"] = self.set_code
        return record

    def payload(self, record: dict[str, Any]) -> dict[str, Any]:
        """THE POST body: the ``records`` envelope plus the top-level request
        flags. ``price_stats`` is a preference; whether it goes on the wire is
        the id type's call (``IdType.price_stats``), so an endpoint that does
        not document the flag never receives it.
        """
        body: dict[str, Any] = {"records": [record]}
        if self.price_stats and self.id_type.price_stats:
            body["price_stats"] = True
        return body
