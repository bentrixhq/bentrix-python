"""**A refused order, explained**: the venue's words, what to change, and the page of the Docs.

A refused order's answer carries, beside the venue's own sentence (``reason``), why as a program
reads it (``kind``), one sentence of remedy (``remedy``) and the Docs' page for it (``docs``) - the
application's, so every language reads the same. ``explain`` puts them in one line; a program
branches on ``kind``:

    book = local.book(bench)
    if book["last_answer"]["outcome"] == "refused":
        print(bentrix.explain(book))
        if book["last_answer"]["kind"] == "volume_out_of_range": ...

The kinds: ``unknown_market``, ``trade_disabled``, ``volume_out_of_range``, ``no_quote``,
``market_closed``, ``no_conversion_rate``, ``stops_wrong_side``, ``stops_too_close``,
``stops_invalid`` (a terminal's, which does not say which of the two), ``price_wrong_side``,
``price_too_close``, ``invalid_price``, ``no_money``, ``position_gone``, ``frozen``, ``expired``,
``autotrading_disabled``, ``rejected`` (as a broker may reject an order: the workbench's execution model, or a
server's), ``days_over`` (the workbench's days, or its recording, end here: nothing is left to fill) and
``other``. A request that never reached a venue has no kind.
"""

from __future__ import annotations

from typing import Any, Mapping

KINDS = (
    "unknown_market",
    "trade_disabled",
    "volume_out_of_range",
    "no_quote",
    "market_closed",
    "no_conversion_rate",
    "stops_wrong_side",
    "stops_too_close",
    "stops_invalid",
    "price_wrong_side",
    "price_too_close",
    "invalid_price",
    "no_money",
    "position_gone",
    "frozen",
    "expired",
    "autotrading_disabled",
    "rejected",
    "days_over",
    "other",
)


def explain(answer: Any) -> str:
    """**One line for a refused order**: its words, what to change, and where the Docs explain it - from a
    book (its ``last_answer``), an answer itself, or a tool's refusal raised as an exception. ``""`` for an
    answer that is not a refusal."""
    if isinstance(answer, BaseException):
        return str(answer)
    if not isinstance(answer, Mapping):
        raise TypeError("explain takes an answer, a book or a refusal")
    # An answer that is itself a refusal speaks for itself: one refused before it reached the venue (the days are over)
    # leaves the book's last_answer on the earlier order.
    own = answer.get("outcome") == "refused" and answer.get("reason")
    last = answer if own or "last_answer" not in answer else answer["last_answer"]
    if not last or last.get("outcome") != "refused":
        return ""
    words = str(last.get("reason") or "refused").rstrip(".")
    parts = [words + "."]
    if last.get("remedy"):
        parts.append(last["remedy"])
    if last.get("docs"):
        parts.append(f"See {last['docs']}")
    return " ".join(parts)
