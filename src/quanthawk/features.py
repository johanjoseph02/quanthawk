"""Point-in-time features, computed one transaction at a time.

There is a single implementation, ``FeatureEngine.update``, used both to build training
data (by replaying history in time order) and to score live transactions. A feature can
therefore never see the future in training while being unavailable in production, which
is the most common way fraud models look good offline and fail live.

Every feature describes the transaction *relative to this account's own past*: how
large compared with what this customer usually spends on this channel, how fast compared
with their usual pace, whether the device, payee, merchant, city or category is new to
them, how unusual the hour is for them.

``AccountState.profile`` summarises an account's behaviour as a whole (its pace, channel
mix, typical amounts, hours) for the customer segmentation in ``quanthawk.model``.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

DAY = 86400.0
HOUR = 3600.0
CHANNELS = ("pos", "online", "atm", "transfer_out", "transfer_in")
OUTFLOWS = frozenset({"pos", "online", "atm", "transfer_out"})

FEATURES: tuple[str, ...] = (
    "log_amount",
    "amount_z",          # vs this account's usual amount on this channel
    "amount_vs_max",     # vs the largest amount this account has used on this channel
    "n_1h",              # transactions in the last hour, this one included
    "n_24h",
    "out_24h_vs_max",    # money out in the last 24 hours vs the largest earlier payment out
    "new_in_24h_vs_max", # money in from *new* payers in the last 24 hours vs the largest
                         # earlier payment in: a salary from the usual employer does not count
    "passthrough",       # share of the last day's money from new payers already sent on
    "log_gap",           # log seconds since this account's previous transaction
    "new_device",
    "new_counterparty",  # payee, payer or merchant never seen on this account
    "new_city",
    "new_mcc",
    "new_cp_24h",        # new counterparties in the last 24 hours
    "hour_rarity",       # -log P(hour) under this account's own hour histogram
    "log_account_age",
    *(f"is_{c}" for c in CHANNELS),
)

PROFILE: tuple[str, ...] = (
    "log_tx_per_day",
    *(f"share_{c}" for c in CHANNELS),
    "mean_log_out",
    "mean_log_in",
    "night_share",
    "weekend_share",
    "log_cities",
    "log_devices",
)
MIN_PROFILE_TX = 25
MIN_PROFILE_DAYS = 7.0


@dataclass
class _Ew:
    """Exponentially weighted mean and variance of log amounts."""

    n: int = 0
    mean: float = 0.0
    var: float = 0.0
    max: float = 0.0

    def update(self, x: float, amount: float, alpha: float = 0.05) -> None:
        self.n += 1
        self.max = max(self.max, amount)
        if self.n == 1:
            self.mean, self.var = x, 0.0
            return
        a = max(alpha, 1.0 / self.n)
        d = x - self.mean
        self.mean += a * d
        self.var = (1 - a) * (self.var + a * d * d)


@dataclass
class AccountState:
    opened: float | None = None
    first_ts: float | None = None
    last_ts: float | None = None
    n: int = 0
    by_channel: dict[str, _Ew] = field(default_factory=dict)
    overall: _Ew = field(default_factory=_Ew)
    window: deque[tuple[float, float, bool, bool]] = field(default_factory=deque)
    devices: set[int] = field(default_factory=set)
    parties: set[int] = field(default_factory=set)
    cities: set[int] = field(default_factory=set)
    mccs: set[int] = field(default_factory=set)
    hours: list[int] = field(default_factory=lambda: [0] * 24)
    channel_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CHANNELS, 0))
    max_out: float = 0.0  # largest single payment out, among those older than 24 hours
    max_in: float = 0.0
    sum_log_out: float = 0.0
    n_out: int = 0
    sum_log_in: float = 0.0
    n_in: int = 0
    night: int = 0
    weekend: int = 0

    def profile(self) -> NDArray[np.float64] | None:
        """Behavioural summary, or None while the account is too new to have one."""
        if self.n < MIN_PROFILE_TX or self.first_ts is None or self.last_ts is None:
            return None
        span = (self.last_ts - self.first_ts) / DAY
        if span < MIN_PROFILE_DAYS:
            return None
        return np.array([
            math.log(self.n / span),
            *(self.channel_counts[c] / self.n for c in CHANNELS),
            self.sum_log_out / max(self.n_out, 1),
            self.sum_log_in / self.n_in if self.n_in else 0.0,
            self.night / self.n,
            self.weekend / self.n,
            math.log1p(len(self.cities)),
            math.log1p(len(self.devices)),
        ])


@dataclass(frozen=True)
class Tx:
    ts: float
    account_id: int
    channel: str
    amount: float
    merchant: int = -1
    mcc: int = -1
    counterparty: int = -1
    city: int = -1
    device: int = 0
    account_opened: float | None = None

    @classmethod
    def from_mapping(cls, m: Mapping[Any, Any]) -> Tx:
        opened = m.get("account_opened")
        return cls(
            ts=float(m["ts"]), account_id=int(m["account_id"]), channel=str(m["channel"]),
            amount=float(m["amount"]), merchant=int(m.get("merchant", -1)),
            mcc=int(m.get("mcc", -1)), counterparty=int(m.get("counterparty", -1)),
            city=int(m.get("city", -1)), device=int(m.get("device", 0)),
            account_opened=None if opened is None or opened != opened else float(opened),
        )


class FeatureEngine:
    def __init__(self) -> None:
        self.accounts: dict[int, AccountState] = {}

    def state(self, account_id: int) -> AccountState:
        st = self.accounts.get(account_id)
        if st is None:
            st = self.accounts[account_id] = AccountState()
        return st

    def update(self, tx: Tx) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        """Features of ``tx`` given everything before it, and the account's profile as it
        stood before ``tx``. Then fold ``tx`` into the account's state."""
        if tx.channel not in CHANNELS:
            raise ValueError(f"unknown channel {tx.channel!r}")
        st = self.state(tx.account_id)
        profile = st.profile()
        if st.opened is None:
            st.opened = tx.account_opened if tx.account_opened is not None else tx.ts
        ts, amt = tx.ts, max(tx.amount, 0.01)
        x = math.log(amt)
        is_out = tx.channel in OUTFLOWS

        ch = st.by_channel.get(tx.channel)
        ref = ch if ch is not None and ch.n >= 3 else (st.overall if st.overall.n >= 3 else None)
        amount_z = 0.0 if ref is None else (x - ref.mean) / math.sqrt(ref.var + 0.25)
        amount_vs_max = 0.0 if ch is None or ch.n == 0 else math.log(amt / (ch.max + 1.0))

        party = tx.counterparty if tx.counterparty >= 0 else tx.merchant
        new_party = party >= 0 and party not in st.parties and st.n > 0
        new_device = tx.device > 0 and tx.device not in st.devices and st.n > 0
        new_city = tx.city >= 0 and tx.city not in st.cities and st.n > 0
        new_mcc = tx.mcc >= 0 and tx.mcc not in st.mccs and st.n > 0

        while st.window and st.window[0][0] <= ts - DAY:
            _, old_amt, old_out, _ = st.window.popleft()
            if old_out:
                st.max_out = max(st.max_out, old_amt)
            else:
                st.max_in = max(st.max_in, old_amt)
        st.window.append((ts, amt, is_out, new_party))
        n_1h = sum(1 for w in st.window if w[0] > ts - HOUR)
        out_24h = sum(w[1] for w in st.window if w[2])
        new_in_24h = sum(w[1] for w in st.window if not w[2] and w[3])
        new_cp_24h = sum(1 for w in st.window if w[3])
        passthrough = min(out_24h, new_in_24h) / new_in_24h if new_in_24h > 0 and is_out else 0.0

        hour = int((ts % DAY) // HOUR)
        hour_rarity = -math.log((st.hours[hour] + 0.5) / (st.n + 12.0))
        log_gap = math.log1p(ts - st.last_ts) if st.last_ts is not None else math.log1p(30 * DAY)
        age = math.log1p(max(ts - (st.opened or ts), 0.0) / DAY)

        # Relative to the largest single payments from before the last day. Inflows count
        # only from new payers, so a monthly salary from the usual employer stays ordinary
        # while a stranger's deposit into a student's account does not.
        out_rel = math.log1p(out_24h) - math.log1p(st.max_out)
        in_rel = math.log1p(new_in_24h) - math.log1p(st.max_in)

        feats = np.array([
            x, amount_z, amount_vs_max, n_1h, len(st.window), out_rel, in_rel, passthrough,
            log_gap, float(new_device), float(new_party),
            float(new_city), float(new_mcc), new_cp_24h, hour_rarity, age,
            *(float(tx.channel == c) for c in CHANNELS),
        ])

        # fold the transaction in
        if ch is None:
            ch = st.by_channel[tx.channel] = _Ew()
        ch.update(x, amt)
        st.overall.update(x, amt)
        st.n += 1
        st.first_ts = ts if st.first_ts is None else st.first_ts
        st.last_ts = ts
        if tx.device > 0:
            st.devices.add(tx.device)
        if party >= 0:
            st.parties.add(party)
        if tx.city >= 0:
            st.cities.add(tx.city)
        if tx.mcc >= 0:
            st.mccs.add(tx.mcc)
        st.hours[hour] += 1
        st.channel_counts[tx.channel] += 1
        if is_out:
            st.sum_log_out += x
            st.n_out += 1
        else:
            st.sum_log_in += x
            st.n_in += 1
        st.night += hour < 6
        st.weekend += int(ts // DAY) % 7 >= 5
        return feats, profile


def featurize(rows: Iterable[Mapping[Any, Any]], engine: FeatureEngine | None = None
              ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_],
                         FeatureEngine]:
    """Replay time-ordered rows. Returns (features, profiles, has_profile, engine)."""
    engine = engine or FeatureEngine()
    feats: list[NDArray[np.float64]] = []
    profiles: list[NDArray[np.float64]] = []
    has: list[bool] = []
    empty = np.zeros(len(PROFILE))
    for r in rows:
        f, p = engine.update(Tx.from_mapping(r))
        feats.append(f)
        profiles.append(empty if p is None else p)
        has.append(p is not None)
    n = len(feats)
    return (np.vstack(feats) if n else np.zeros((0, len(FEATURES))),
            np.vstack(profiles) if n else np.zeros((0, len(PROFILE))),
            np.array(has, dtype=bool), engine)
