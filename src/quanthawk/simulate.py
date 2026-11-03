"""A labelled synthetic bank: customers, their habits, and five kinds of fraud.

Real transaction data with fraud labels is not something a bank publishes, and the public
sets that exist either hide the customer (PCA-scrambled card data) or have no labels. So
the benchmark runs on a simulator whose every assumption is written down here and can be
changed. It is built to make the problem *hard in the right ways*:

* Customers differ a lot. A small business moves thousands in and out daily; a student
  spends a few euros at a time. What is normal for one is alarming for the other. Six
  archetypes, and every account also draws its own scale, so customers of one archetype
  differ from each other too.
* Legitimate behaviour is noisy: people travel, buy new phones, pay new people, have a big
  month. Those are the false positives a fraud team drowns in.
* Fraud comes in different shapes (``FRAUD_TYPES``) and some of it is deliberately quiet.
* Some accounts open late, during the test period, which is when mule accounts tend to
  appear.

The archetype of each account is kept in the output for evaluation only. Models never see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DAY = 86400.0
HOUR = 3600.0
CHANNELS = ("pos", "online", "atm", "transfer_out", "transfer_in")
OUTFLOWS = ("pos", "online", "atm", "transfer_out")
FRAUD_TYPES = ("account_takeover", "card_testing", "atm_cashout", "mule", "low_and_slow")
N_CITIES = 40
N_MERCHANTS = 6000
N_MCC = 20
HIGH_RISK_MCC = (17, 18, 19)  # electronics, jewellery, gift cards
ATM_MAX = 500.0


@dataclass(frozen=True)
class Archetype:
    name: str
    share: float
    rates: dict[str, float]  # transactions per day by channel
    medians: dict[str, float]  # median amount by channel
    hours: tuple[float, float]  # centre and spread of activity, hours
    night_owl: float = 0.02  # share of activity at random hours
    weekend: float = 1.0  # weekend activity relative to weekdays
    monthly_in: float = 0.0  # salary or pension, paid on the 1st
    monthly_rent: float = 0.0
    travel_per_month: float = 0.1
    devices: tuple[int, int] = (1, 2)


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype("student", 0.18, {"pos": 1.6, "online": 0.8, "atm": 0.15, "transfer_out": 0.1,
                                "transfer_in": 0.05},
              {"pos": 8, "online": 14, "atm": 20, "transfer_out": 25, "transfer_in": 60},
              (16.0, 4.0), 0.06, 1.2, monthly_in=400, travel_per_month=0.15),
    Archetype("salaried", 0.40, {"pos": 1.8, "online": 0.7, "atm": 0.2, "transfer_out": 0.25,
                                 "transfer_in": 0.03},
              {"pos": 25, "online": 45, "atm": 60, "transfer_out": 120, "transfer_in": 150},
              (14.0, 4.0), 0.02, 1.0, monthly_in=3200, monthly_rent=1100, travel_per_month=0.12),
    Archetype("small_business", 0.08, {"pos": 1.5, "online": 1.2, "atm": 0.1,
                                       "transfer_out": 2.5, "transfer_in": 4.0},
              {"pos": 60, "online": 140, "atm": 150, "transfer_out": 900, "transfer_in": 320},
              (12.5, 2.8), 0.01, 0.35, travel_per_month=0.2, devices=(2, 4)),
    Archetype("retiree", 0.14, {"pos": 1.0, "online": 0.1, "atm": 0.25, "transfer_out": 0.05,
                                "transfer_in": 0.0},
              {"pos": 22, "online": 35, "atm": 80, "transfer_out": 150, "transfer_in": 100},
              (12.0, 2.5), 0.005, 0.9, monthly_in=1600, travel_per_month=0.05, devices=(1, 1)),
    Archetype("high_net_worth", 0.05, {"pos": 1.2, "online": 0.8, "atm": 0.05,
                                       "transfer_out": 0.4, "transfer_in": 0.1},
              {"pos": 160, "online": 320, "atm": 300, "transfer_out": 4500, "transfer_in": 6000},
              (14.0, 4.0), 0.03, 1.1, monthly_in=15000, travel_per_month=0.6, devices=(2, 4)),
    Archetype("gig_worker", 0.15, {"pos": 1.5, "online": 0.6, "atm": 0.2, "transfer_out": 0.15,
                                   "transfer_in": 1.2},
              {"pos": 14, "online": 22, "atm": 40, "transfer_out": 60, "transfer_in": 55},
              (15.0, 6.0), 0.15, 1.3, travel_per_month=0.1),
)
AMOUNT_SIGMA = 0.8


@dataclass(frozen=True)
class SimConfig:
    accounts: int = 3000
    days: int = 120
    seed: int = 0
    fraud_share: float = 0.06  # accounts hit by one fraud episode
    late_open_share: float = 0.05  # accounts opened in the second half
    archetype_mix: dict[str, float] | None = None  # override shares, e.g. {"salaried": 1}
    fraud_mix: dict[str, float] = field(default_factory=lambda: {
        "account_takeover": 0.25, "card_testing": 0.2, "atm_cashout": 0.15, "mule": 0.2,
        "low_and_slow": 0.2})


@dataclass
class _Account:
    id: int
    arch: Archetype
    scale: float
    open_day: float
    home: int
    devices: list[int]
    merchants: np.ndarray
    payees: np.ndarray
    payers: np.ndarray


def _hours(rng: np.random.Generator, arch: Archetype, n: int) -> np.ndarray:
    h = rng.normal(arch.hours[0], arch.hours[1], n)
    owl = rng.random(n) < arch.night_owl
    h[owl] = rng.uniform(0, 24, owl.sum())
    return np.mod(h, 24.0)


class _Builder:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.rows: list[dict[str, Any]] = []
        self.next_device = 1
        self.next_party = 1_000_000
        pop = 1.0 / np.arange(1, N_MERCHANTS + 1) ** 0.9
        self.merchant_p = pop / pop.sum()
        self.merchant_mcc = self.rng.integers(0, N_MCC, N_MERCHANTS)
        # high-risk categories are a small slice of everyday merchants
        risky = np.isin(self.merchant_mcc, HIGH_RISK_MCC)
        self.merchant_mcc[risky & (self.rng.random(N_MERCHANTS) < 0.7)] = 0

    def new_device(self) -> int:
        self.next_device += 1
        return self.next_device

    def new_party(self) -> int:
        self.next_party += 1
        return self.next_party

    def add(self, acc: _Account, ts: float, channel: str, amount: float, *, city: int,
            device: int = 0, merchant: int = -1, party: int = -1, fraud: str = "") -> None:
        mcc = int(self.merchant_mcc[merchant]) if merchant >= 0 else -1
        self.rows.append({
            "ts": float(ts), "account_id": acc.id, "channel": channel,
            "amount": round(float(amount), 2), "merchant": merchant, "mcc": mcc,
            "counterparty": party, "city": city, "device": device,
            "is_fraud": int(bool(fraud)), "fraud_type": fraud, "archetype": acc.arch.name,
        })

    # ---------------------------------------------------------------- accounts

    def make_accounts(self) -> list[_Account]:
        cfg, rng = self.cfg, self.rng
        mix = cfg.archetype_mix
        archs = [a for a in ARCHETYPES if mix is None or a.name in mix]
        shares = np.array([(mix or {}).get(a.name, a.share) for a in archs], float)
        picks = rng.choice(len(archs), cfg.accounts, p=shares / shares.sum())
        accounts = []
        for i, k in enumerate(picks):
            arch = archs[k]
            late = rng.random() < cfg.late_open_share and cfg.days >= 20
            open_day = rng.uniform(cfg.days / 2, cfg.days - 10) if late else 0.0
            nd = int(rng.integers(arch.devices[0], arch.devices[1] + 1))
            accounts.append(_Account(
                id=100_000 + i, arch=arch, scale=float(np.exp(rng.normal(0, 0.45))),
                open_day=open_day, home=int(rng.integers(0, N_CITIES)),
                devices=[self.new_device() for _ in range(nd)],
                merchants=rng.choice(N_MERCHANTS, int(rng.integers(8, 30)), replace=False,
                                     p=self.merchant_p),
                payees=np.array([self.new_party() for _ in range(int(rng.integers(2, 8)))]),
                payers=np.array([self.new_party() for _ in range(int(rng.integers(1, 6)))]),
            ))
        return accounts

    # ---------------------------------------------------------------- normal life

    def normal_life(self, acc: _Account) -> None:
        cfg, rng, arch = self.cfg, self.rng, acc.arch
        start = acc.open_day
        trips: list[tuple[float, float, int]] = []
        t = start
        while True:
            t += rng.exponential(30.0 / max(arch.travel_per_month, 1e-6))
            if t >= cfg.days:
                break
            trips.append((t, t + rng.uniform(2, 6), int(rng.integers(0, N_CITIES))))

        def city_at(ts_day: float) -> int:
            for a, b, c in trips:
                if a <= ts_day < b:
                    return c
            return acc.home

        # a new phone now and then, after which the old one is rarely used
        device_changes = sorted(rng.uniform(start, cfg.days, rng.poisson(0.4 * cfg.days / 120)))
        active_devices = list(acc.devices)

        days = np.arange(int(np.floor(start)), cfg.days)
        for d in days:
            weekday = (d % 7) < 5
            mult = 1.0 if weekday else arch.weekend
            busy = float(np.exp(rng.normal(0, 0.35)))  # some days are busier
            while device_changes and device_changes[0] <= d:
                device_changes.pop(0)
                active_devices.append(self.new_device())
                if len(active_devices) > 1 and rng.random() < 0.7:
                    active_devices.pop(0)
            for ch in CHANNELS:
                n = rng.poisson(arch.rates[ch] * mult * busy)
                if n == 0:
                    continue
                hrs = _hours(rng, arch, n)
                amounts = arch.medians[ch] * acc.scale * np.exp(rng.normal(0, AMOUNT_SIGMA, n))
                for h, amt in zip(hrs, amounts, strict=True):
                    ts = (d + h / 24.0) * DAY
                    if ts < start * DAY:
                        continue
                    city = city_at(ts / DAY)
                    if ch in ("pos", "online"):
                        habitual = rng.random() < 0.85
                        m = int(rng.choice(acc.merchants)) if habitual else int(
                            rng.choice(N_MERCHANTS, p=self.merchant_p))
                        dev = int(rng.choice(active_devices)) if ch == "online" else 0
                        self.add(acc, ts, ch, max(amt, 0.5), city=city, merchant=m, device=dev)
                    elif ch == "atm":
                        self.add(acc, ts, ch, min(max(round(amt / 10) * 10, 10), ATM_MAX),
                                 city=city)
                    elif ch == "transfer_out":
                        habitual = rng.random() < 0.85
                        if not habitual:
                            acc.payees = np.append(acc.payees, self.new_party())
                        p = int(acc.payees[-1] if not habitual else rng.choice(acc.payees))
                        self.add(acc, ts, ch, amt, city=city, party=p,
                                 device=int(rng.choice(active_devices)))
                    else:
                        habitual = rng.random() < 0.8
                        if not habitual:
                            acc.payers = np.append(acc.payers, self.new_party())
                        p = int(acc.payers[-1] if not habitual else rng.choice(acc.payers))
                        self.add(acc, ts, ch, amt, city=city, party=p)
            if d % 30 == 1:  # the 1st of the (30-day) month
                if arch.monthly_in:
                    self.add(acc, (d + 0.3) * DAY, "transfer_in",
                             arch.monthly_in * acc.scale * float(np.exp(rng.normal(0, 0.05))),
                             city=acc.home, party=int(acc.payers[0]))
                if arch.monthly_rent:
                    self.add(acc, (d + 0.4) * DAY, "transfer_out", arch.monthly_rent * acc.scale,
                             city=acc.home, party=int(acc.payees[0]),
                             device=int(active_devices[0]))

    # ---------------------------------------------------------------- fraud

    def fraud(self, acc: _Account, kind: str) -> None:
        cfg, rng, arch = self.cfg, self.rng, acc.arch
        lo = max(acc.open_day + 3, 15)
        if lo >= cfg.days - 3:
            return
        t0 = rng.uniform(lo, cfg.days - 3) * DAY
        night = rng.random() < 0.5
        if night:
            t0 = np.floor(t0 / DAY) * DAY + rng.uniform(0, 5) * HOUR
        inflow = arch.monthly_in * acc.scale or arch.medians["transfer_in"] * acc.scale * 10

        if kind == "account_takeover":
            dev = self.new_device()
            for k in range(int(rng.integers(2, 6))):
                ts = t0 + rng.uniform(0, 2 * HOUR) * (k > 0)
                self.add(acc, ts, "transfer_out", inflow * rng.uniform(0.2, 0.6),
                         city=int(rng.integers(0, N_CITIES)), party=self.new_party(),
                         device=dev, fraud=kind)
        elif kind == "card_testing":
            dev = self.new_device()
            for _ in range(int(rng.integers(4, 12))):
                m = int(rng.integers(0, N_MERCHANTS))
                self.add(acc, t0 + rng.uniform(0, 20 * 60), "online", rng.uniform(0.5, 3.0),
                         city=acc.home, merchant=m, device=dev, fraud=kind)
            risky = np.flatnonzero(np.isin(self.merchant_mcc, HIGH_RISK_MCC))
            for _ in range(int(rng.integers(1, 4))):
                m = int(rng.choice(risky))
                self.add(acc, t0 + rng.uniform(20 * 60, 90 * 60), "online",
                         arch.medians["online"] * acc.scale * rng.uniform(6, 15),
                         city=acc.home, merchant=m, device=dev, fraud=kind)
        elif kind == "atm_cashout":
            city = int((acc.home + rng.integers(1, N_CITIES)) % N_CITIES)
            for _ in range(int(rng.integers(3, 7))):
                self.add(acc, t0 + rng.uniform(0, 3 * HOUR), "atm",
                         float(rng.choice([400.0, 450.0, 500.0])), city=city, fraud=kind)
        elif kind == "mule":
            total = 0.0
            for _ in range(int(rng.integers(2, 6))):
                amt = rng.uniform(400, 3000)
                total += amt
                self.add(acc, t0 + rng.uniform(0, 2 * HOUR), "transfer_in", amt,
                         city=acc.home, party=self.new_party(), fraud=kind)
            for _ in range(int(rng.integers(1, 3))):
                self.add(acc, t0 + rng.uniform(2 * HOUR, 8 * HOUR), "transfer_out",
                         total * rng.uniform(0.3, 0.6), city=acc.home, party=self.new_party(),
                         device=int(acc.devices[0]), fraud=kind)
        elif kind == "low_and_slow":
            for d in range(int(rng.integers(6, 14))):
                ts = t0 + d * DAY + rng.normal(0, 2 * HOUR)
                ch = "online" if rng.random() < 0.6 else "pos"
                self.add(acc, ts, ch, arch.medians[ch] * acc.scale * rng.uniform(2.0, 4.0),
                         city=acc.home, merchant=int(rng.integers(0, N_MERCHANTS)),
                         device=int(acc.devices[0]) if ch == "online" else 0, fraud=kind)

    def build(self) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        accounts = self.make_accounts()
        for acc in accounts:
            self.normal_life(acc)
        kinds = list(cfg.fraud_mix)
        probs = np.array([cfg.fraud_mix[k] for k in kinds], float)
        victims = rng.choice(len(accounts), round(cfg.fraud_share * len(accounts)),
                             replace=False)
        for i in victims:
            acc = accounts[i]
            p = probs.copy()
            if acc.arch.name in ("student", "gig_worker") or acc.open_day > 0:
                p[kinds.index("mule")] *= 3  # mules are recruited young, or open new accounts
            self.fraud(acc, kinds[int(rng.choice(len(kinds), p=p / p.sum()))])
        df = pd.DataFrame(self.rows)
        df = df[df.ts < cfg.days * DAY].sort_values(["ts", "account_id"], kind="stable")
        df.insert(0, "tx_id", np.arange(len(df), dtype=np.int64))
        opened = {a.id: a.open_day * DAY for a in accounts}
        df["account_opened"] = df.account_id.map(opened).astype(float)
        return df.reset_index(drop=True)


def simulate(cfg: SimConfig | None = None) -> pd.DataFrame:
    """One row per transaction, time-ordered. ``ts`` is seconds since the start."""
    return _Builder(cfg or SimConfig()).build()
