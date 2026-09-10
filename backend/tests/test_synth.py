"""Synthetic scenario generator: schema conformance, rate, ramp, determinism."""
from __future__ import annotations

import random

import pytest

from app.sentinel_infer.forecast import REQUIRED as INFER_REQUIRED
from app.synth import scenarios as S


ALL = sorted(S.SCENARIOS)


@pytest.mark.parametrize("name", ALL)
def test_rows_conform_to_required_schema(name):
    cfg = S.build_config(name, rate=30, duration_s=60, seed=1)
    rows = S.emit(0.0, 10.0, cfg, random.Random(1))
    assert rows, "generator produced nothing"
    for r in rows:
        for col in INFER_REQUIRED:
            assert col in r, f"{name}: row missing {col!r}"
        assert 0.0 <= r["flow_start_epoch"] < 10.0
        assert r["Protocol"] in (6, 17)
        for f in ("fin", "syn", "rst", "psh", "ack", "urg"):
            assert r[f"flag_true_{f}"] in (0, 1)
    epochs = [r["flow_start_epoch"] for r in rows]
    assert epochs == sorted(epochs)


@pytest.mark.parametrize("name", ALL)
def test_benign_rate_in_band_at_start(name):
    # first 10s is always pre-ramp (phase "benign") -> ~pure baseline
    cfg = S.build_config(name, rate=40, duration_s=300, seed=7)
    rows = S.emit(0.0, 10.0, cfg, random.Random(7))
    assert 10 * 40 * 0.6 <= len(rows) <= 10 * 40 * 1.4


@pytest.mark.parametrize("name", [n for n in ALL if n != "benign"])
def test_attack_fraction_non_decreasing_across_ramp(name):
    cfg = S.build_config(name, rate=25, duration_s=120, seed=3)
    rng = random.Random(3)
    rows = S.emit(0.0, cfg.duration_s, cfg, rng)
    thirds = [[], [], []]
    for r in rows:
        thirds[min(2, int(r["flow_start_epoch"] / cfg.duration_s * 3))].append(r)

    def atk_frac(chunk):
        if not chunk:
            return 0.0
        return sum(1 for r in chunk if r["attack_family"] != "BENIGN") / len(chunk)

    f0, f1, f2 = (atk_frac(c) for c in thirds)
    # ramp is non-decreasing (small tolerance for per-bucket Poisson noise)
    assert f0 <= f1 + 0.05
    assert f1 <= f2 + 0.05
    assert f0 < f2, "attack fraction did not grow from first to last third"
    assert f2 > 0.0


def test_seed_reproducible():
    cfg = S.build_config("portscan", rate=20, duration_s=40, seed=99)
    a = S.emit(0.0, 40.0, cfg, random.Random(99))
    b = S.emit(0.0, 40.0, cfg, random.Random(99))
    assert a == b


def test_build_config_clamps_and_validates():
    from app.settings import settings
    assert S.build_config("portscan", rate=10 ** 9).rate == float(settings.synth_max_rate)
    assert S.build_config("portscan", rate=0).rate == 1.0
    with pytest.raises(KeyError):
        S.build_config("nope")
