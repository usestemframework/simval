"""Spec section 22 modal audio reference tests."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

from simval.ontos_audio import (
    AUDIO_SR,
    RING,
    TAIL_BLOCKS,
    audio_hash,
    collect_excitations,
    synthesize,
    synthesize_stream,
    wav_bytes,
)
from simval.ontos_gravity import parse_stream_v2

CONTACT_EXAMPLES = Path(__file__).parent.parent / "examples" / "ontos_contact"


def test_silence_is_zero_pcm():
    pcm = synthesize([], 3)
    assert len(pcm) == (3 + TAIL_BLOCKS) * 64 * 2
    assert pcm == bytes(len(pcm))
    assert audio_hash(pcm) == audio_hash(bytes(len(pcm)))


def test_known_excitation_pins_hash():
    mu_a = (1.25 * 0.75) / (1.25 + 0.75)
    mu_b = (2.0 * 1.5) / (2.0 + 1.5)
    pcm = synthesize([(5, mu_a, 0.3), (40, mu_b, 0.05)], 100)
    assert audio_hash(pcm) == 0xAF22AA8908656DBD
    assert pcm == synthesize([(5, mu_a, 0.3), (40, mu_b, 0.05)], 100)


def test_ring_decays_to_silence():
    pcm = synthesize([(0, (1.0 * 1.0) / (1.0 + 1.0), 0.8)], 300)
    peak = max(abs(struct.unpack_from("<h", pcm, i)[0]) for i in range(0, len(pcm), 2))
    assert peak > 8000
    start = (64 + RING) * 2
    tail = struct.unpack_from("<64h", pcm, start)
    assert all(s == 0 for s in tail)


def test_wav_container_layout():
    wav = wav_bytes(struct.pack("<4h", 0, 1, -1, 32767))
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert wav[36:40] == b"data"
    assert struct.unpack_from("<I", wav, 4)[0] == 44
    assert struct.unpack_from("<I", wav, 40)[0] == 8
    assert struct.unpack_from("<I", wav, 24)[0] == AUDIO_SR
    assert len(wav) == 52


def test_reference_matches_recorded_wav():
    pcm, digest = synthesize_stream(CONTACT_EXAMPLES / "contact" / "ontos.stream")
    recorded = (CONTACT_EXAMPLES / "contact" / "ontos.wav").read_bytes()
    assert wav_bytes(pcm) == recorded
    assert digest == 0xEE83454527DBE43B


def test_reference_matches_composition_wav():
    pcm, digest = synthesize_stream(
        CONTACT_EXAMPLES / "contact_collapse" / "ontos.stream"
    )
    recorded = (CONTACT_EXAMPLES / "contact_collapse" / "ontos.wav").read_bytes()
    assert wav_bytes(pcm) == recorded
    assert digest == 0x85B6B183008818DF


def test_excitations_use_tick_body_masses():
    _, records = parse_stream_v2(CONTACT_EXAMPLES / "contact" / "ontos.stream")
    excitations, final_tick = collect_excitations(records)
    assert final_tick == 400
    assert len(excitations) == 5
    first = excitations[0]
    assert first[0] == 1
    assert first[1] > 0.0 and first[2] > 0.0


def test_cli_expect_mismatch_fails(tmp_path):
    from simval.ontos_audio import _main

    stream = CONTACT_EXAMPLES / "contact" / "ontos.stream"
    assert _main([str(stream), "--expect", "0" * 16]) == 1
    assert _main([str(stream), "--wav", str(tmp_path / "a.wav")]) == 0
    assert (tmp_path / "a.wav").read_bytes() == (
        CONTACT_EXAMPLES / "contact" / "ontos.wav"
    ).read_bytes()


def test_walls_example_wav_matches():
    # Includes two collapsed-region monopole contacts: the pseudo-id mu
    # rule (reduced mass vs the RegionCollapsed mass) must hold for the
    # resynthesis to stay bit-identical.
    pcm, digest = synthesize_stream(CONTACT_EXAMPLES / "walls" / "ontos.stream")
    recorded = (CONTACT_EXAMPLES / "walls" / "ontos.wav").read_bytes()
    assert wav_bytes(pcm) == recorded
    assert digest == 0x70A1D0539170B265


def test_restitution_example_wav_matches():
    pcm, digest = synthesize_stream(CONTACT_EXAMPLES / "restitution" / "ontos.stream")
    recorded = (CONTACT_EXAMPLES / "restitution" / "ontos.wav").read_bytes()
    assert wav_bytes(pcm) == recorded
    assert digest == 0xC4317E51BCE80E83


def test_wallshot_example_wav_matches():
    # Wall-hit corpus (test-only wallshot ICs): 16 wall contacts, mu = m_a
    # per the section 24 pseudo-id rule.
    pcm, digest = synthesize_stream(CONTACT_EXAMPLES / "wallshot" / "ontos.stream")
    recorded = (CONTACT_EXAMPLES / "wallshot" / "ontos.wav").read_bytes()
    assert wav_bytes(pcm) == recorded
    assert digest == 0x8B5D1286CFA35900


def test_coarsehit_example_wav_matches():
    # Fine x ephemeris-coarse static-contact corpus (test-only coarsehit
    # ICs): real-id static pairs use the section 22 reduced mass.
    pcm, digest = synthesize_stream(CONTACT_EXAMPLES / "coarsehit" / "ontos.stream")
    recorded = (CONTACT_EXAMPLES / "coarsehit" / "ontos.wav").read_bytes()
    assert wav_bytes(pcm) == recorded
    assert digest == 0x11EADC8A83DF194E


def test_monopole_mu_uses_collapse_mass_at_contact_time():
    from simval.ontos_gravity import MONOPOLE_BASE, parse_stream_v2

    _, records = parse_stream_v2(CONTACT_EXAMPLES / "walls" / "ontos.stream")
    excitations, _ = collect_excitations(records)
    masses = {}
    collapse_mass = {}
    contacts = []
    for record in records:
        if record[0] == "body":
            masses[record[2]] = record[9]
        elif record[0] == "collapsed":
            collapse_mass[record[3] * 2 + record[2]] = record[5]
        elif record[0] == "contact":
            contacts.append((record, dict(collapse_mass)))
    want = []
    for record, mass_at_contact in contacts:
        _, tick, a, b, jn, _cx, _cy = record
        ma = masses[a]
        if b >= MONOPOLE_BASE:
            m = mass_at_contact[b - MONOPOLE_BASE]
            mu = (ma * m) / (ma + m)
        else:
            mu = (ma * masses[b]) / (ma + masses[b])
        want.append((tick, mu, jn))
    assert any(r[0][3] >= MONOPOLE_BASE for r in contacts)
    assert excitations == want


def test_recollapse_mass_timeline_keyed_by_event_order():
    # The falsifying case for the old two-pass bug: collapse mass 10 ->
    # contact -> recollapse mass 20 -> contact. The first excitation
    # must use 10 (mass at contact time), the second 20 — never the
    # final map for both.
    from simval.ontos_gravity import MONOPOLE_BASE

    def body(tick, bid, mass):
        return ("body", tick, bid, 0, 1, 1.0, 1.0, 0.0, 0.0, mass)

    def collapsed(tick, region, mass):
        return ("collapsed", tick, region % 2, region // 2, 3, mass, 1.0, 1.0, 0.0, 0.0, 0.0)

    def contact(tick, b):
        return ("contact", tick, 0, b, 0.5, 1.0, 1.0)

    mono = MONOPOLE_BASE + 2
    records = [
        body(1, 0, 2.0),
        collapsed(1, 2, 10.0),
        contact(2, mono),
        collapsed(3, 2, 20.0),
        contact(4, mono),
        ("tick", 4),
    ]
    excitations, final_tick = collect_excitations(records)
    assert final_tick == 4
    assert len(excitations) == 2
    mu_first = (2.0 * 10.0) / (2.0 + 10.0)
    mu_second = (2.0 * 20.0) / (2.0 + 20.0)
    assert excitations[0] == (2, mu_first, 0.5)
    assert excitations[1] == (4, mu_second, 0.5)
    assert struct.pack("<d", excitations[0][1]) != struct.pack("<d", excitations[1][1])


def test_monopole_contact_without_collapse_record_rejected():
    from simval.ontos_gravity import MONOPOLE_BASE

    records = [
        ("body", 1, 0, 0, 1, 1.0, 1.0, 0.0, 0.0, 2.0),
        ("contact", 2, 0, MONOPOLE_BASE + 1, 0.5, 1.0, 1.0),
        ("tick", 2),
    ]
    with pytest.raises(ValueError, match="no preceding RegionCollapsed"):
        collect_excitations(records)


# --- AUDIO-002: the strict parser enforces header + ContactParams invariants ---


def test_non_128_world_header_rejected_by_parser_and_audio(tmp_path):
    src = CONTACT_EXAMPLES / "contact" / "ontos.stream"
    data = bytearray(src.read_bytes())
    struct.pack_into("<I", data, 8, 129)
    bad = tmp_path / "wide.stream"
    bad.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="pins 128x128"):
        parse_stream_v2(bad)
    with pytest.raises(ValueError, match="pins 128x128"):
        synthesize_stream(bad)


def test_nan_restitution_rejected_by_parser_and_audio(tmp_path):
    # The restitution corpus stream carries ContactParams as its first
    # record (header 20 bytes: tag + restitution + friction + walls).
    src = CONTACT_EXAMPLES / "restitution" / "ontos.stream"
    data = bytearray(src.read_bytes())
    assert data[20] == 0x0C
    struct.pack_into("<d", data, 21, float("nan"))
    bad = tmp_path / "nan.stream"
    bad.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="restitution must be finite"):
        parse_stream_v2(bad)
    with pytest.raises(ValueError, match="restitution must be finite"):
        synthesize_stream(bad)


def test_out_of_range_friction_rejected_by_parser(tmp_path):
    src = CONTACT_EXAMPLES / "restitution" / "ontos.stream"
    data = bytearray(src.read_bytes())
    struct.pack_into("<d", data, 29, -0.5)  # friction slot
    bad = tmp_path / "negfriction.stream"
    bad.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="friction finite"):
        parse_stream_v2(bad)
