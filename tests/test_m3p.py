import math

import pytest

from mb3d.params import Scene, load, parse, save, save_text, to_bytes, to_text
from mb3d.params.m3p_text import raw_from_text, raw_to_text
from mb3d.params.bytes_io import short_float, to_short_float


def test_short_float_roundtrip():
    for v in (1.0, 0.5, 2.5, 12.0, 0.03, 99.0):
        assert short_float(to_short_float(v)) == pytest.approx(v, rel=0.06)


def test_all_samples_parse(param_dir):
    files = sorted(param_dir.glob("*.m3p"))
    assert len(files) >= 70
    for f in files:
        s = load(f)
        assert 1 <= s.width < 32768 and 1 <= s.height < 32768
        assert s.zoom > 0
        assert len(s.used_formulas()) >= 1
        assert len(s.lighting.lights) == 6 and len(s.lighting.palette) == 10


def test_known_values(param_dir):
    s = load(param_dir / "ABoxScale2Start.m3p")
    assert (s.width, s.height) == (400, 400)
    f = s.used_formulas()
    assert f[0].name == "Amazing Box" and f[0].fnr == 4
    assert f[0].options[0] == pytest.approx(2.0)  # scale 2
    q = load(param_dir / "QuatP4hybridJulia.m3p")
    assert q.julia
    names = [x.name for x in q.used_formulas()]
    assert names == ["Quaternion", "Integer Power"]
    assert q.used_formulas()[1].iterations == 2


def test_roundtrip_binary(param_dir):
    for name in ("ABoxScale2Start.m3p", "QuatP4hybridJulia.m3p", "Theli-At - Black Mandelbulb.m3p", "spineJulia.m3p"):
        s = load(param_dir / name)
        b = to_bytes(s)
        s2 = parse(b)
        assert s2.width == s.width and s2.iterations == s.iterations
        assert s2.zoom == pytest.approx(s.zoom)
        assert s2.mid == pytest.approx(s.mid)
        assert [f.name for f in s2.used_formulas()] == [f.name for f in s.used_formulas()]
        for a, c in zip(s2.used_formulas(), s.used_formulas()):
            assert a.options == pytest.approx(c.options)
            assert a.iterations == c.iterations
        assert s2.lighting.palette[0].diffuse == s.lighting.palette[0].diffuse
        assert s2.lighting.lights[0].x_angle == pytest.approx(s.lighting.lights[0].x_angle)
        assert s2.lighting.color_start == s.lighting.color_start
        assert s2.de_stop == pytest.approx(s.de_stop)
        # the normalised camera matrix
        for i in range(3):
            n1 = math.sqrt(sum(v * v for v in s.vgrads[i]))
            n2 = math.sqrt(sum(v * v for v in s2.vgrads[i]))
            for j in range(3):
                assert s2.vgrads[i][j] / n2 == pytest.approx(s.vgrads[i][j] / n1, abs=1e-12)


def test_roundtrip_text(param_dir):
    s = load(param_dir / "ABoxScale3Start.m3p")
    text = to_text(s)
    assert text.startswith("Mandelbulb3Dv18{")
    raw, title = raw_from_text(text)
    s2 = parse(raw)
    assert s2.width == s.width and s2.used_formulas()[0].options[0] == pytest.approx(3.0)


def test_default_scene_roundtrip(tmp_path):
    s = Scene()
    s.title = "fresh"
    save(s, tmp_path / "a.m3p")
    save_text(s, tmp_path / "a.txt")
    s2 = load(tmp_path / "a.m3p")
    s3 = load(tmp_path / "a.txt")
    for t in (s2, s3):
        assert t.width == 800 and t.iterations == 60
        assert t.used_formulas()[0].name == "Integer Power"
        assert t.lighting.lights[0].on and not t.lighting.lights[1].on
        assert t.lighting.palette[0].diffuse == (0xE1, 0xA0, 0x59)
        assert t.lighting.lights[0].color == (0xFF, 0xE8, 0xA0)
