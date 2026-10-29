import polars as pl

from mb3d.formulas import catalog
from mb3d.params.m3f_meta import parse_m3f


def test_parse_m3f_options():
    text = """[OPTIONS]
.Version = 4
.DEscale = .2
.DEoption = 11
.RStop = 20
.SIpower = 2
.Double Scale = 1.5
.Boxscale Min R = 0.5
.Double Fold = 1
.3SingleAngles Rotation1 = 5
.Double Scale vary = 0
.Integer Sphere or Cylinder = 1
[CODE]
00
[END]
Description:
Formula proposed by Kali.
"""
    m = parse_m3f(text, "Amazing Surf")
    assert m.de_option == 11 and m.de_scale == 0.2 and m.rstop == 20 and m.version == 4
    assert [o.name for o in m.options] == ["Scale", "Min R", "Fold", "Rotation1 X", "Rotation1 Y", "Rotation1 Z",
                                           "Scale vary", "Sphere or Cylinder"]
    assert m.options[3].default == 5.0
    assert m.category == "3Da"
    assert "Kali" in m.description


def test_catalog_frame():
    df = catalog.catalog()
    assert isinstance(df, pl.DataFrame)
    assert df.filter(pl.col("name") == "Integer Power")["available"][0]
    assert df["name"].n_unique() == df.height
    assert "Amazing Surf" in catalog.search("surf")["name"].to_list()
    assert catalog.option_names("Menger3")[:4] == ["Scale", "CScale X", "CScale Y", "CScale Z"]
    assert catalog.defaults("Integer Power") == [8.0, -1.0]
