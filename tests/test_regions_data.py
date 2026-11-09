import numpy as np
import pytest

from globe_quiz.model import SURFACE_WATER
from globe_quiz.regions import RegionSet

from .conftest import face_at


@pytest.fixture(scope="module")
def regions(earth):
    return RegionSet.load().assign(earth)


CONTROL_POINTS = [
    # базовые регионы
    ((30, 31), "africa"),  # Каир
    ((55.7, 37.6), "eurasia"),  # Москва
    ((40, -100), "north_america"),
    ((10, -84), "north_america"),  # Коста-Рика
    ((75, -40), "north_america"),  # Гренландия
    ((-10, -55), "south_america"),
    ((4.6, -74), "south_america"),  # Богота
    ((-25, 135), "australia"),
    ((-80, 0), "antarctica"),
    ((0, -150), "pacific_ocean"),
    ((30, -40), "atlantic_ocean"),
    ((-20, 80), "indian_ocean"),
    ((89.9, 0), "arctic_ocean"),  # Северный полюс — на льду океана
    ((74, -160), "arctic_ocean"),  # Чукотское море
    ((42, 50.5), "caspian_sea"),
    ((-41, 174), "new_zealand"),
    # острова и полуострова
    ((55, -3), "great_britain"),
    ((54, -9), "ireland"),
    ((-5, 142), "new_guinea"),
    ((1, 114), "kalimantan"),
    ((-42, 146.5), "tasmania"),
    ((7.5, 80.7), "sri_lanka"),
    ((63, 14), "scandinavian_peninsula"),
    ((42, -5), "iberian_peninsula"),
    ((41.5, 15), "apennine_peninsula"),
    ((23, 46), "arabian_peninsula"),
    ((18, 77), "hindustan"),
    ((56, 159), "kamchatka"),
    ((65, -152), "alaska"),
    ((28, -81.5), "florida"),
    # моря и заливы
    ((35, 18), "mediterranean_sea"),
    ((43.5, 35), "black_sea"),
    ((56, 17), "baltic_sea"),
    ((20, 38.5), "red_sea"),
    ((15, -75), "caribbean_sea"),
    ((27, 51), "persian_gulf"),
    ((25, -90), "gulf_of_mexico"),
    ((60, -86), "hudson_bay"),
    # горы, пустыни, лёд
    ((22, 10), "sahara"),
    ((-25, 125), "australian_deserts"),
    ((28.5, 85), "himalayas"),
    ((33, 90), "tibetan_plateau"),
    ((72, -40), "greenland"),
    ((72, -40), "north_america"),
    ((47, 10), "alps"),
    ((42.5, 44), "caucasus"),
    ((9, 39), "ethiopian_highlands"),
    ((55, 40), "east_european_plain"),
    ((-5, -62), "amazon_lowland"),
    ((87, 0), "arctic_ocean"),  # полярная ледяная шапка — лёд Северного Ледовитого океана
    ((50, -120), "cordillera"),
    ((-80, 0), "antarctic_ice_sheet"),
    # полушария
    ((45, 0), "northern_hemisphere"),
    ((-45, 0), "southern_hemisphere"),
]


@pytest.mark.parametrize("point,region_id", CONTROL_POINTS)
def test_control_point(earth, regions, point, region_id):
    face = face_at(earth, *point)
    names = [r.id for r in regions.regions_of_face(face)]
    assert region_id in names, f"{point}: {names}"


def test_andes_follow_the_mountain_strip(regions):
    lat, lon = regions["andes"].center
    assert -40 < lat < -5 and -76 < lon < -64


def test_suez_asian_side_is_eurasia(earth, regions):
    names = [r.id for r in regions.regions_of_face(face_at(earth, 29.5, 34.3))]
    assert "eurasia" in names and "africa" not in names


def test_no_unlabeled_faces(regions):
    assert not regions.unassigned().any()


def test_every_face_region_has_faces(regions):
    for r in regions.regions:
        if r.kind not in ("point", "peak", "line"):
            assert r.faces.any(), r.id


def test_seas_contain_only_water(earth, regions):
    for r in regions.regions:
        if r.kind in ("sea", "gulf", "lake", "ocean") and r.id not in ("arctic_ocean", "hudson_bay"):
            assert (earth.surface[r.faces] == SURFACE_WATER).all(), r.id


def test_most_specific_name(earth, regions):
    assert regions.most_specific(face_at(earth, 41.5, 15)).id == "apennine_peninsula"
    assert regions.most_specific(face_at(earth, 0, 25)).id == "africa"


def test_continents_do_not_overlap(regions):
    cont = np.stack([r.faces for r in regions.regions if r.kind == "continent"])
    assert (cont.sum(axis=0) <= 1).all()


# --- Рельеф: горы и равнины по высоте, вершины ------------------------------------

MOUNTAINS = ["alps", "caucasus", "ural_mountains", "scandinavian_mountains", "atlas", "himalayas",
             "tibetan_plateau", "ethiopian_highlands", "andes", "cordillera", "great_dividing_range"]
PLAINS = ["east_european_plain", "west_siberian_plain", "amazon_lowland"]


@pytest.mark.parametrize("rid", MOUNTAINS)
def test_mountains_are_raised(earth, regions, rid):
    faces = regions[rid].faces
    assert faces.sum() >= 5
    # Горы — только возвышенные грани, заметно выше суши вокруг.
    lat, lon = regions[rid].center
    from globe_quiz.geo import angular_distance
    d = np.array([angular_distance((lat, lon), tuple(x)) for x in earth.latlon])
    any_mountains = np.zeros(len(earth), bool)
    for m in MOUNTAINS:
        any_mountains |= regions[m].faces
    around = (d < 15) & (earth.surface == 1) & ~any_mountains
    assert earth.height[faces].min() >= regions[rid].parts[0].min_height
    assert np.median(earth.height[faces]) > np.median(earth.height[around])


@pytest.mark.parametrize("rid", PLAINS)
def test_plains_are_low(earth, regions, rid):
    faces = regions[rid].faces
    assert faces.sum() >= 10
    assert earth.height[faces].max() <= regions[rid].parts[0].max_height


PEAKS = {  # вершина: (настоящие координаты, материк)
    "everest": ((27.99, 86.93), "eurasia"),
    "kilimanjaro": ((-3.07, 37.35), "africa"),
    "elbrus": ((43.35, 42.44), "eurasia"),
    "mont_blanc": ((45.83, 6.86), "eurasia"),
    "aconcagua": ((-32.65, -70.01), "south_america"),
    "denali": ((63.07, -151.0), "north_america"),
}


@pytest.mark.parametrize("rid", PEAKS)
def test_peaks_on_raised_land_near_real_place(earth, regions, rid):
    from globe_quiz.geo import angular_distance

    real, continent = PEAKS[rid]
    peak = regions[rid]
    assert peak.kind == "peak"
    assert angular_distance(peak.at, real) <= 6.0
    f = face_at(earth, *peak.at)
    assert earth.surface[f] != SURFACE_WATER
    assert continent in [r.id for r in regions.regions_of_face(f)]
    # Отдельных конусов в модели нет: вершина стоит в горах или выше равнин вокруг.
    any_mountains = np.zeros(len(earth), bool)
    for m in MOUNTAINS:
        any_mountains |= regions[m].faces
    d = np.array([angular_distance(peak.at, tuple(x)) for x in earth.latlon])
    around = (d > 3) & (d < 7) & (earth.surface == 1) & ~any_mountains
    assert any_mountains[f] or earth.height[f] > np.median(earth.height[around])
