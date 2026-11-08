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
    ((33, 90), "himalayas"),
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
        if r.kind not in ("point", "line"):
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
