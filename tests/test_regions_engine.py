import numpy as np
import pytest

from globe_quiz.model import SURFACE_ICE, SURFACE_LAND, SURFACE_WATER, EarthModel
from globe_quiz.geo import latlon_to_vec
from globe_quiz.regions import RegionSet, parse_region, point_in_polygon


def _fake_model(points, heights=None):
    """Модель из маленьких граней в заданных точках: [(lat, lon, surface), ...]."""
    lat = np.array([p[0] for p in points], float)
    lon = np.array([p[1] for p in points], float)
    c = latlon_to_vec(lat, lon)
    tris = np.repeat(c[:, None, :], 3, axis=1)
    return EarthModel(
        tris=tris,
        normals=c,
        centers=c,
        colors=np.zeros((len(points), 3), np.uint8),
        surface=np.array([p[2] for p in points], np.int8),
        latlon=np.stack([lat, lon], 1),
        height=np.zeros(len(points)) if heights is None else np.asarray(heights, float),
    )


def test_point_in_polygon_square():
    poly = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], float)
    lat = np.array([5, 15, 5])
    lon = np.array([5, 5, -1])
    assert point_in_polygon(lat, lon, poly).tolist() == [True, False, False]


def test_polygon_across_dateline():
    region = parse_region({"id": "p", "name": "P", "kind": "island",
                           "polygon": [[-10, 170], [-10, 190], [10, 190], [10, 170]]})
    inside = region.parts[0].contains(np.array([[0, 175], [0, -175], [0, 160]], float))
    assert inside.tolist() == [True, True, False]


def test_lon_range_across_dateline():
    region = parse_region({"id": "p", "name": "P", "kind": "ocean", "lon_min": 150, "lon_max": -100})
    inside = region.parts[0].contains(np.array([[0, 160], [0, -120], [0, 0]], float))
    assert inside.tolist() == [True, True, False]


def test_base_regions_split_faces_and_overlays_stack():
    model = _fake_model([
        (10, 10, SURFACE_LAND),   # 0 материк А + полуостров
        (10, 10.5, SURFACE_WATER),  # 1 вода рядом — океан, не материк
        (80, 0, SURFACE_ICE),     # 2 лёд на севере — океан
        (-10, 10, SURFACE_LAND),  # 3 материк Б (южнее), А его не забирает
    ])
    regions = RegionSet([
        parse_region({"id": "a", "name": "А", "kind": "continent", "surface": "land", "lat_min": 0}),
        parse_region({"id": "b", "name": "Б", "kind": "continent", "surface": "land"}),
        parse_region({"id": "o", "name": "О", "kind": "ocean", "surface": ["water", "ice"]}),
        parse_region({"id": "pen", "name": "П", "kind": "peninsula", "surface": "land",
                      "polygon": [[5, 5], [5, 15], [15, 15], [15, 5]]}),
        parse_region({"id": "north", "name": "С", "kind": "hemisphere", "lat_min": 0}),
        parse_region({"id": "np", "name": "Северный полюс", "kind": "point", "at": [90, 0]}),
    ]).assign(model)
    assert [r.id for r in regions.regions_of_face(0)] == ["a", "pen", "north"]
    assert [r.id for r in regions.regions_of_face(1)] == ["o", "north"]
    assert [r.id for r in regions.regions_of_face(2)] == ["o", "north"]
    assert [r.id for r in regions.regions_of_face(3)] == ["b"]
    assert regions.most_specific(0).id == "pen"
    assert regions.most_specific(1).id == "o"
    assert not regions.unassigned().any()
    assert regions["np"].center == (90, 0)
    assert regions["a"].center[0] == pytest.approx(10, abs=0.5)


def test_nearest_target_point():
    model = _fake_model([(0, 0, SURFACE_LAND), (0, 40, SURFACE_LAND)])
    regions = RegionSet([
        parse_region({"id": "a", "name": "А", "kind": "continent", "surface": "land"}),
        parse_region({"id": "eq", "name": "Экватор", "kind": "line", "lat": 0}),
    ]).assign(model)
    assert regions.nearest_target_point(regions["a"], (0, 35)) == (0, 40)
    assert regions.nearest_target_point(regions["eq"], (30, 77)) == (0, 77)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        RegionSet([parse_region({"id": "x", "name": "X", "kind": "island"})] * 2)


def test_height_limits_pick_mountains_and_plains():
    model = _fake_model([(45, 10, SURFACE_LAND)] * 3 + [(45, 30, SURFACE_LAND)], heights=[2.2, 4.0, 6.0, 7.0])
    box = [[40, 0], [40, 20], [50, 20], [50, 0]]
    regions = RegionSet([
        parse_region({"id": "alps", "name": "Альпы", "kind": "mountains", "surface": "land",
                      "polygon": box, "min_height": 3.5}),
        parse_region({"id": "plain", "name": "равнина", "kind": "plain", "surface": "land",
                      "polygon": box, "max_height": 3.0}),
    ]).assign(model)
    assert regions["alps"].faces.tolist() == [False, True, True, False]  # только высокие и в контуре
    assert regions["plain"].faces.tolist() == [True, False, False, False]
    assert regions.most_specific(0).id == "plain"


def test_peak_is_a_target_only_point():
    model = _fake_model([(-3, 37, SURFACE_LAND)])
    regions = RegionSet([
        parse_region({"id": "kili", "name": "Килиманджаро", "kind": "peak", "at": [-5.5, 34.0]}),
    ]).assign(model)
    kili = regions["kili"]
    assert kili.at == (-5.5, 34.0) and kili.faces is None
    assert regions.nearest_target_point(kili, (10, 10)) == (-5.5, 34.0)
    assert regions.most_specific(0) is None  # вершина не называется при промахе


def _two_regions():
    return RegionSet([parse_region({"id": "x", "name": "Икс", "kind": "island"}),
                      parse_region({"id": "y", "name": "Игрек", "kind": "island"})])


def _facts(n):
    return [{"title": f"Факт {i}", "text": f"Текст {i}."} for i in range(n)]


def test_facts_attached_five_each():
    regions = _two_regions()
    regions.attach_facts({"x": _facts(5), "y": _facts(5)})
    assert len(regions["x"].facts) == 5 and regions["y"].facts[2].title == "Факт 2"


def test_facts_wrong_count_names_region():
    with pytest.raises(ValueError, match="Игрек.*4 фактов"):
        _two_regions().attach_facts({"x": _facts(5), "y": _facts(4)})


def test_facts_unknown_id_rejected():
    with pytest.raises(ValueError, match="неизвестных объектов: z"):
        _two_regions().attach_facts({"x": _facts(5), "y": _facts(5), "z": _facts(5)})
