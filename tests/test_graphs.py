"""Graph census: exhaustive counts (OEIS A000088) and sampler/channel mapping."""

import pytest

from agora.graphs import channel_spec, enumerate_graphs, features_from_edges, sample_graphs, edges_of


@pytest.fixture(scope="module")
def classes():
    return enumerate_graphs(6)


def test_exhaustive_counts_match_oeis(classes):
    assert {n: len(v) for n, v in classes.items()} == {1: 1, 2: 2, 3: 4, 4: 11, 5: 34, 6: 156}


def test_channel_spec_is_pairwise(classes):
    key = classes[3][-1]  # some 3-vertex graph
    n, edges = edges_of(key)
    agents = ["a", "b", "c"]
    spec = channel_spec(key, agents)
    assert len(spec) == len(edges)
    for name, ch in spec.items():
        assert ch["readers"] == ch["writers"] and len(ch["readers"]) == 2


def test_channel_spec_size_mismatch_raises(classes):
    with pytest.raises(ValueError):
        channel_spec(classes[3][0], ["a", "b"])


def test_features_known_shapes():
    ring = features_from_edges(5, [(i, (i + 1) % 5) for i in range(5)])
    assert ring["connected"] and not ring["is_tree"] and ring["diameter"] == 2
    star = features_from_edges(5, [(0, i) for i in range(1, 5)])
    assert star["is_tree"] and star["max_degree"] == 4 and star["diameter"] == 2
    assert star["clustering"] == 0.0


def test_sampler_deterministic_and_covers_families():
    a = sample_graphs(12, 2, 7)
    b = sample_graphs(12, 2, 7)
    assert a == b
    assert {s["family"] for s in a} >= {"tree", "kary_tree_k3", "barbell"}
    for s in a:
        assert features_from_edges(12, s["edges"])["n"] == 12
