import pytest

from vczstore.utils import merge_lists


@pytest.mark.parametrize(
    ("l1", "l2", "expected"),
    [
        pytest.param(["b", "a"], ["a", "c"], ["b", "a", "c"], id="basic"),
        pytest.param(["x", "y"], ["z"], ["x", "y", "z"], id="unique_to_l1"),
        pytest.param(["z"], ["x", "y"], ["z", "x", "y"], id="unique_to_l2"),
        pytest.param([], [], [], id="both_empty"),
        pytest.param([], ["a", "b"], ["a", "b"], id="l1_empty"),
        pytest.param(["a", "b"], [], ["a", "b"], id="l2_empty"),
        pytest.param(["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"], id="identical"),
        pytest.param(["a"], ["b"], ["a", "b"], id="single_no_conflict"),
        pytest.param(["a"], ["a"], ["a"], id="single_same"),
        pytest.param(
            ["a", "c"], ["b", "c"], ["a", "b", "c"], id="ambiguous_deterministic"
        ),
        pytest.param(["a", "b", "c"], ["a", "c"], ["a", "b", "c"], id="longer_chain"),
    ],
)
def test_merge_lists(l1, l2, expected):
    assert merge_lists(l1, l2) == expected


@pytest.mark.parametrize(
    ("l1", "l2"),
    [
        pytest.param(["a", "b"], ["b", "a"], id="conflict"),
    ],
)
def test_merge_lists_raises_conflict(l1, l2):
    with pytest.raises(ValueError, match="conflict"):
        merge_lists(l1, l2)


def test_within_list_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        merge_lists(["a", "a"], ["a"])


# --- key= tests ---


def test_key_basic():
    # Objects compared by .id, not by equality
    class Item:
        def __init__(self, id, val):
            self.id = id
            self.val = val

        def __eq__(self, other):
            raise AssertionError("__eq__ must not be called")

        def __hash__(self):
            raise AssertionError("__hash__ must not be called")

    b = Item("b", 1)
    a1 = Item("a", 2)
    a2 = Item("a", 3)  # same key "a" as a1 — item from l1 (a1) should be kept
    c = Item("c", 4)

    result = merge_lists([b, a1], [a2, c], key=lambda x: x.id)
    assert [x.id for x in result] == ["b", "a", "c"]
    assert result[1] is a1  # l1's item is kept when keys collide


def test_key_conflict():
    class Item:
        def __init__(self, id):
            self.id = id

        def __eq__(self, other):
            raise AssertionError("__eq__ must not be called")

        def __hash__(self):
            raise AssertionError("__hash__ must not be called")

    a, b = Item("a"), Item("b")
    with pytest.raises(ValueError, match="conflict"):
        merge_lists([a, b], [b, a], key=lambda x: x.id)


def test_key_duplicate_in_input():
    class Item:
        def __init__(self, id):
            self.id = id

    a1, a2 = Item("a"), Item("a")
    with pytest.raises(ValueError, match="duplicate"):
        merge_lists([a1, a2], [], key=lambda x: x.id)
