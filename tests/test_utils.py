import pytest

from vczstore.utils import merge_with


def eq(a, b):
    return a == b


def keep(a, b):
    return a


def cat(a, b):
    return a + b


@pytest.mark.parametrize(
    ("l1", "l2", "expected"),
    [
        # basic: no matches
        (["a", "b"], ["c", "d"], ["a", "b", "c", "d"]),
        # single match in middle: combined, ordering preserved
        (["a", "b"], ["b", "c"], ["a", "b", "c"]),
        # match at start
        (["a", "b"], ["a", "c"], ["a", "b", "c"]),
        # match at end
        (["a", "b"], ["c", "b"], ["a", "c", "b"]),
        # all match
        (["a", "b"], ["a", "b"], ["a", "b"]),
        # empty lists
        ([], ["a"], ["a"]),
        (["a"], [], ["a"]),
        # combine is called: cat concatenates
        (["a"], ["a"], ["aa"]),
        # greedy: l1[0] matches l2[0], l1[1] gets no match (l2[0] already taken)
        (["a", "a"], ["a"], ["a", "a"]),
    ],
)
def test_merge_with(l1, l2, expected):
    fn = cat if expected == ["aa"] else keep
    assert merge_with(l1, l2, equiv=eq, combine=fn) == expected


def test_merge_with_conflict():
    with pytest.raises(ValueError, match="conflict"):
        merge_with(["a", "b"], ["b", "a"], equiv=eq, combine=keep)


def test_merge_with_combine_called():
    combined = []

    def record_combine(a, b):
        combined.append((a, b))
        return f"{a}+{b}"

    result = merge_with(["x", "y"], ["y", "z"], equiv=eq, combine=record_combine)
    assert result == ["x", "y+y", "z"]
    assert combined == [("y", "y")]
