import warnings

import pytest

from menpo.visualize import textutils
from menpo.visualize.textutils import (
    bytes_str,
    print_dynamic,
    print_progress,
    progress_bar_str,
)


class FakeClock:
    """A time() that moves on by `step` seconds each time it is called."""

    def __init__(self, step=1.0, start=1000.0):
        self.now, self.step = start, step

    def __call__(self):
        self.now += self.step
        return self.now


def reports(capsys):
    """The progress reports that were printed, in order (print_dynamic starts each with a carriage return)."""
    out = capsys.readouterr().out
    return [r.strip() for r in out.split("\r") if r.strip()], out


# ---- progress_bar_str
def test_progress_bar_str_draws_the_bar_and_the_percentage():
    assert progress_bar_str(0) == "[" + " " * 20 + "] 0%"
    assert progress_bar_str(0.5) == "[" + "=" * 10 + " " * 10 + "] 50%"
    assert progress_bar_str(1) == "[" + "=" * 20 + "] 100%"


def test_progress_bar_str_options():
    assert progress_bar_str(0.5, bar_length=10) == "[=====     ] 50%"
    assert progress_bar_str(0.5, bar_length=10, bar_marker="#") == "[#####     ] 50%"
    assert progress_bar_str(0.5, show_bar=False) == "50%"


def test_progress_bar_str_above_one_is_a_full_bar():
    assert progress_bar_str(3) == progress_bar_str(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(percentage=-0.1),
        dict(percentage=0.5, bar_length=0),
        dict(percentage=0.5, bar_length=2.5),
        dict(percentage=0.5, bar_marker="=="),
        dict(percentage=0.5, bar_marker=""),
    ],
)
def test_progress_bar_str_rejects_what_it_cannot_draw(kwargs):
    with pytest.raises(ValueError):
        progress_bar_str(**kwargs)


# ---- bytes_str
@pytest.mark.parametrize(
    "num, text",
    [
        (0, "0.00 bytes"),
        (1023, "1023.00 bytes"),
        (1024, "1.00 KB"),
        (12345, "12.06 KB"),
        (123456789, "117.74 MB"),
        (1024**3, "1.00 GB"),
        (1024**4, "1.00 TB"),
        (1024**5, "1024.00 TB"),
    ],
)
def test_bytes_str(num, text):
    assert bytes_str(num) == text


@pytest.mark.parametrize("bad", [-1, 1.5, "10", None])
def test_bytes_str_needs_a_non_negative_integer(bad):
    with pytest.raises(ValueError):
        bytes_str(bad)


# ---- print_dynamic
def test_print_dynamic_overwrites_the_line_and_pads_it_to_80_characters(capsys):
    print_dynamic("abc")
    out = capsys.readouterr().out
    assert out == "\rabc" + " " * 77


# ---- print_progress
def test_print_progress_passes_the_items_through_and_ends_with_done(capsys):
    assert list(print_progress([10, 20, 30], min_seconds_between_updates=0)) == [
        10,
        20,
        30,
    ]
    seen, out = reports(capsys)
    assert seen[-1] == "[" + "=" * 20 + "] 100% (3/3) - done."
    assert out.endswith("\n")  # a new line after it, by default


def test_print_progress_shows_the_count_and_the_time_that_remains(capsys, monkeypatch):
    monkeypatch.setattr(textutils, "time", FakeClock(step=1.0))
    list(print_progress(range(4), min_seconds_between_updates=0))
    seen, _ = reports(capsys)
    # each item takes 1 second, so after the first there are 3 left, then 2, then 1
    assert seen[0] == "[" + "=" * 5 + " " * 15 + "] 25% (1/4) - 00:00:03 remaining"
    assert [s.split(" - ")[1] for s in seen[:-1]] == [
        "00:00:03 remaining",
        "00:00:02 remaining",
        "00:00:01 remaining",
    ] + ["00:00:00 remaining"]
    assert len(seen) == 5 and seen[-1].endswith("(4/4) - done.")


def test_print_progress_does_not_wrap_a_time_of_more_than_a_day(capsys, monkeypatch):
    monkeypatch.setattr(textutils, "time", FakeClock(step=30000.0))  # 8h20m per item
    list(print_progress(range(4), min_seconds_between_updates=0))
    seen, _ = reports(capsys)
    # 3 items left: 90000 seconds, which is 25 hours (a datetime would say 01:00:00)
    assert seen[0].endswith("- 25:00:00 remaining")


def test_print_progress_with_too_few_items_says_no_time_is_left(capsys, monkeypatch):
    monkeypatch.setattr(textutils, "time", FakeClock(step=5.0))
    # n_items says 2, but 4 come: the number left is negative, not a time before 1970
    list(print_progress(iter(range(4)), n_items=2, min_seconds_between_updates=0))
    seen, _ = reports(capsys)
    assert seen[3].endswith("- 00:00:00 remaining")


def test_print_progress_does_not_use_a_deprecated_clock(capsys, monkeypatch):
    # utcfromtimestamp is deprecated (and scheduled for removal): a warning would be an error
    monkeypatch.setattr(textutils, "time", FakeClock(step=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        list(print_progress(range(5), min_seconds_between_updates=0))


def test_print_progress_prefix_shortens_the_bar(capsys):
    list(print_progress(range(2), prefix="Fitting", min_seconds_between_updates=0))
    seen, _ = reports(capsys)
    assert seen[-1] == "Fitting: [" + "=" * 10 + "] 100% (2/2) - done."


def test_print_progress_can_hide_the_bar_the_count_and_the_time(capsys, monkeypatch):
    monkeypatch.setattr(textutils, "time", FakeClock(step=1.0))
    list(
        print_progress(
            range(2),
            show_bar=False,
            show_count=False,
            show_eta=False,
            min_seconds_between_updates=0,
        )
    )
    seen, _ = reports(capsys)
    # one for each item, then the done message
    assert seen == ["50%", "100%", "100% - done."]


def test_print_progress_without_a_new_line_leaves_the_cursor_on_the_report(capsys):
    list(
        print_progress(range(2), end_with_newline=False, min_seconds_between_updates=0)
    )
    _, out = reports(capsys)
    assert not out.endswith("\n")


def test_print_progress_of_a_generator_needs_n_items(capsys):
    assert list(print_progress((x for x in range(3)), n_items=3)) == [0, 1, 2]
    with pytest.raises(TypeError):  # a generator has no length to fall back on
        list(print_progress(x for x in range(3)))


def test_print_progress_offset_counts_items_that_were_already_done(capsys, monkeypatch):
    monkeypatch.setattr(textutils, "time", FakeClock(step=1.0))
    list(print_progress(range(3), n_items=5, offset=2, min_seconds_between_updates=0))
    seen, _ = reports(capsys)
    assert "(3/5)" in seen[0] and "(5/5)" in seen[2]
    with pytest.raises(ValueError, match="offset"):
        list(print_progress(range(3), offset=2))


def test_print_progress_does_not_print_every_item_of_a_fast_loop(capsys, monkeypatch):
    # the throttle: a second report needs min_seconds_between_updates after the one before
    monkeypatch.setattr(textutils, "time", FakeClock(step=1.0))
    list(print_progress(range(50), min_seconds_between_updates=10))
    seen, _ = reports(capsys)
    counts = [s.split("(")[1].split(")")[0] for s in seen]
    assert counts == ["1/50", "11/50", "21/50", "31/50", "41/50", "50/50"]
    assert seen[-1].endswith("- done.")
