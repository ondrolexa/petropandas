import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import ResizeEvent
from matplotlib.colors import to_rgba

from petropandas._plotting import ProfilePlot, ScatterPlot, TernaryPlot
from petropandas._plotting import _axis_ticks, _polygon_vertices, _project, _variance

# --- ScatterPlot ---


@pytest.fixture
def scatter_rendered(garnet_groups):
    g1, g2, g3 = garnet_groups
    s = ScatterPlot("Prp", "Sps+Grs")
    s.add(g1, label="Garnet 1")
    s.add(g2, label="Garnet 2")
    s.add(g3, label="Garnet 3")
    fig, ax = s.render()
    yield fig, ax
    plt.close(fig)


def test_scatter_render_returns_figure_and_axes(scatter_rendered):
    fig, ax = scatter_rendered
    assert fig is not None
    assert ax is not None


def test_scatter_uses_constrained_layout(scatter_rendered):
    fig, _ = scatter_rendered
    assert fig.get_constrained_layout() is True


def test_scatter_one_collection_per_group(scatter_rendered):
    _, ax = scatter_rendered
    assert len(ax.collections) == 3


def test_scatter_eval_expressions_produce_correct_offsets(
    scatter_rendered, garnet_groups
):
    g1, _, _ = garnet_groups
    _, ax = scatter_rendered
    offsets = np.asarray(ax.collections[0].get_offsets())
    expected_x = g1["Prp"].to_numpy()
    expected_y = (g1["Sps"] + g1["Grs"]).to_numpy()
    assert offsets[:, 0] == pytest.approx(expected_x)
    assert offsets[:, 1] == pytest.approx(expected_y)


def test_scatter_ion_notation_column_names():
    """Ion/APFU column names (e.g. "Al{3+}") contain characters that are not
    valid in a bare pandas.eval() expression -- plotting them directly must
    not raise SyntaxError."""
    df = pd.DataFrame({"Al{3+}": [1.0, 2.0], "Si{4+}": [3.0, 4.0]})
    s = ScatterPlot("Al{3+}", "Si{4+}")
    s.add(df)
    fig, ax = s.render()
    offsets = np.asarray(ax.collections[0].get_offsets())
    assert offsets[:, 0] == pytest.approx(df["Al{3+}"].to_numpy())
    assert offsets[:, 1] == pytest.approx(df["Si{4+}"].to_numpy())
    plt.close(fig)


def test_scatter_backtick_quoted_ion_expression():
    """Special-character column names can still be combined in an expression
    via backtick-quoting (standard pandas.DataFrame.eval() syntax)."""
    df = pd.DataFrame({"Al{3+}": [1.0, 2.0], "Si{4+}": [3.0, 4.0]})
    s = ScatterPlot("`Al{3+}` + `Si{4+}`", "Al{3+}")
    s.add(df)
    fig, ax = s.render()
    offsets = np.asarray(ax.collections[0].get_offsets())
    assert offsets[:, 0] == pytest.approx((df["Al{3+}"] + df["Si{4+}"]).to_numpy())
    plt.close(fig)


def test_scatter_default_label_strips_backticks():
    """Backtick quoting is eval() syntax, not part of the intended label text."""
    df = pd.DataFrame({"Al{3+}": [1.0, 2.0], "Si{4+}": [3.0, 4.0]})
    s = ScatterPlot("`Al{3+}` + `Si{4+}`", "Al{3+}")
    s.add(df)
    fig, ax = s.render()
    assert ax.get_xlabel() == "Al{3+} + Si{4+}"
    assert ax.get_ylabel() == "Al{3+}"
    plt.close(fig)


def test_scatter_explicit_label_with_backtick_unchanged():
    df = pd.DataFrame({"Al{3+}": [1.0, 2.0]})
    s = ScatterPlot("Al{3+}", "Al{3+}", xlabel="literal `backtick` label")
    s.add(df)
    fig, ax = s.render()
    assert ax.get_xlabel() == "literal `backtick` label"
    plt.close(fig)


def test_scatter_missing_name_in_expression_defaults_to_zero():
    """Groups don't always share every column (e.g. not every mineral group
    reports Sps) -- a name missing from *within* a multi-term expression
    should default to 0 rather than raising UndefinedVariableError."""
    df = pd.DataFrame({"Grs": [1.0, 2.0]})
    s = ScatterPlot("Sps+Grs", "Grs")
    s.add(df)
    fig, ax = s.render()
    offsets = np.asarray(ax.collections[0].get_offsets())
    assert offsets[:, 0] == pytest.approx(df["Grs"].to_numpy())
    plt.close(fig)


def test_scatter_missing_backtick_name_in_expression_defaults_to_zero():
    df = pd.DataFrame({"Al{3+}": [1.0, 2.0]})
    s = ScatterPlot("`Al{3+}` + `Si{4+}`", "Al{3+}")
    s.add(df)
    fig, ax = s.render()
    offsets = np.asarray(ax.collections[0].get_offsets())
    assert offsets[:, 0] == pytest.approx(df["Al{3+}"].to_numpy())
    plt.close(fig)


def test_scatter_missing_single_column_still_raises():
    """A lone column reference that's entirely missing is a real usage error
    (e.g. a typo), not a "some groups lack this value" situation -- it must
    keep raising rather than silently plotting zeros."""
    df = pd.DataFrame({"Grs": [1.0, 2.0]})
    s = ScatterPlot("Sps", "Grs")
    s.add(df)
    with pytest.raises(pd.errors.UndefinedVariableError):
        s.render()


def test_scatter_legend_labels(scatter_rendered):
    _, ax = scatter_rendered
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Garnet 1", "Garnet 2", "Garnet 3"]


def test_scatter_default_axis_labels_from_expressions(scatter_rendered):
    _, ax = scatter_rendered
    assert ax.get_xlabel() == "Prp"
    assert ax.get_ylabel() == "Sps+Grs"


def test_scatter_add_style_kwargs_override_color(garnet_groups):
    g1, g2, _ = garnet_groups
    s = ScatterPlot("Prp", "Sps+Grs")
    s.add(g1, label="Default cycle")
    s.add(g2, label="Forced red", color="red")
    fig, ax = s.render()
    try:
        forced_facecolor = ax.collections[1].get_facecolor()[0]
        assert tuple(forced_facecolor) == to_rgba("red")
    finally:
        plt.close(fig)


def test_scatter_scalar_expression_raises_type_error(garnet_groups):
    g1, _, _ = garnet_groups
    s = ScatterPlot("Prp", "1")
    s.add(g1, label="Garnet 1")
    with pytest.raises(TypeError):
        s.render()


def test_scatter_savefig_writes_file(garnet_groups, tmp_path):
    g1, _, _ = garnet_groups
    s = ScatterPlot("Prp", "Sps+Grs")
    s.add(g1, label="Garnet 1")
    out = tmp_path / "scatter.png"
    s.savefig(out, dpi=150)
    assert out.exists()
    assert out.stat().st_size > 0


def test_scatter_savefig_closes_its_figure(garnet_groups, tmp_path):
    g1, _, _ = garnet_groups
    s = ScatterPlot("Prp", "Sps+Grs")
    s.add(g1, label="Garnet 1")
    open_before = len(plt.get_fignums())
    s.savefig(tmp_path / "scatter.png")
    assert len(plt.get_fignums()) == open_before


def test_scatter_show_syncs_figsize_on_manual_resize(garnet_groups):
    g1, _, _ = garnet_groups
    s = ScatterPlot("Prp", "Sps+Grs", figsize=(4, 3))
    s.add(g1, label="Garnet 1")
    fig, _ = s.render()
    fig.canvas.mpl_connect("resize_event", lambda _event: s._sync_figsize(fig))
    try:
        fig.set_size_inches(8, 6, forward=False)
        fig.canvas.callbacks.process(
            "resize_event", ResizeEvent("resize_event", fig.canvas)
        )
        assert s.figsize == (8.0, 6.0)
    finally:
        plt.close(fig)


def test_scatter_explicit_axis_labels_and_limits(garnet_groups):
    g1, _, _ = garnet_groups
    s = ScatterPlot(
        "Prp",
        "Sps+Grs",
        xlabel="Pyrope",
        ylabel="Sps+Grs total",
        xlim=(0, 80),
        ylim=(0, 30),
    )
    s.add(g1, label="Garnet 1")
    fig, ax = s.render()
    try:
        assert ax.get_xlabel() == "Pyrope"
        assert ax.get_ylabel() == "Sps+Grs total"
        assert ax.get_xlim() == (0, 80)
        assert ax.get_ylim() == (0, 30)
    finally:
        plt.close(fig)


# --- TernaryPlot ---


@pytest.fixture
def ternary_rendered(garnet_groups):
    g1, g2, g3 = garnet_groups
    s = TernaryPlot("Prp", "Sps", "Grs")
    s.add(g1, label="Garnet 1")
    s.add(g2, label="Garnet 2")
    s.add(g3, label="Garnet 3")
    fig, ax = s.render()
    yield fig, ax
    plt.close(fig)


def test_ternary_render_returns_figure_and_axes(ternary_rendered):
    fig, ax = ternary_rendered
    assert fig is not None
    assert ax is not None


def test_ternary_uses_constrained_layout(ternary_rendered):
    fig, _ = ternary_rendered
    assert fig.get_constrained_layout() is True


def test_ternary_one_collection_per_group(ternary_rendered):
    _, ax = ternary_rendered
    assert len(ax.collections) == 3


def test_ternary_legend_labels(ternary_rendered):
    _, ax = ternary_rendered
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Garnet 1", "Garnet 2", "Garnet 3"]


def test_ternary_default_corner_labels_from_expressions(ternary_rendered):
    _, ax = ternary_rendered
    texts = [t.get_text() for t in ax.texts]
    assert "Prp" in texts
    assert "Sps" in texts
    assert "Grs" in texts


def test_ternary_default_labels_strip_backticks():
    df = pd.DataFrame(
        {"Al{3+}": [1.0, 2.0], "Si{4+}": [3.0, 4.0], "Ca{2+}": [0.5, 0.5]}
    )
    t = TernaryPlot("`Al{3+}` + `Si{4+}`", "Al{3+}", "Ca{2+}")
    t.add(df)
    fig, ax = t.render()
    texts = [txt.get_text() for txt in ax.texts]
    assert "Al{3+} + Si{4+}" in texts
    assert "Al{3+}" in texts
    assert "Ca{2+}" in texts
    plt.close(fig)


def test_ternary_default_ternary_sum():
    s = TernaryPlot("Prp", "Sps", "Grs")
    assert s.ternary_sum == 100.0


def test_ternary_sum_override():
    s = TernaryPlot("Prp", "Sps", "Grs", ternary_sum=1.0)
    assert s.ternary_sum == 1.0


def test_ternary_add_style_kwargs_override_color(garnet_groups):
    g1, g2, _ = garnet_groups
    s = TernaryPlot("Prp", "Sps", "Grs")
    s.add(g1, label="Default cycle")
    s.add(g2, label="Forced red", color="red")
    fig, ax = s.render()
    try:
        forced_facecolor = ax.collections[1].get_facecolor()[0]
        assert tuple(forced_facecolor) == to_rgba("red")
    finally:
        plt.close(fig)


def test_ternary_scalar_expression_raises_type_error(garnet_groups):
    g1, _, _ = garnet_groups
    s = TernaryPlot("1", "Sps", "Grs")
    s.add(g1, label="Garnet 1")
    with pytest.raises(TypeError):
        s.render()


def test_ternary_savefig_writes_file(garnet_groups, tmp_path):
    g1, _, _ = garnet_groups
    s = TernaryPlot("Prp", "Sps", "Grs")
    s.add(g1, label="Garnet 1")
    out = tmp_path / "ternary.png"
    s.savefig(out, dpi=150)
    assert out.exists()
    assert out.stat().st_size > 0


def test_ternary_no_cartesian_axis_attributes():
    s = TernaryPlot("Prp", "Sps", "Grs")
    assert not hasattr(s, "xlabel")
    assert not hasattr(s, "ylabel")
    assert not hasattr(s, "xlim")
    assert not hasattr(s, "ylim")


def test_ternary_explicit_limits_not_clipped(garnet_groups):
    """Regression test for an extreme asymmetric zoom (tlim=(0, 20) with
    llim/rlim left at the full range) that clipped the triangle under an
    earlier mpltern-based implementation's default fit="rectangle"
    heuristic. TernaryPlot's exact polygon clipping must render the full
    trapezoid, not an empty/degenerate axes.
    """
    g1, _, _ = garnet_groups
    s = TernaryPlot("Sps", "Prp", "Grs", tlim=(0, 20))
    s.add(g1, label="Garnet 1")
    fig, ax = s.render()
    try:
        assert s.tlim == (0, 20)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[1] - xlim[0] > 0.5
        assert ylim[1] - ylim[0] > 0
    finally:
        plt.close(fig)


# --- ProfilePlot ---


@pytest.fixture
def profile_rendered(profile_groups):
    p1, p2 = profile_groups
    s = ProfilePlot()
    s.add(p1, label="Profile 1")
    s.add(p2, label="Profile 2")
    fig, ax = s.render()
    yield fig, ax
    plt.close(fig)


def test_profile_render_returns_figure_and_axes(profile_rendered):
    fig, ax = profile_rendered
    assert fig is not None
    assert ax is not None


def test_profile_uses_constrained_layout(profile_rendered):
    fig, _ = profile_rendered
    assert fig.get_constrained_layout() is True


def test_profile_one_line_per_column_across_groups(profile_rendered):
    fig, ax = profile_rendered
    assert len(ax.lines) == 6


def test_profile_legend_labels(profile_rendered):
    _, ax = profile_rendered
    labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert labels == {
        "Profile 1 (CaO)",
        "Profile 1 (FeO)",
        "Profile 1 (MgO)",
        "Profile 1 (MnO)",
        "Profile 2 (ZnO)",
        "Profile 2 (Na2O)",
    }


def test_profile_xlabel_defaults_to_index_name(profile_rendered):
    _, ax = profile_rendered
    assert ax.get_xlabel() == "Point"


def test_profile_explicit_axis_labels(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot(xlabel="Distance", ylabel="wt%")
    s.add(p1, label="Profile 1")
    fig, ax = s.render()
    try:
        assert ax.get_xlabel() == "Distance"
        assert ax.get_ylabel() == "wt%"
    finally:
        plt.close(fig)


def test_profile_split_off_uses_single_axes(profile_rendered):
    fig, _ = profile_rendered
    assert len(fig.axes) == 1


def test_profile_explicit_secondary_columns_creates_second_axes(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot(secondary_columns=["MnO"])
    s.add(p1, label="Profile 1")
    fig, ax = s.render()
    try:
        assert len(fig.axes) == 2
        ax2 = fig.axes[1]
        primary_labels = {line.get_label() for line in ax.lines}
        secondary_labels = {line.get_label() for line in ax2.lines}
        assert secondary_labels == {"Profile 1 (MnO)"}
        assert primary_labels == {
            "Profile 1 (CaO)",
            "Profile 1 (FeO)",
            "Profile 1 (MgO)",
        }
    finally:
        plt.close(fig)


def test_profile_secondary_ylabel_only_set_with_secondary_axes(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot(secondary_columns=["MnO"], secondary_ylabel="Mn (wt%)")
    s.add(p1, label="Profile 1")
    fig, ax = s.render()
    try:
        assert fig.axes[1].get_ylabel() == "Mn (wt%)"
    finally:
        plt.close(fig)


def test_profile_ylabel_defaults_to_joined_column_names_when_split(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot(secondary_columns=["MnO"])
    s.add(p1, label="Profile 1")
    fig, ax = s.render()
    try:
        assert ax.get_ylabel() == "CaO FeO MgO"
        assert fig.axes[1].get_ylabel() == "MnO"
    finally:
        plt.close(fig)


def test_profile_explicit_ylabel_overrides_joined_default(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot(
        secondary_columns=["MnO"], ylabel="wt%", secondary_ylabel="Mn (wt%)"
    )
    s.add(p1, label="Profile 1")
    fig, ax = s.render()
    try:
        assert ax.get_ylabel() == "wt%"
        assert fig.axes[1].get_ylabel() == "Mn (wt%)"
    finally:
        plt.close(fig)


def test_profile_ylabel_blank_when_not_split(profile_rendered):
    _, ax = profile_rendered
    assert ax.get_ylabel() == ""


def test_profile_ylabel_joined_default_reflects_only_plotted_columns():
    # Within-group means are identical, so {small_a, small_b} vs
    # {large_a, large_b} is the unique zero-variance partition, and
    # "excluded" (not in `columns`) must not appear in either label.
    data = pd.DataFrame(
        {
            "small_a": [1.0, 1.0, 1.0, 1.0],
            "small_b": [1.0, 1.0, 1.0, 1.0],
            "large_a": [100.0, 100.0, 100.0, 100.0],
            "large_b": [100.0, 100.0, 100.0, 100.0],
            "excluded": [50.0, 50.0, 50.0, 50.0],
        }
    )
    s = ProfilePlot(columns=["small_a", "small_b", "large_a", "large_b"], split="auto")
    s.add(data, label="Data")
    fig, ax = s.render()
    try:
        assert ax.get_ylabel() == "small_a small_b"
        assert fig.axes[1].get_ylabel() == "large_a large_b"
    finally:
        plt.close(fig)


def test_profile_auto_split_groups_by_value_scale():
    # Within-group means are identical, so {small_a, small_b} vs
    # {large_a, large_b} is the unique zero-variance (and thus optimal)
    # partition - any other split mixes a small and a large mean together.
    data = pd.DataFrame(
        {
            "small_a": [1.0, 1.0, 1.0, 1.0],
            "small_b": [1.0, 1.0, 1.0, 1.0],
            "large_a": [100.0, 100.0, 100.0, 100.0],
            "large_b": [100.0, 100.0, 100.0, 100.0],
        }
    )
    s = ProfilePlot(split="auto")
    s.add(data, label="Data")
    fig, ax = s.render()
    try:
        secondary = s._secondary_axis_columns
        assert secondary in ({"small_a", "small_b"}, {"large_a", "large_b"})
    finally:
        plt.close(fig)


def test_profile_columns_str_normalized_to_list():
    s = ProfilePlot(columns="CaO")
    assert s.columns == ["CaO"]


def test_profile_secondary_columns_str_normalized_to_list():
    s = ProfilePlot(secondary_columns="FeO")
    assert s.secondary_columns == ["FeO"]


def test_profile_columns_without_secondary_split_off_drops_rest(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot(columns=["CaO", "MgO"], split="off")
    s.add(p1, label="Profile 1")
    fig, ax = s.render()
    try:
        assert len(fig.axes) == 1
        primary_labels = {line.get_label() for line in ax.lines}
        assert primary_labels == {"Profile 1 (CaO)", "Profile 1 (MgO)"}
    finally:
        plt.close(fig)


def test_profile_columns_auto_split_restricted_to_selected_columns():
    # Within-group means are identical, so {small_a, small_b} vs
    # {large_a, large_b} is the unique zero-variance partition among the
    # selected columns - "excluded" (a third distinct scale, not in
    # `columns`) must not be plotted at all.
    data = pd.DataFrame(
        {
            "small_a": [1.0, 1.0, 1.0, 1.0],
            "small_b": [1.0, 1.0, 1.0, 1.0],
            "large_a": [100.0, 100.0, 100.0, 100.0],
            "large_b": [100.0, 100.0, 100.0, 100.0],
            "excluded": [50.0, 50.0, 50.0, 50.0],
        }
    )
    s = ProfilePlot(columns=["small_a", "small_b", "large_a", "large_b"], split="auto")
    s.add(data, label="Data")
    fig, ax = s.render()
    try:
        ax2 = fig.axes[1]
        all_labels = {line.get_label() for line in ax.lines} | {
            line.get_label() for line in ax2.lines
        }
        assert "Data (excluded)" not in all_labels
        secondary = s._secondary_axis_columns
        assert secondary in ({"small_a", "small_b"}, {"large_a", "large_b"})
    finally:
        plt.close(fig)


def test_profile_columns_without_secondary_invalid_split_raises(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot(columns=["CaO", "MgO"], split="bogus")
    s.add(p1, label="Profile 1")
    with pytest.raises(ValueError, match="split must be"):
        s.render()


def test_profile_columns_and_secondary_columns_filters_others_out(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot(columns=["CaO"], secondary_columns=["FeO"])
    s.add(p1, label="Profile 1")
    fig, ax = s.render()
    try:
        ax2 = fig.axes[1]
        assert len(ax.lines) + len(ax2.lines) == 2
        assert {line.get_label() for line in ax.lines} == {"Profile 1 (CaO)"}
        assert {line.get_label() for line in ax2.lines} == {"Profile 1 (FeO)"}
    finally:
        plt.close(fig)


def test_profile_split_compresses_gap_between_groups():
    # Directly test _compress_split_gap with controlled pre-existing ylim
    # (rather than going through render()'s own autoscale), so the expected
    # gap can be computed exactly without matplotlib's margin/expander
    # behavior for the underlying data getting in the way.
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot([0, 1], [0.0, 10.0])
    ax2.plot([0, 1], [50.0, 100.0])
    ax.set_ylim(-1.0, 11.0)
    ax2.set_ylim(45.0, 105.0)
    lo_span_orig = 11.0 - (-1.0)
    hi_span_orig = 105.0 - 45.0

    s = ProfilePlot()
    s._secondary_ax = ax2
    try:
        s._compress_split_gap(ax)
        lo_min, lo_max = ax.get_ylim()
        hi_min, hi_max = ax2.get_ylim()
        available = 1.0 - ProfilePlot._GAP_FRACTION
        lo_frac = available * lo_span_orig / (lo_span_orig + hi_span_orig)
        hi_frac = available - lo_frac
        assert lo_min == pytest.approx(-1.0)
        assert lo_max == pytest.approx(-1.0 + lo_span_orig / lo_frac)
        assert hi_max == pytest.approx(105.0)
        assert hi_min == pytest.approx(105.0 - hi_span_orig / hi_frac)

        lo_top_frac = (11.0 - lo_min) / (lo_max - lo_min)
        hi_bottom_frac = (45.0 - hi_min) / (hi_max - hi_min)
        assert hi_bottom_frac - lo_top_frac == pytest.approx(ProfilePlot._GAP_FRACTION)
    finally:
        plt.close(fig)


def test_profile_split_gap_not_compressed_when_ranges_overlap():
    data = pd.DataFrame(
        {
            "primary_col": [0.0, 2.0, 5.0, 10.0],
            "secondary_col": [5.0, 8.0, 12.0, 15.0],
        }
    )
    s = ProfilePlot(secondary_columns=["secondary_col"])
    s.add(data, label="Data")
    fig, ax = s.render()
    try:
        primary_min, primary_max = sorted(ax.dataLim.intervaly)
        ylim = ax.get_ylim()
        top_frac = (primary_max - ylim[0]) / (ylim[1] - ylim[0])
        assert top_frac > 0.9
    finally:
        plt.close(fig)


def test_profile_columns_missing_from_data_silently_skipped(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot(columns=["CaO", "DoesNotExist"])
    s.add(p1, label="Profile 1")
    fig, ax = s.render()
    try:
        assert {line.get_label() for line in ax.lines} == {"Profile 1 (CaO)"}
    finally:
        plt.close(fig)


def test_profile_columns_filters_across_add_calls(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot(columns=["CaO"])
    s.add(p1, label="Profile 1")
    s.add(p1, label="Profile 2")
    fig, ax = s.render()
    try:
        assert len(fig.axes) == 1
        primary_labels = {line.get_label() for line in ax.lines}
        assert primary_labels == {"Profile 1 (CaO)", "Profile 2 (CaO)"}
    finally:
        plt.close(fig)


def test_profile_add_color_kwarg_ignored_colors_still_differ(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot()
    s.add(p1, label="Profile 1", color="red")
    fig, ax = s.render()
    try:
        colors = {line.get_color() for line in ax.lines}
        assert len(colors) == len(ax.lines)
    finally:
        plt.close(fig)


def test_profile_add_style_kwargs_applied_to_all_lines(profile_groups):
    p1, _ = profile_groups
    s = ProfilePlot()
    s.add(p1, label="Profile 1", linestyle="--")
    fig, ax = s.render()
    try:
        assert all(line.get_linestyle() == "--" for line in ax.lines)
    finally:
        plt.close(fig)


def test_profile_savefig_writes_file(profile_groups, tmp_path):
    p1, _ = profile_groups
    s = ProfilePlot()
    s.add(p1, label="Profile 1")
    out = tmp_path / "profile.png"
    s.savefig(out, dpi=150)
    assert out.exists()
    assert out.stat().st_size > 0


# --- ProfilePlot split unit tests ---


def test_variance_of_single_value_is_zero():
    assert _variance([5.0]) == 0.0


def test_variance_matches_hand_computed_value():
    assert _variance([1.0, 2.0, 3.0]) == pytest.approx(2.0 / 3.0)


def test_auto_split_picks_lower_variance_grouping():
    s = ProfilePlot(split="auto")
    s.add(
        pd.DataFrame(
            {
                "a": [1.0, 1.0],
                "b": [1.1, 1.1],
                "x": [100.0, 100.0],
                "y": [101.0, 101.0],
            }
        ),
        label="Data",
    )
    secondary = s._auto_split()
    assert secondary in ({"a", "b"}, {"x", "y"})


# --- ternary geometry unit tests ---


def test_project_vertices_match_expected_coordinates():
    assert _project(1, 0, 0) == pytest.approx((0.0, 1.0))
    assert _project(0, 1, 0) == pytest.approx((-0.5773502691896258, 0.0))
    assert _project(0, 0, 1) == pytest.approx((0.5773502691896258, 0.0))


def test_polygon_vertices_full_triangle_when_unlimited():
    poly = _polygon_vertices(None, None, None, 100.0)
    assert set(poly) == {(100.0, 0.0, 0.0), (0.0, 0.0, 100.0), (0.0, 100.0, 0.0)}


def test_polygon_vertices_extreme_tlim_not_degenerate():
    poly = _polygon_vertices((0, 20), None, None, 100.0)
    assert len(poly) == 4
    for t, left, r in poly:
        assert t == pytest.approx(0.0) or t == pytest.approx(20.0)
        assert t + left + r == pytest.approx(100.0)


def test_axis_ticks_match_verified_positions():
    poly = _polygon_vertices(None, None, None, 100.0)
    ticks = dict(
        (value, (round(x, 3), round(y, 3)))
        for (x, y), value, _normal in _axis_ticks(poly, 0, 0, 100)
    )
    assert ticks[0.0] == (0.577, 0.0)
    assert ticks[20.0] == (0.462, 0.2)
    assert ticks[100.0] == (0.0, 1.0)
