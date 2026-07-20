from __future__ import annotations

import itertools
import math
import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# Default legend placement: outside the axes to the right, vertically
# centered. `constrained_layout` shrinks the axes to make room for a legend
# placed this way as long as it's attached via `Axes.legend()` (not
# `Figure.legend()`).
_LEGEND_OUTSIDE_KWARGS = {"loc": "center left", "bbox_to_anchor": (1.02, 0.5)}

# Matches an expr that is *only* a single column reference (bare identifier or one
# backtick-quoted name) with nothing else - used to tell "plot this one column" (a
# missing name should raise) apart from a genuine multi-term expression (a missing
# name should default to 0, see `_eval`).
_SINGLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$|^`[^`]+`$")
_NAME_TOKEN_RE = re.compile(r"`([^`]+)`|\b([A-Za-z_][A-Za-z0-9_]*)\b")
# Python/numexpr keywords and function names DataFrame.eval() resolves itself - must
# not be mistaken for missing columns and zero-filled.
_EVAL_RESERVED = {
    "and",
    "or",
    "not",
    "in",
    "is",
    "if",
    "else",
    "True",
    "False",
    "None",
    "abs",
    "sqrt",
    "log",
    "log10",
    "log1p",
    "exp",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "where",
    "arctan2",
}


def _referenced_names(expr: str) -> set[str]:
    """Bare/backtick-quoted identifiers referenced in a `DataFrame.eval()` expression."""
    names = set()
    for backtick, ident in _NAME_TOKEN_RE.findall(expr):
        name = backtick or ident
        if backtick or name not in _EVAL_RESERVED:
            names.add(name)
    return names


def _strip_backticks(expr: str) -> str:
    """Remove backtick quoting from an expression for use as a display label."""
    return expr.replace("`", "")


def _default_label(label: str | None, expr: str) -> str:
    """Axis/label default: the given `label`, or `expr` with backticks stripped."""
    return label if label is not None else _strip_backticks(expr)


class BasePlot(ABC):
    """Shared foundation for matplotlib-based petrology plots.

    Subclasses implement `_plot_group` to draw the DataFrame passed to a
    single `add()` call onto the shared axes.
    """

    def __init__(
        self,
        *,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        grid: bool = False,
        legend: bool = True,
        legend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Configure shared plot properties.

        Args:
            title: Axes title.
            figsize: Figure size in inches as ``(width, height)``. Uses
                matplotlib's rcParams default when ``None``.
            grid: Whether to draw axis gridlines.
            legend: Whether to draw a legend when at least one group has
                a label.
            legend_kwargs: Extra keyword arguments forwarded to
                ``Axes.legend``.
        """
        self.title = title
        self.figsize = figsize
        self.grid = grid
        self.legend = legend
        self.legend_kwargs = legend_kwargs or {}
        self._groups: list[tuple[pd.DataFrame, str | None, dict[str, Any]]] = []

    def add(
        self, data: pd.DataFrame, *, label: str | None = None, **style: Any
    ) -> None:
        """Register a group of samples to be plotted with one style/legend entry.

        Args:
            data: Samples in rows, variables in columns.
            label: Legend entry for this group. Groups without a label are
                still plotted but omitted from the legend.
            **style: Extra keyword arguments forwarded to the subclass's
                plot call (e.g. ``color``, ``marker`` for ``ScatterPlot``),
                overriding the automatic style cycle.
        """
        self._groups.append((data, label, style))

    def render(self) -> tuple[Figure, Axes]:
        """Build the figure from all registered groups without displaying it."""
        fig, ax = plt.subplots(
            figsize=self.figsize, constrained_layout=True, **self._subplot_kwargs()
        )
        self._prepare_axes(ax)
        for data, label, style in self._groups:
            self._plot_group(ax, data, label=label, **style)
        if self.title:
            ax.set_title(self.title)
        self._apply_axis_labels(ax)
        if self.grid:
            ax.grid(True)
        self._finalize_legend(ax)
        return fig, ax

    def show(self) -> None:
        """Build and display the figure.

        While shown, a manual window resize is captured back into
        `figsize`, so a later `savefig()`/`show()` call reuses the new size.
        """
        fig, _ = self.render()
        fig.canvas.mpl_connect("resize_event", lambda _event: self._sync_figsize(fig))
        plt.show()

    def savefig(self, *args: Any, **kwargs: Any) -> None:
        """Build and save the figure to file. Same signature as `matplotlib.pyplot.savefig`."""
        fig, _ = self.render()
        fig.savefig(*args, **kwargs)
        plt.close(fig)

    def _sync_figsize(self, fig: Figure) -> None:
        """Update `figsize` from the figure's current size in inches.

        Args:
            fig: Figure whose current size should be captured.
        """
        width, height = fig.get_size_inches()
        self.figsize = (float(width), float(height))

    def _subplot_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments for `plt.subplots()`.

        Override to select a non-default projection (e.g. ternary).
        """
        return {}

    def _prepare_axes(self, ax: Axes) -> None:
        """Set up ``ax`` before any group is plotted. No-op by default.

        Override to create extra axes (e.g. a twin axis) that `_plot_group`
        needs to already exist.

        Args:
            ax: The primary axes, freshly created by `plt.subplots()`.
        """

    def _apply_axis_labels(self, ax: Axes) -> None:
        """Label ``ax``'s axes. No-op by default; override per plot type.

        Args:
            ax: Axes to configure.
        """

    def _finalize_legend(self, ax: Axes) -> None:
        """Draw the legend once all groups are plotted.

        Placed outside the axes to the right by default (`legend_kwargs`
        can override `loc`/`bbox_to_anchor` to change that). Default
        behavior collects ``ax``'s own labeled artists. Override to combine
        handles from extra axes (e.g. a twin axis).

        Args:
            ax: The primary axes.
        """
        if self.legend and any(label for _, label, _ in self._groups):
            ax.legend(**{**_LEGEND_OUTSIDE_KWARGS, **self.legend_kwargs})

    @staticmethod
    def _eval(expr: str, data: pd.DataFrame) -> pd.Series:
        """Evaluate a column expression against a group's DataFrame.

        Args:
            expr: A column name of ``data`` (matched directly, so exotic
                names like ion notation ``"Al{3+}"`` work with no
                escaping), or a ``DataFrame.eval()`` expression — wrap
                special-character column names in backticks to combine
                them (e.g. ``` "`Al{3+}` + `Si{4+}`" ```). A name missing
                from ``data`` defaults to 0 *within a multi-term
                expression* (e.g. plotting ``"Sps+Grs"`` across mineral
                groups that don't all report ``Sps``); a single column
                reference that's entirely missing still raises.
            data: Samples in rows, variables in columns.

        Returns:
            The per-row values as a ``pandas.Series``.

        Raises:
            TypeError: If ``expr`` evaluates to something other than a
                ``pandas.Series`` (e.g. a constant expression).
        """
        stripped = expr.strip()
        if stripped in data.columns:
            result = data[stripped]
        elif _SINGLE_NAME_RE.fullmatch(stripped):
            result = data.eval(expr)
        else:
            missing = _referenced_names(expr) - set(data.columns)
            if missing:
                data = data.copy()
                for name in missing:
                    data[name] = 0.0
            result = data.eval(expr)
        if not isinstance(result, pd.Series):
            raise TypeError(
                f"Expression {expr!r} must evaluate to a pandas Series, "
                f"got {type(result).__name__}"
            )
        return result

    @abstractmethod
    def _plot_group(
        self, ax: Axes, data: pd.DataFrame, *, label: str | None, **style: Any
    ) -> None:
        """Draw a single group (one ``add()`` call) onto ``ax``.

        Args:
            ax: Axes to draw onto.
            data: Samples in rows, variables in columns, as passed to
                ``add()``.
            label: Legend entry for this group, or ``None``.
            **style: Style overrides passed to ``add()`` for this group.
        """


class ScatterPlot(BasePlot):
    """Scatter plot where x/y are `pandas.eval()` expressions over DataFrame columns."""

    def __init__(
        self,
        x: str,
        y: str,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure a scatter plot over expressions of DataFrame columns.

        Args:
            x: ``pandas.eval()`` expression evaluated against each group's
                DataFrame to produce the x-axis values, e.g. ``"Prp"`` or
                ``"Sps+Grs"``. Used as the default x-axis label.
            y: Same as ``x``, for the y-axis values.
            xlabel: X-axis label. Defaults to ``x``.
            ylabel: Y-axis label. Defaults to ``y``.
            xlim: X-axis limits as ``(min, max)``.
            ylim: Y-axis limits as ``(min, max)``.
            **kwargs: Forwarded to ``BasePlot.__init__``.
        """
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.xlabel = _default_label(xlabel, x)
        self.ylabel = _default_label(ylabel, y)
        self.xlim = xlim
        self.ylim = ylim

    def _apply_axis_labels(self, ax: Axes) -> None:
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)
        if self.xlim:
            ax.set_xlim(self.xlim)
        if self.ylim:
            ax.set_ylim(self.ylim)

    def _plot_group(
        self, ax: Axes, data: pd.DataFrame, *, label: str | None, **style: Any
    ) -> None:
        """Draw one group's DataFrame as a scatter series."""
        x_vals = self._eval(self.x, data)
        y_vals = self._eval(self.y, data)
        ax.scatter(x_vals, y_vals, label=label, **style)


_X_SCALE = 1.0 / math.sqrt(3.0)


def _project(t: Any, left: Any, r: Any) -> tuple[Any, Any]:
    """Project barycentric (t, left, r) onto the equilateral-triangle plane.

    Args:
        t: Top-axis value(s).
        left: Left-axis value(s).
        r: Right-axis value(s).

    Returns:
        ``(x, y)`` Cartesian coordinates. Scale-invariant: only the
        relative proportions of ``t``/``left``/``r`` matter, matching the
        "recalculate to 100%" ternary convention.
    """
    t = np.asarray(t, dtype=float)
    left = np.asarray(left, dtype=float)
    r = np.asarray(r, dtype=float)
    denom = t + left + r
    ft = t / denom
    fl = left / denom
    fr = r / denom
    x = (fr - fl) * _X_SCALE
    y = ft
    return x, y


def _interpolate(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    key: int,
    bound: float,
) -> tuple[float, float, float]:
    v1, v2 = p1[key], p2[key]
    frac = (bound - v1) / (v2 - v1)
    return tuple(a + frac * (b - a) for a, b in zip(p1, p2))


def _clip_polygon(
    vertices: list[tuple[float, float, float]], key: int, bound: float, keep_ge: bool
) -> list[tuple[float, float, float]]:
    """One Sutherland-Hodgman pass, clipping to ``vertex[key] >= bound`` (or ``<=``)."""
    if not vertices:
        return vertices
    result: list[tuple[float, float, float]] = []
    n = len(vertices)
    for i in range(n):
        curr = vertices[i]
        prev = vertices[i - 1]
        curr_in = curr[key] >= bound if keep_ge else curr[key] <= bound
        prev_in = prev[key] >= bound if keep_ge else prev[key] <= bound
        if curr_in:
            if not prev_in:
                result.append(_interpolate(prev, curr, key, bound))
            result.append(curr)
        elif prev_in:
            result.append(_interpolate(prev, curr, key, bound))
    return result


def _polygon_vertices(
    tlim: tuple[float, float] | None,
    llim: tuple[float, float] | None,
    rlim: tuple[float, float] | None,
    ternary_sum: float,
) -> list[tuple[float, float, float]]:
    """Visible-region vertices as ``(t, left, r)`` tuples, in boundary order.

    Args:
        tlim: Top-axis ``(min, max)`` limits, or ``None`` for the full range.
        llim: Left-axis ``(min, max)`` limits, or ``None``.
        rlim: Right-axis ``(min, max)`` limits, or ``None``.
        ternary_sum: The full-triangle value (each vertex is 100% one axis).

    Returns:
        Polygon vertices in ``(t, left, r)`` barycentric coordinates,
        exactly clipped to the requested limits (no approximation, so it
        cannot reproduce mpltern's rectangle-fit clipping bug).
    """
    vertices = [
        (ternary_sum, 0.0, 0.0),
        (0.0, 0.0, ternary_sum),
        (0.0, ternary_sum, 0.0),
    ]
    for key, lim in ((0, tlim), (1, llim), (2, rlim)):
        if lim is None:
            continue
        vmin, vmax = lim
        vertices = _clip_polygon(vertices, key, vmin, keep_ge=True)
        vertices = _clip_polygon(vertices, key, vmax, keep_ge=False)
    return vertices


def _nice_ticks(vmin: float, vmax: float) -> list[float]:
    ticks = MaxNLocator(nbins=5).tick_values(vmin, vmax)
    return [float(v) for v in ticks if vmin - 1e-9 <= v <= vmax + 1e-9]


# Ternary tick-ownership convention (verified against mpltern's own tick
# placement, both unzoomed and zoomed): every polygon edge is constant
# along exactly one barycentric key (it's either an original triangle edge
# or a limit-clipping cut). The edge where that key is at its minimum
# shows ticks for _OWNER_MIN[key]; where it's at its maximum, for
# _OWNER_MAX[key].
_OWNER_MIN = {0: 2, 1: 0, 2: 1}  # t=tmin->r, l=lmin->t, r=rmin->l
_OWNER_MAX = {0: 1, 1: 2, 2: 0}  # t=tmax->l, l=lmax->r, r=rmax->t
_EPS = 1e-9


def _polygon_centroid(polygon: list[tuple[float, float, float]]) -> tuple[float, float]:
    points = [_project(*v) for v in polygon]
    n = len(points)
    return sum(p[0] for p in points) / n, sum(p[1] for p in points) / n


# Tick-mark direction convention: ticks for axis X are drawn parallel to
# the triangle side *opposite* X's own vertex (e.g. right-side/t ticks
# parallel to the bottom L-R side), not perpendicular to the side the
# ticks sit on. These are the fixed (constant) directions of the 3 full
# -triangle sides: opposite T is L->R, opposite L is T->R, opposite R is
# T->L.
_T, _L, _R = (0.0, 1.0), (-_X_SCALE, 0.0), (_X_SCALE, 0.0)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    return (dx / length, dy / length)


def _safe_unit(
    dx: float, dy: float, default: tuple[float, float] = (0.0, 1.0)
) -> tuple[float, float]:
    """Unit vector for (dx, dy), or `default` when its length is ~0."""
    length = math.hypot(dx, dy)
    return (dx / length, dy / length) if length > _EPS else default


_TICK_DIRECTION = {
    0: _unit(_R[0] - _L[0], _R[1] - _L[1]),  # t-ticks parallel to L->R
    1: _unit(_R[0] - _T[0], _R[1] - _T[1]),  # l-ticks parallel to T->R
    2: _unit(_L[0] - _T[0], _L[1] - _T[1]),  # r-ticks parallel to T->L
}


def _edge_outward_normal(
    x1: float, y1: float, x2: float, y2: float, centroid: tuple[float, float]
) -> tuple[float, float]:
    """Unit vector perpendicular to edge (x1,y1)-(x2,y2), pointing away from `centroid`."""
    edx, edy = x2 - x1, y2 - y1
    if math.hypot(edx, edy) < _EPS:
        return 0.0, 0.0
    nx, ny = _unit(-edy, edx)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if nx * (mx - centroid[0]) + ny * (my - centroid[1]) < 0:
        nx, ny = -nx, -ny
    return nx, ny


def _outward_tick_direction(
    key: int, edge_normal: tuple[float, float]
) -> tuple[float, float]:
    """Unit vector for axis `key`'s tick marks.

    Parallel to the side opposite `key`'s own vertex, oriented toward the
    outward side of the edge the tick actually sits on (`edge_normal`) -
    using the edge's own normal to pick the sign is more robust than
    testing the tick point against the polygon centroid directly, which
    is ambiguous exactly at vertices lying near the tick direction's own
    line through the centroid.
    """
    dx, dy = _TICK_DIRECTION[key]
    if dx * edge_normal[0] + dy * edge_normal[1] < 0:
        dx, dy = -dx, -dy
    return dx, dy


def _vertex_anchor(
    polygon: list[tuple[float, float, float]], key: int, centroid: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float], bool]:
    """Anchor point and outward direction for axis `key`'s vertex label.

    When `key`'s limit cuts off its vertex, two polygon points tie for
    its maximum instead of one; centering on their midpoint (rather than
    an arbitrary one of the two) and pointing outward along that cut
    edge's own normal keeps the label centered above the cut, matching
    the un-cut case where it's centered above the single vertex.

    Returns:
        ``(anchor, direction, is_cut)`` - `anchor` is the (x, y) point to
        place the label relative to, `direction` the outward unit vector,
        and `is_cut` whether the vertex is a cut edge (two tied points)
        rather than a single point.
    """
    max_val = max(v[key] for v in polygon)
    tied = [v for v in polygon if abs(v[key] - max_val) < _EPS]
    points = [_project(*v) for v in tied]
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    if len(points) >= 2:
        normal = _edge_outward_normal(
            points[0][0], points[0][1], points[1][0], points[1][1], centroid
        )
        if math.hypot(normal[0], normal[1]) < _EPS:
            normal = _safe_unit(cx - centroid[0], cy - centroid[1])
        return (cx, cy), normal, True
    normal = _safe_unit(points[0][0] - centroid[0], points[0][1] - centroid[1])
    return (cx, cy), normal, False


def _label_rotation(dx: float, dy: float) -> float:
    """Text rotation (degrees) so a label reads along direction (dx, dy).

    Normalized to ``(-90, 90]`` so the text stays upright rather than
    upside down.
    """
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return angle


def _axis_ticks(
    polygon: list[tuple[float, float, float]], key: int, vmin: float, vmax: float
) -> list[tuple[tuple[float, float], float, tuple[float, float]]]:
    """Tick positions for one axis, placed on its owning polygon edges.

    A single axis's tick scale can be split across two different edges
    (e.g. when a different axis's limit cuts the polygon) - collecting
    every edge owned by `key` and placing ticks using each edge's own
    local value range handles that correctly.

    Args:
        polygon: Visible-region vertices, as returned by `_polygon_vertices`.
        key: 0/1/2 for t/left/r.
        vmin: Axis minimum (matches the corresponding lim, or the
            polygon's natural minimum for this key).
        vmax: Axis maximum (matches the corresponding lim, or the
            polygon's natural maximum for this key).

    Returns:
        List of ``((x, y), value, (dx, dy))`` tuples, one per nice tick
        value, where ``(dx, dy)`` is the unit vector along which the tick
        mark and label are drawn: parallel to the side opposite `key`'s
        vertex, oriented outward.
    """
    n = len(polygon)
    global_min = [min(v[k] for v in polygon) for k in range(3)]
    global_max = [max(v[k] for v in polygon) for k in range(3)]
    centroid = _polygon_centroid(polygon)

    nice = _nice_ticks(vmin, vmax)
    found: dict[float, tuple[tuple[float, float], tuple[float, float]]] = {}
    for i in range(n):
        v1, v2 = polygon[i], polygon[(i + 1) % n]
        edge_key = next((k for k in range(3) if abs(v1[k] - v2[k]) < _EPS), None)
        if edge_key is None:
            continue
        const_val = v1[edge_key]
        if abs(const_val - global_min[edge_key]) < _EPS:
            owner = _OWNER_MIN[edge_key]
        elif abs(const_val - global_max[edge_key]) < _EPS:
            owner = _OWNER_MAX[edge_key]
        else:
            continue
        if owner != key:
            continue
        x1, y1 = _project(*v1)
        x2, y2 = _project(*v2)
        edge_normal = _edge_outward_normal(x1, y1, x2, y2, centroid)
        direction = _outward_tick_direction(key, edge_normal)
        lo, hi = sorted((v1[key], v2[key]))
        for value in nice:
            rounded = round(value, 6)
            if lo - _EPS <= value <= hi + _EPS and rounded not in found:
                point = (
                    v1
                    if abs(v2[key] - v1[key]) < _EPS
                    else _interpolate(v1, v2, key, value)
                )
                x, y = _project(*point)
                found[rounded] = ((float(x), float(y)), direction)

    return [
        (found[round(v, 6)][0], v, found[round(v, 6)][1])
        for v in nice
        if round(v, 6) in found
    ]


def _axis_gridline(
    polygon: list[tuple[float, float, float]], key: int, value: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Segment where ``key == value`` crosses the polygon boundary, or `None`."""
    n = len(polygon)
    crossings: list[tuple[float, float, float]] = []
    for i in range(n):
        p1, p2 = polygon[i], polygon[(i + 1) % n]
        v1, v2 = p1[key], p2[key]
        if abs(v2 - v1) < 1e-12:
            continue
        frac = (value - v1) / (v2 - v1)
        if -1e-9 <= frac <= 1 + 1e-9:
            crossings.append(_interpolate(p1, p2, key, value))
    if len(crossings) < 2:
        return None
    p1, p2 = crossings[0], crossings[-1]
    x1, y1 = _project(p1[0], p1[1], p1[2])
    x2, y2 = _project(p2[0], p2[1], p2[2])
    return (float(x1), float(y1)), (float(x2), float(y2))


class TernaryPlot(BasePlot):
    """Ternary plot where top/left/right are `pandas.eval()` expressions over
    DataFrame columns.

    Built entirely on plain matplotlib primitives (no third-party ternary
    projection library).
    """

    _TITLE_PAD = 44  # points between the axes and the title, clearing vertex labels
    _TICK_LEN_FRAC = 0.01  # tick mark length, as a fraction of the triangle's span
    _LABEL_GAP_FRAC = 0.035  # gap between a tick/vertex and its label, as a
    # fraction of the triangle's span
    _TICK_FONTSIZE = 9
    _VERTEX_FONTSIZE = 11
    _VERTEX_GAP_MULT = 2.5  # side-vertex label gap, in units of _LABEL_GAP_FRAC
    _TOP_VERTEX_GAP_MULT = 3.3  # top vertex sits further out to clear the title
    _CUT_VERTEX_GAP_EXTRA = 1.5  # extra gap (label-gap units) for a cut vertex
    _TOP_CUT_VERTEX_GAP_EXTRA = 1.7  # additional extra gap for a cut top vertex

    def __init__(
        self,
        top: str,
        left: str,
        right: str,
        *,
        ternary_sum: float = 100.0,
        top_label: str | None = None,
        left_label: str | None = None,
        right_label: str | None = None,
        tlim: tuple[float, float] | None = None,
        llim: tuple[float, float] | None = None,
        rlim: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure a ternary plot over expressions of DataFrame columns.

        Args:
            top: ``pandas.eval()`` expression for the top-axis values, e.g.
                ``"Prp"``. Used as the default top-axis label.
            left: Same as ``top``, for the left-axis values.
            right: Same as ``top``, for the right-axis values.
            ternary_sum: The value that top/left/right are rescaled to sum
                to for each row before being placed (the "recalculate to
                100%" convention). Defaults to ``100.0``.
            top_label: Overrides the default top-axis label (``top``).
            left_label: Overrides the default left-axis label (``left``).
            right_label: Overrides the default right-axis label (``right``).
            tlim: Top-axis limits as ``(min, max)``.
            llim: Left-axis limits as ``(min, max)``.
            rlim: Right-axis limits as ``(min, max)``.
            **kwargs: Forwarded to ``BasePlot.__init__``.
        """
        super().__init__(**kwargs)
        self.top = top
        self.left = left
        self.right = right
        self.ternary_sum = ternary_sum
        self.top_label = _default_label(top_label, top)
        self.left_label = _default_label(left_label, left)
        self.right_label = _default_label(right_label, right)
        self.tlim = tlim
        self.llim = llim
        self.rlim = rlim

    def _plot_group(
        self, ax: Axes, data: pd.DataFrame, *, label: str | None, **style: Any
    ) -> None:
        """Draw one group's DataFrame as a ternary scatter series."""
        t = self._eval(self.top, data)
        left = self._eval(self.left, data)
        r = self._eval(self.right, data)
        x, y = _project(t, left, r)
        ax.scatter(x, y, label=label, **style)

    def _apply_axis_labels(self, ax: Axes) -> None:
        ax.set_aspect("equal")
        ax.axis("off")
        if self.title:
            ax.set_title(self.title, pad=self._TITLE_PAD)

        polygon = _polygon_vertices(self.tlim, self.llim, self.rlim, self.ternary_sum)
        scale = self._draw_outline(ax, polygon)
        tick_len = self._TICK_LEN_FRAC * scale
        label_gap = self._LABEL_GAP_FRAC * scale

        centroid = _polygon_centroid(polygon)
        axis_rotation = self._draw_ticks_and_gridlines(ax, polygon, tick_len, label_gap)
        self._draw_vertex_labels(
            ax, polygon, centroid, axis_rotation, tick_len, label_gap
        )

    def _draw_outline(
        self, ax: Axes, polygon: list[tuple[float, float, float]]
    ) -> float:
        """Draw the (possibly clipped) triangle outline and set axis limits.

        Returns:
            The triangle's span, used to scale tick/label offsets.
        """
        xs, ys = zip(*(_project(v[0], v[1], v[2]) for v in polygon))
        xs = [float(x) for x in xs]
        ys = [float(y) for y in ys]
        ax.plot(
            [*xs, xs[0]], [*ys, ys[0]], color="black", lw=1, zorder=3, clip_on=False
        )
        ax.set_xlim(min(xs), max(xs))
        ax.set_ylim(min(ys), max(ys))
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def _draw_ticks_and_gridlines(
        self,
        ax: Axes,
        polygon: list[tuple[float, float, float]],
        tick_len: float,
        label_gap: float,
    ) -> dict[int, float]:
        """Draw tick marks/labels and (optional) gridlines for all 3 axes.

        Returns:
            Each axis key's label rotation (degrees), reused for its vertex label.
        """
        tick_mark_xs: list[float] = []
        tick_mark_ys: list[float] = []
        gridline_xs: list[float] = []
        gridline_ys: list[float] = []
        axis_rotation: dict[int, float] = {}

        for key in (0, 1, 2):
            vmin = min(v[key] for v in polygon)
            vmax = max(v[key] for v in polygon)
            ticks = _axis_ticks(polygon, key, vmin, vmax)
            corners = {round(v, 6) for v in (vmin, vmax)}
            for (x, y), value, (nx, ny) in ticks:
                axis_rotation.setdefault(key, _label_rotation(nx, ny))
                tick_mark_xs += [x, x + nx * tick_len, float("nan")]
                tick_mark_ys += [y, y + ny * tick_len, float("nan")]
                ax.annotate(
                    f"{value:g}",
                    xy=(x + nx * label_gap, y + ny * label_gap),
                    ha="center",
                    va="center",
                    rotation=axis_rotation[key],
                    rotation_mode="anchor",
                    fontsize=self._TICK_FONTSIZE,
                    annotation_clip=False,
                    clip_on=False,
                )
                if self.grid and round(value, 6) not in corners:
                    segment = _axis_gridline(polygon, key, value)
                    if segment is not None:
                        (gx1, gy1), (gx2, gy2) = segment
                        gridline_xs += [gx1, gx2, float("nan")]
                        gridline_ys += [gy1, gy2, float("nan")]

        if gridline_xs:
            ax.plot(
                gridline_xs, gridline_ys, color="0.8", lw=0.8, zorder=1, clip_on=False
            )
        if tick_mark_xs:
            ax.plot(
                tick_mark_xs,
                tick_mark_ys,
                color="black",
                lw=0.8,
                zorder=3,
                clip_on=False,
            )
        return axis_rotation

    def _draw_vertex_labels(
        self,
        ax: Axes,
        polygon: list[tuple[float, float, float]],
        centroid: tuple[float, float],
        axis_rotation: dict[int, float],
        tick_len: float,
        label_gap: float,
    ) -> None:
        """Draw the top/left/right axis-name labels, anchored outward from
        each (possibly cut) polygon vertex."""
        vertex_gap = label_gap * self._VERTEX_GAP_MULT
        top_vertex_gap = label_gap * self._TOP_VERTEX_GAP_MULT
        cut_vertex_gap = tick_len + label_gap * self._CUT_VERTEX_GAP_EXTRA
        top_cut_vertex_gap = cut_vertex_gap + label_gap * self._TOP_CUT_VERTEX_GAP_EXTRA
        for key, text in (
            (0, self.top_label),
            (1, self.left_label),
            (2, self.right_label),
        ):
            (cx, cy), (nx, ny), is_cut = _vertex_anchor(polygon, key, centroid)
            if key == 0:
                gap = top_cut_vertex_gap if is_cut else top_vertex_gap
            else:
                gap = cut_vertex_gap if is_cut else vertex_gap
            rotation = axis_rotation.get(key, 0.0)
            ax.annotate(
                text,
                xy=(cx + nx * gap, cy + ny * gap),
                ha="center",
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=self._VERTEX_FONTSIZE,
                annotation_clip=False,
                clip_on=False,
            )


def _variance(values: list[float]) -> float:
    """Population variance of `values`, or 0.0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


class ProfilePlot(BasePlot):
    """Line plot of a DataFrame's columns against its index.

    Each `add()` call plots one line per column (colored by matplotlib's
    automatic color cycle), labeled `"{label} ({column})"`. Columns may
    optionally be split across two y-axes (`Axes.twinx()`) for readability
    when they span very different value scales.
    """

    _GAP_FRACTION = 0.08  # fraction of the shared axes height reserved as
    # a compressed gap between the primary/secondary bands, when their
    # value ranges don't overlap.

    def __init__(
        self,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        secondary_ylabel: str | None = None,
        split: str = "off",
        columns: str | list[str] | None = None,
        secondary_columns: str | list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure a profile plot of DataFrame columns against their index.

        Args:
            xlabel: X-axis label. Defaults to the first group's
                `DataFrame.index.name`, if set.
            ylabel: Primary y-axis label. When a secondary axis exists (the
                plot is split) and this isn't given, defaults to the
                space-joined names of the columns plotted on the primary
                axis.
            secondary_ylabel: Secondary y-axis label. Only shown if a
                secondary axis is created (see `split`/`secondary_columns`).
                Defaults to the space-joined names of the columns plotted
                on the secondary axis when not given.
            split: How to divide columns between the primary and secondary
                y-axis when `secondary_columns` is not given. `"off"`
                (default) plots every column (or, if `columns` is given,
                only `columns`) on the primary axis alone - no secondary
                axis is created. `"auto"` picks the 2-way split that
                minimizes within-group variance of each column's mean
                value, so columns of similar scale end up together - over
                all accumulated columns by default, or restricted to just
                `columns` when `columns` is given (see `columns`). When a
                secondary axis is created, the empty value range between
                the two groups is compressed into a small fixed-height
                gap instead of each axis independently filling the whole
                plot height (skipped when the two groups' value ranges
                actually overlap).
            columns: Explicit column names to plot (a single name may be
                given as a plain string); any other column present in the
                data is dropped, not plotted. When given without
                `secondary_columns`: if `split="auto"`, `columns` is split
                between primary/secondary by the automatic
                variance-minimizing search. Otherwise (`split="off"`),
                every column in `columns` goes on the primary axis alone -
                no secondary axis is created. When given together with
                `secondary_columns`, only columns named in one of the two
                lists are plotted at all; any other column present in the
                data is skipped. Columns named here that aren't present in
                a given group's DataFrame are silently skipped, not an
                error.
            secondary_columns: Explicit column names to plot on the
                secondary axis (a single name may be given as a plain
                string). Overrides `split` when given. See `columns` for
                how the two interact.
            **kwargs: Forwarded to `BasePlot.__init__`.
        """
        super().__init__(**kwargs)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.secondary_ylabel = secondary_ylabel
        self.split = split
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(secondary_columns, str):
            secondary_columns = [secondary_columns]
        self.columns = list(columns) if columns is not None else None
        self.secondary_columns = (
            list(secondary_columns) if secondary_columns is not None else None
        )
        self._allowed_columns: set[str] | None = None
        self._secondary_axis_columns: set[str] = set()
        self._secondary_ax: Axes | None = None
        self._colors: Any = None

    def _prepare_axes(self, ax: Axes) -> None:
        self._allowed_columns, self._secondary_axis_columns = (
            self._resolve_column_routing()
        )
        self._secondary_ax = ax.twinx() if self._secondary_axis_columns else None
        # ax and a twinx() axes each cycle colors independently, which would
        # let lines on different axes collide on the same color; drawing
        # every line from one shared cycle keeps colors unique across the
        # whole plot regardless of which axis a column lands on.
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self._colors = itertools.cycle(colors)

    def _plot_group(
        self, ax: Axes, data: pd.DataFrame, *, label: str | None, **style: Any
    ) -> None:
        """Draw one line per column, colored by a shared color cycle."""
        style.pop("color", None)
        for column in data.columns:
            if (
                self._allowed_columns is not None
                and column not in self._allowed_columns
            ):
                continue
            target = (
                self._secondary_ax if column in self._secondary_axis_columns else ax
            )
            line_label = f"{label} ({column})" if label else column
            target.plot(
                data.index,
                data[column],
                label=line_label,
                color=next(self._colors),
                **style,
            )

    def _compress_split_gap(self, ax: Axes) -> None:
        """Shrink the empty value range between the primary/secondary axes.

        Each axis normally autoscales independently to fill the whole plot
        height. When the two axes' actual plotted data ranges don't overlap,
        this instead gives each axis a proportional band of the height
        (based on its own already-autoscaled span) separated by a small
        fixed-height gap, so the plot reads like a single axis with its
        empty middle compressed rather than two axes each claiming 100% of
        the height. Left alone (independent autoscale) whenever the ranges
        actually overlap, since there's no clean gap to shrink.

        Args:
            ax: The primary axes.
        """
        secondary_ax = self._secondary_ax
        if secondary_ax is None:
            return

        primary_data_min, primary_data_max = sorted(ax.dataLim.intervaly)
        secondary_data_min, secondary_data_max = sorted(secondary_ax.dataLim.intervaly)
        if primary_data_max <= secondary_data_min:
            lo_ax, hi_ax = ax, secondary_ax
        elif secondary_data_max <= primary_data_min:
            lo_ax, hi_ax = secondary_ax, ax
        else:
            return  # ranges overlap - nothing to compress

        lo_min, lo_max = lo_ax.get_ylim()
        hi_min, hi_max = hi_ax.get_ylim()
        lo_span = lo_max - lo_min
        hi_span = hi_max - hi_min
        if lo_span <= 0 or hi_span <= 0:
            return  # degenerate axis, leave autoscale as-is

        available = 1.0 - self._GAP_FRACTION
        lo_frac = available * lo_span / (lo_span + hi_span)
        hi_frac = available - lo_frac

        lo_ax.set_ylim(lo_min, lo_min + lo_span / lo_frac)
        hi_ax.set_ylim(hi_max - hi_span / hi_frac, hi_max)

    def _apply_axis_labels(self, ax: Axes) -> None:
        self._compress_split_gap(ax)
        xlabel = self.xlabel
        if xlabel is None and self._groups:
            xlabel = self._groups[0][0].index.name
        if xlabel:
            ax.set_xlabel(xlabel)

        ylabel = self.ylabel
        secondary_ylabel = self.secondary_ylabel
        if self._secondary_ax is not None:
            plotted = self._plotted_columns()
            if ylabel is None:
                primary_cols = [
                    c for c in plotted if c not in self._secondary_axis_columns
                ]
                ylabel = " ".join(primary_cols)
            if secondary_ylabel is None:
                secondary_cols = [
                    c for c in plotted if c in self._secondary_axis_columns
                ]
                secondary_ylabel = " ".join(secondary_cols)
        if ylabel:
            ax.set_ylabel(ylabel)
        if self._secondary_ax is not None and secondary_ylabel:
            self._secondary_ax.set_ylabel(secondary_ylabel)

    def _plotted_columns(self) -> list[str]:
        """Columns actually plotted, in first-seen order."""
        columns = self._all_columns()
        if self._allowed_columns is not None:
            columns = [c for c in columns if c in self._allowed_columns]
        return columns

    def _finalize_legend(self, ax: Axes) -> None:
        if not self.legend:
            return
        handles, labels = ax.get_legend_handles_labels()
        if self._secondary_ax is not None:
            extra_handles, extra_labels = self._secondary_ax.get_legend_handles_labels()
            handles += extra_handles
            labels += extra_labels
        if not handles:
            return
        kwargs = {**_LEGEND_OUTSIDE_KWARGS, **self.legend_kwargs}
        if (
            self._secondary_ax is not None
            and "bbox_to_anchor" not in self.legend_kwargs
        ):
            kwargs["bbox_to_anchor"] = (self._legend_x_past_secondary_axis(ax), 0.5)
        ax.legend(handles, labels, **kwargs)

    def _legend_x_past_secondary_axis(self, ax: Axes) -> float:
        """X position, in `ax` axes-fraction units, just past the secondary
        y-axis's tick labels.

        The default `_LEGEND_OUTSIDE_KWARGS` offset alone isn't enough here:
        the secondary axis's own tick labels sit between `ax`'s right edge
        and that offset, so the legend would land on top of them. Forcing a
        draw pass exposes their real rendered extent to measure against.
        """
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer)
        secondary_bbox = self._secondary_ax.get_tightbbox(renderer)
        margin_px = 6.0
        overhang_px = max(secondary_bbox.x1 - ax_bbox.x1, 0.0) + margin_px
        return 1.0 + overhang_px / ax_bbox.width

    def _resolve_column_routing(self) -> tuple[set[str] | None, set[str]]:
        """Determine which columns are plotted, and which axis each goes on.

        Returns:
            ``(allowed_columns, secondary_axis_columns)`` - ``allowed_columns``
            is ``None`` when every column present should be plotted, or the
            exact set of columns to keep (dropping everything else) whenever
            `columns` is given. `secondary_axis_columns` is the set of
            columns routed to the secondary y-axis.

        Raises:
            ValueError: If `split` is neither `"auto"` nor `"off"`.
        """
        if self.columns is None:
            return None, self._resolve_secondary_columns()
        selected = set(self.columns)
        if self.secondary_columns is None:
            return selected, self._split_secondary_columns(self.columns)
        secondary = set(self.secondary_columns)
        return selected | secondary, secondary

    def _resolve_secondary_columns(self) -> set[str]:
        """Determine which columns go on the secondary axis.

        Raises:
            ValueError: If `split` is neither `"auto"` nor `"off"`.
        """
        if self.secondary_columns is not None:
            return set(self.secondary_columns)
        return self._split_secondary_columns(None)

    def _split_secondary_columns(self, candidates: list[str] | None) -> set[str]:
        """Resolve `self.split` into a secondary-axis column set.

        Only called once `self.secondary_columns is None` has already been
        established by the caller - when `secondary_columns` is given
        explicitly, `split` is ignored entirely and this is never reached.

        Args:
            candidates: Columns to restrict an `"auto"` split to, or `None`
                for every accumulated column (see `_auto_split`).

        Raises:
            ValueError: If `split` is neither `"auto"` nor `"off"`.
        """
        if self.split == "off":
            return set()
        if self.split == "auto":
            return self._auto_split(candidates)
        raise ValueError(f"split must be 'auto' or 'off', got {self.split!r}")

    def _all_columns(self) -> list[str]:
        """Column names across all accumulated groups, first-seen order."""
        seen: dict[str, None] = {}
        for data, _label, _style in self._groups:
            for column in data.columns:
                seen.setdefault(column, None)
        return list(seen)

    def _auto_split(self, candidates: list[str] | None = None) -> set[str]:
        """Split candidate columns into two axes by minimizing within-group
        variance of each column's mean value.

        The optimal 2-way partition of a set of scalar means, in the sense of
        minimizing the summed within-group variance, is always a contiguous
        split of the values in sorted order (an exchange argument shows any
        non-contiguous assignment can only be improved by moving toward
        contiguity - the same property behind 1-D k-means/Jenks natural
        breaks) - so only the `n - 1` contiguous splits of the sorted means
        need checking, not every subset.

        Args:
            candidates: Columns to consider for the split. Defaults to
                every column across all accumulated groups
                (`self._all_columns()`).
        """
        candidate_set = set(candidates) if candidates is not None else None
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for data, _label, _style in self._groups:
            for column in data.columns:
                if candidate_set is not None and column not in candidate_set:
                    continue
                values = data[column].dropna()
                sums[column] = sums.get(column, 0.0) + float(values.sum())
                counts[column] = counts.get(column, 0) + len(values)

        order = candidates if candidates is not None else self._all_columns()
        means = {c: sums[c] / counts[c] for c in order if counts.get(c)}
        columns = [c for c in order if c in means]
        if len(columns) < 2:
            return set()

        sorted_cols = sorted(columns, key=lambda c: means[c])
        best_variance: float | None = None
        best_group_b: set[str] = set()
        for k in range(1, len(sorted_cols)):
            group_a = set(sorted_cols[:k])
            group_b = set(sorted_cols[k:])
            variance = _variance([means[c] for c in group_a]) + _variance(
                [means[c] for c in group_b]
            )
            if best_variance is None or variance < best_variance:
                best_variance = variance
                # Keep the side containing the first-seen column primary.
                best_group_b = group_b if columns[0] in group_a else group_a
        return best_group_b
