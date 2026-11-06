#!/usr/bin/env python3
"""Adaptive non-uniform rectangular grid from a fine TH2.

Starts from native TH2 bins.  Merges axis-aligned blocks that contain
**negative** yields until every cell has content >= 0 (strict).

## Production algorithm (default): ``residual_local`` (mega-free)
1. **Soft strip-expand** into free bins only (exclusive claim).  Clears the
   easy negatives with small merges; residual free negs may remain where
   positives were already claimed by earlier soft merges.
2. **Residual local dissolve** (second stage) — NO soft-block bbox cascade:
   for each residual free negative, strip-grow / expanding-square, dissolve
   only soft that intersects the clearing path.  **Frozen** residual claims
   that block a later seed are **unfrozen** (cascade bbox of the min-clear
   region) and the region is re-committed via recursive non-neg bipartition.
3. **HARD_MAX** = max(400, n_fine//20) ≈ 5% of the map: no merge cell may
   exceed it.  Oversized sum≥0 regions are split by bipartition / strip-split.

## Legacy (do not use for production)
``force_clear_legacy`` — old second stage that bbox-unions every soft block
intersecting a residual grow path.  Empirically swallows clean regions
(e.g. m400 DNN<-3 ∧ HME<400 with 0 residual negs absorbed into a 14k-bin
mega-cell).  Kept only for A/B comparison.

``soft`` — stage 1 only (residual negs allowed as singletons; inspection).

Positive/empty fine bins stay singletons unless absorbed.  Each cell is
[xmin, xmax) × [ymin, ymax) with summed content and quadrature-combined error.

CLI writes:
  - JSON grid definition (list of cells + metadata)
  - PDF overview (fine content + final cell outlines)

Importable API::

  from grid_merge import build_grid_from_file, build_grid_from_th2, Grid

Example::

  python3 -u grid_merge.py \\
      --input hadd_m500_res2b.root --hist plots_2d/DY_2d \\
      --output-json m500_grid.json --output-pdf m500_grid.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import ROOT

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

ROOT.gROOT.SetBatch(True)


# --------------------------------------------------------------------------- data model


class Cell(object):
    """One rectangle of the non-uniform grid (fine bin index box [ix0,ix1) x [iy0,iy1))."""

    __slots__ = (
        "id",
        "ix0",
        "ix1",
        "iy0",
        "iy1",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "content",
        "error",
        "merged",
    )

    def __init__(
        self,
        id,
        ix0,
        ix1,
        iy0,
        iy1,
        xmin,
        xmax,
        ymin,
        ymax,
        content,
        error,
        merged,
    ):
        self.id = int(id)
        self.ix0 = int(ix0)
        self.ix1 = int(ix1)
        self.iy0 = int(iy0)
        self.iy1 = int(iy1)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.content = float(content)
        self.error = float(error)
        self.merged = bool(merged)

    @property
    def n_fine(self):
        return (self.ix1 - self.ix0) * (self.iy1 - self.iy0)

    def to_dict(self):
        return {
            "id": self.id,
            "xmin": self.xmin,
            "xmax": self.xmax,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "content": self.content,
            "error": self.error,
            "ix0": self.ix0,
            "ix1": self.ix1,
            "iy0": self.iy0,
            "iy1": self.iy1,
            "n_fine": self.n_fine,
            "merged": self.merged,
        }


class Grid(object):
    """Non-uniform rectangular partition of a fine TH2."""

    def __init__(
        self,
        cells,
        x_edges,
        y_edges,
        meta=None,
        fine_content=None,
        fine_error=None,
    ):
        self.cells = list(cells)
        self.x_edges = np.asarray(x_edges, dtype=float)
        self.y_edges = np.asarray(y_edges, dtype=float)
        self.meta = dict(meta or {})
        # Original fine TH2 (for plots / validation cross-checks)
        self.fine_content = (
            None if fine_content is None else np.asarray(fine_content, dtype=float)
        )
        self.fine_error = (
            None if fine_error is None else np.asarray(fine_error, dtype=float)
        )

    @property
    def n_cells(self):
        return len(self.cells)

    def to_dict(self):
        d = dict(self.meta)
        d["n_cells"] = self.n_cells
        d["n_merged"] = sum(1 for c in self.cells if c.merged)
        d["n_singleton"] = sum(1 for c in self.cells if not c.merged)
        d["x_edges_fine"] = [float(x) for x in self.x_edges]
        d["y_edges_fine"] = [float(y) for y in self.y_edges]
        d["cells"] = [c.to_dict() for c in self.cells]
        return d


class GridValidationError(RuntimeError):
    """Raised when the grid partition is inconsistent with the fine TH2."""


def validate_grid(grid, fine_content=None):
    """Check (a) no overlapping cells and (b) full coverage of the fine grid.

    Also checks that every cell is a non-empty axis-aligned fine-index
    rectangle aligned to the TH2 edges, and that cell contents match the
    sum of fine bins when fine_content is available.

    Raises GridValidationError on failure; returns a stats dict on success.
    """
    nx = int(grid.meta.get("nx_fine") or (len(grid.x_edges) - 1))
    ny = int(grid.meta.get("ny_fine") or (len(grid.y_edges) - 1))
    if len(grid.x_edges) != nx + 1 or len(grid.y_edges) != ny + 1:
        raise GridValidationError(
            "edge length mismatch: x_edges=%d (need %d), y_edges=%d (need %d)"
            % (len(grid.x_edges), nx + 1, len(grid.y_edges), ny + 1)
        )

    cover = np.zeros((nx, ny), dtype=np.int32)
    issues = []
    for c in grid.cells:
        if c.ix1 <= c.ix0 or c.iy1 <= c.iy0:
            issues.append(
                "cell %d has empty index range [%d:%d]x[%d:%d]"
                % (c.id, c.ix0, c.ix1, c.iy0, c.iy1)
            )
            continue
        if c.ix0 < 0 or c.iy0 < 0 or c.ix1 > nx or c.iy1 > ny:
            issues.append(
                "cell %d out of bounds [%d:%d]x[%d:%d] vs fine %dx%d"
                % (c.id, c.ix0, c.ix1, c.iy0, c.iy1, nx, ny)
            )
            continue
        # Edge alignment (cell is a true axis-aligned rectangle of fine bins)
        if (
            abs(c.xmin - float(grid.x_edges[c.ix0])) > 1e-9
            or abs(c.xmax - float(grid.x_edges[c.ix1])) > 1e-9
        ):
            issues.append(
                "cell %d x-edges mismatch fine edges (%.6g,%.6g) vs (%.6g,%.6g)"
                % (
                    c.id,
                    c.xmin,
                    c.xmax,
                    float(grid.x_edges[c.ix0]),
                    float(grid.x_edges[c.ix1]),
                )
            )
        if (
            abs(c.ymin - float(grid.y_edges[c.iy0])) > 1e-9
            or abs(c.ymax - float(grid.y_edges[c.iy1])) > 1e-9
        ):
            issues.append(
                "cell %d y-edges mismatch fine edges (%.6g,%.6g) vs (%.6g,%.6g)"
                % (
                    c.id,
                    c.ymin,
                    c.ymax,
                    float(grid.y_edges[c.iy0]),
                    float(grid.y_edges[c.iy1]),
                )
            )
        cover[c.ix0 : c.ix1, c.iy0 : c.iy1] += 1

    n_uncovered = int(np.sum(cover == 0))
    n_overlap = int(np.sum(cover > 1))
    if n_uncovered:
        issues.append(
            "coverage incomplete: %d fine bins uncovered (need union = full TH2)"
            % n_uncovered
        )
    if n_overlap:
        issues.append(
            "cells overlap: %d fine bins covered more than once (max cover=%d)"
            % (n_overlap, int(cover.max()))
        )

    fc = fine_content if fine_content is not None else grid.fine_content
    if fc is not None:
        fc = np.asarray(fc, dtype=float)
        if fc.shape != (nx, ny):
            issues.append("fine_content shape %s != (%d,%d)" % (fc.shape, nx, ny))
        else:
            for c in grid.cells:
                expect = float(fc[c.ix0 : c.ix1, c.iy0 : c.iy1].sum())
                if abs(expect - c.content) > 1e-6 * max(1.0, abs(expect)):
                    issues.append(
                        "cell %d content %.6g != fine sum %.6g"
                        % (c.id, c.content, expect)
                    )
                    if len(issues) > 20:
                        issues.append("... further content mismatches truncated")
                        break

    if issues:
        raise GridValidationError(
            "grid validation failed (%d issues):\n  - %s"
            % (len(issues), "\n  - ".join(issues[:30]))
        )

    return {
        "ok": True,
        "nx_fine": nx,
        "ny_fine": ny,
        "n_cells": len(grid.cells),
        "n_uncovered": 0,
        "n_overlap": 0,
        "cover_min": int(cover.min()) if cover.size else 0,
        "cover_max": int(cover.max()) if cover.size else 0,
    }


# --------------------------------------------------------------------------- TH2 → arrays


def th2_to_arrays(h):
    """Return content, error (nx,ny), x_edges (nx+1), y_edges (ny+1)."""
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    content = np.zeros((nx, ny), dtype=float)
    error = np.zeros((nx, ny), dtype=float)
    x_edges = np.zeros(nx + 1)
    y_edges = np.zeros(ny + 1)
    for ix in range(1, nx + 1):
        x_edges[ix - 1] = h.GetXaxis().GetBinLowEdge(ix)
    x_edges[nx] = h.GetXaxis().GetBinUpEdge(nx)
    for iy in range(1, ny + 1):
        y_edges[iy - 1] = h.GetYaxis().GetBinLowEdge(iy)
    y_edges[ny] = h.GetYaxis().GetBinUpEdge(ny)
    for ix in range(1, nx + 1):
        for iy in range(1, ny + 1):
            content[ix - 1, iy - 1] = h.GetBinContent(ix, iy)
            error[ix - 1, iy - 1] = h.GetBinError(ix, iy)
    return content, error, x_edges, y_edges


def load_th2(path, hist_path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise FileNotFoundError("Cannot open %s" % path)
    h = f.Get(hist_path)
    if not h:
        f.Close()
        raise KeyError("Histogram '%s' not found in %s" % (hist_path, path))
    h.SetDirectory(0)
    # PyROOT does not own what Get() returns; detached, the histogram would be owned
    # by nobody, never freed
    ROOT.SetOwnership(h, True)
    f.Close()
    return h


# --------------------------------------------------------------------------- merge algorithm


def _bbox_err2(error, ix0, ix1, iy0, iy1):
    e = error[ix0:ix1, iy0:iy1]
    return float(np.sum(e * e))


def _prefix_2d(content):
    """P[i+1, j+1] = sum of content[0:i, 0:j] (half-open)."""
    nx, ny = content.shape
    P = np.zeros((nx + 1, ny + 1), dtype=float)
    for i in range(nx):
        row = 0.0
        for j in range(ny):
            row += float(content[i, j])
            P[i + 1, j + 1] = P[i, j + 1] + row
    return P


def _rect_sum(P, ix0, ix1, iy0, iy1):
    return float(P[ix1, iy1] - P[ix0, iy1] - P[ix1, iy0] + P[ix0, iy0])


def _rects_intersect(a, b):
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _rebuild_claimed(blocks, nx, ny):
    claimed = np.zeros((nx, ny), dtype=bool)
    for ix0, ix1, iy0, iy1 in blocks:
        claimed[ix0:ix1, iy0:iy1] = True
    return claimed


def _tighten_rect(content, ix0, ix1, iy0, iy1, sx, sy):
    """Peel empty/unnecessary borders while sum stays >=0 and seed stays inside.

    Soft strip-grow often walks through vast near-zero empty regions to reach a
    distant positive.  The resulting rectangle is valid (sum>=0) but visually a
    mega-block.  Tightening collapses it to a minimal clearing rect.
    """
    content = np.asarray(content, dtype=float)
    assert ix0 <= sx < ix1 and iy0 <= sy < iy1
    # safety: do not shrink a non-clearing seed rect
    if float(content[ix0:ix1, iy0:iy1].sum()) < 0.0:
        return ix0, ix1, iy0, iy1
    changed = True
    while changed:
        changed = False
        if ix1 - ix0 > 1 and ix0 < sx:
            if float(content[ix0 + 1 : ix1, iy0:iy1].sum()) >= 0.0:
                ix0 += 1
                changed = True
                continue
        if ix1 - ix0 > 1 and ix1 - 1 > sx:
            if float(content[ix0 : ix1 - 1, iy0:iy1].sum()) >= 0.0:
                ix1 -= 1
                changed = True
                continue
        if iy1 - iy0 > 1 and iy0 < sy:
            if float(content[ix0:ix1, iy0 + 1 : iy1].sum()) >= 0.0:
                iy0 += 1
                changed = True
                continue
        if iy1 - iy0 > 1 and iy1 - 1 > sy:
            if float(content[ix0:ix1, iy0 : iy1 - 1].sum()) >= 0.0:
                iy1 -= 1
                changed = True
                continue
    return ix0, ix1, iy0, iy1


def _grow_rectangle(content, claimed, ix, iy, mode="greedy", respect_claimed=True):
    """Grow a solid rectangle from seed by whole strips until sum>=0.

    Fallback / single-path candidate.  mode 'greedy' prefers max strip add
    among non-clearing steps; 'mild' prefers low mean when clearing.
    Always tightens the result so empty mega-strips collapse.
    """
    nx, ny = content.shape
    if respect_claimed:
        assert not claimed[ix, iy]
    ix0, ix1, iy0, iy1 = ix, ix + 1, iy, iy + 1

    def sm():
        return float(content[ix0:ix1, iy0:iy1].sum())

    def strip_ok(sx0, sx1, sy0, sy1):
        if not respect_claimed:
            return True
        return not claimed[sx0:sx1, sy0:sy1].any()

    for _ in range(nx + ny + 5):
        s = sm()
        if s >= 0.0:
            return _tighten_rect(content, ix0, ix1, iy0, iy1, ix, iy)
        opts = []
        if ix0 > 0 and strip_ok(ix0 - 1, ix0, iy0, iy1):
            strip = content[ix0 - 1, iy0:iy1]
            opts.append((float(strip.sum()), float(strip.mean()), "L"))
        if ix1 < nx and strip_ok(ix1, ix1 + 1, iy0, iy1):
            strip = content[ix1, iy0:iy1]
            opts.append((float(strip.sum()), float(strip.mean()), "R"))
        if iy0 > 0 and strip_ok(ix0, ix1, iy0 - 1, iy0):
            strip = content[ix0:ix1, iy0 - 1]
            opts.append((float(strip.sum()), float(strip.mean()), "D"))
        if iy1 < ny and strip_ok(ix0, ix1, iy1, iy1 + 1):
            strip = content[ix0:ix1, iy1]
            opts.append((float(strip.sum()), float(strip.mean()), "U"))
        if not opts:
            return ix0, ix1, iy0, iy1

        def key(o):
            add, mean, _d = o
            clears = 0 if s + add >= 0.0 else 1
            mean_pen = mean if mean > 0.0 else 0.0
            if mode == "greedy":
                return (clears, -add, mean_pen)
            if clears == 0:
                return (0, mean_pen, -add)
            return (1, -add, mean_pen)

        opts.sort(key=key)
        _a, _m, d = opts[0]
        if d == "L":
            ix0 -= 1
        elif d == "R":
            ix1 += 1
        elif d == "D":
            iy0 -= 1
        else:
            iy1 += 1
    return _tighten_rect(content, ix0, ix1, iy0, iy1, ix, iy)


def _expanding_square_clear(P, sx, sy, nx, ny):
    """Smallest Chebyshev-ball rectangle around seed with sum >= 0, or None."""
    for r in range(0, max(nx, ny) + 1):
        ix0, ix1 = max(0, sx - r), min(nx, sx + r + 1)
        iy0, iy1 = max(0, sy - r), min(ny, sy + r + 1)
        if _rect_sum(P, ix0, ix1, iy0, iy1) >= 0.0:
            return (ix0, ix1, iy0, iy1)
    return None


def _strip_extension_candidates(
    P,
    content,
    sx,
    sy,
    *,
    max_candidates=12,
    max_expansions=40000,
    area_slack_factor=2.5,
    area_slack_abs=40,
):
    """Strip-extension clearing rectangles for one seed (independent).

    BFS over axis-aligned rectangles containing the seed, expanding by one
    full strip (L/R/D/U) at a time.  A state with sum >= 0 is recorded as a
    candidate and not expanded further (growth-minimal along that path).

    Also always includes the greedy/mild strip-grow paths and the expanding
    square clear, so hard seeds still get at least one solution.
    """
    import heapq

    nx, ny = content.shape
    cands = {}  # rect -> area

    def add(rect):
        ix0, ix1, iy0, iy1 = rect
        if ix1 <= ix0 or iy1 <= iy0:
            return
        if not (ix0 <= sx < ix1 and iy0 <= sy < iy1):
            return
        sm = _rect_sum(P, ix0, ix1, iy0, iy1)
        if sm < 0.0:
            return
        area = (ix1 - ix0) * (iy1 - iy0)
        prev = cands.get(rect)
        if prev is None or area < prev:
            cands[rect] = area

    # --- BFS by area ---
    start = (sx, sx + 1, sy, sy + 1)
    heap = [(1, start)]
    seen = {start}
    min_area = None
    expansions = 0
    while heap and expansions < max_expansions and len(cands) < max_candidates:
        area, rect = heapq.heappop(heap)
        ix0, ix1, iy0, iy1 = rect
        if min_area is not None and area > max(
            min_area * area_slack_factor, min_area + area_slack_abs
        ):
            continue
        sm = _rect_sum(P, ix0, ix1, iy0, iy1)
        if sm >= 0.0:
            add(rect)
            if min_area is None:
                min_area = area
            continue
        for nxt in (
            (ix0 - 1, ix1, iy0, iy1),
            (ix0, ix1 + 1, iy0, iy1),
            (ix0, ix1, iy0 - 1, iy1),
            (ix0, ix1, iy0, iy1 + 1),
        ):
            a, b, c, d = nxt
            if a < 0 or c < 0 or b > nx or d > ny:
                continue
            if nxt in seen:
                continue
            seen.add(nxt)
            heapq.heappush(heap, ((b - a) * (d - c), nxt))
        expansions += 1

    # --- guaranteed single-path fallbacks ---
    dummy = np.zeros((nx, ny), dtype=bool)
    for mode in ("greedy", "mild"):
        g = _grow_rectangle(content, dummy, sx, sy, mode=mode, respect_claimed=False)
        if _rect_sum(P, *g) >= 0.0:
            add(g)
    sq = _expanding_square_clear(P, sx, sy, nx, ny)
    if sq is not None:
        add(sq)

    if not cands:
        if float(content.sum()) >= 0.0:
            add((0, nx, 0, ny))
    return sorted(cands.keys(), key=lambda r: cands[r])


def _seeds_covered_by_rect(seed_map, rect):
    """seed_map[ix,iy] = seed index+1 (0 = not a seed)."""
    ix0, ix1, iy0, iy1 = rect
    sub = seed_map[ix0:ix1, iy0:iy1]
    ids = sub[sub > 0]
    if ids.size == 0:
        return frozenset()
    return frozenset(int(v) - 1 for v in np.unique(ids))


def _select_orthogonal_merges(content, P, seeds, seed_candidates):
    """Greedy pack of independent candidates + min-clears; residual drop-in.

    Used when algorithm='independent'.  Falls back to pairwise expand for
    leftovers.  Prefer algorithm='force_clear' (default) for production.
    """
    nx, ny = content.shape
    n_seeds = len(seeds)
    if n_seeds == 0:
        return []

    seed_map = np.zeros((nx, ny), dtype=np.int32)
    for k, (x, y) in enumerate(seeds):
        seed_map[int(x), int(y)] = k + 1

    def seeds_in(rect):
        sub = seed_map[rect[0] : rect[1], rect[2] : rect[3]]
        ids = np.unique(sub[sub > 0])
        return set(int(v) - 1 for v in ids)

    def min_clear(s_idx):
        sx, sy = int(seeds[s_idx][0]), int(seeds[s_idx][1])
        dummy = np.zeros((nx, ny), dtype=bool)
        g = _grow_rectangle(
            content, dummy, sx, sy, mode="greedy", respect_claimed=False
        )
        if _rect_sum(P, *g) >= 0.0:
            return g
        sq = _expanding_square_clear(P, sx, sy, nx, ny)
        return sq if sq is not None else (0, nx, 0, ny)

    def area(r):
        return (r[1] - r[0]) * (r[3] - r[2])

    items = []
    seen = set()
    for s_idx, rects in enumerate(seed_candidates):
        for rect in list(rects) + [min_clear(s_idx)]:
            if rect in seen:
                continue
            seen.add(rect)
            if _rect_sum(P, *rect) < 0.0:
                continue
            items.append((area(rect), rect))
    items.sort()
    claimed = np.zeros((nx, ny), dtype=bool)
    selected = []
    for _a, rect in items:
        ix0, ix1, iy0, iy1 = rect
        if claimed[ix0:ix1, iy0:iy1].any():
            continue
        claimed[ix0:ix1, iy0:iy1] = True
        selected.append(rect)

    def rebuild():
        claimed[:, :] = False
        for rect in selected:
            claimed[rect[0] : rect[1], rect[2] : rect[3]] = True

    def uncovered():
        cov = set()
        for rect in selected:
            cov |= seeds_in(rect)
        return [s for s in range(n_seeds) if s not in cov]

    # Residual: for each uncovered seed, claim min-clear dropping conflicts
    # Process small-first once, then a second sweep for re-opened seeds.
    for _sweep in range(3):
        unc = uncovered()
        if not unc:
            break
        unc.sort(key=lambda s: area(min_clear(s)))
        for s in unc:
            cov = set()
            for rect in selected:
                cov |= seeds_in(rect)
            if s in cov:
                continue
            r = min_clear(s)
            selected = [t for t in selected if not _rects_intersect(t, r)]
            selected.append(r)
            rebuild()

    # Geometric validation
    check = np.zeros((nx, ny), dtype=np.int32)
    for rect in selected:
        check[rect[0] : rect[1], rect[2] : rect[3]] += 1
    if int(check.max()) > 1:
        kept = []
        check[:, :] = 0
        for rect in selected:
            if check[rect[0] : rect[1], rect[2] : rect[3]].any():
                continue
            check[rect[0] : rect[1], rect[2] : rect[3]] = 1
            kept.append(rect)
        selected = kept
        check[:, :] = 0
        for rect in selected:
            check[rect[0] : rect[1], rect[2] : rect[3]] += 1
    # Cover any remaining seeds; freeze each placed min-clear so later seeds
    # cannot drop it (they expand the frozen rect instead).
    frozen = set(selected)  # freeze greedy pack results
    for s in range(n_seeds):
        x, y = int(seeds[s][0]), int(seeds[s][1])
        if any(r[0] <= x < r[1] and r[2] <= y < r[3] for r in selected):
            continue
        r = min_clear(s)
        fr_hits = [t for t in selected if t in frozen and _rects_intersect(t, r)]
        if fr_hits:
            fr_hits.sort(key=area)
            base = fr_hits[0]
            # expand base to include seed
            sx, sy = x, y
            ix0 = min(base[0], sx)
            ix1 = max(base[1], sx + 1)
            iy0 = min(base[2], sy)
            iy1 = max(base[3], sy + 1)
            exp = (ix0, ix1, iy0, iy1)
            if _rect_sum(P, *exp) < 0.0:
                dummy = np.zeros((nx, ny), dtype=bool)
                exp = _grow_rectangle(
                    content, dummy, sx, sy, mode="greedy", respect_claimed=False
                )
                # union with base
                exp = (
                    min(exp[0], base[0]),
                    max(exp[1], base[1]),
                    min(exp[2], base[2]),
                    max(exp[3], base[3]),
                )
                if _rect_sum(P, *exp) < 0.0:
                    exp = (0, nx, 0, ny)
            selected = [
                t for t in selected if t != base and not _rects_intersect(t, exp)
            ]
            frozen.discard(base)
            frozen.intersection_update(selected)
            selected.append(exp)
            frozen.add(exp)
        else:
            selected = [
                t for t in selected if t in frozen or not _rects_intersect(t, r)
            ]
            frozen.intersection_update(selected)
            selected.append(r)
            frozen.add(r)

    check = np.zeros((nx, ny), dtype=np.int32)
    for rect in selected:
        check[rect[0] : rect[1], rect[2] : rect[3]] += 1
    if int(check.max()) > 1:
        kept = []
        check[:, :] = 0
        for rect in selected:
            if check[rect[0] : rect[1], rect[2] : rect[3]].any():
                continue
            check[rect[0] : rect[1], rect[2] : rect[3]] = 1
            kept.append(rect)
        selected = kept
        check[:, :] = 0
        for rect in selected:
            check[rect[0] : rect[1], rect[2] : rect[3]] += 1
    for s in range(n_seeds):
        x, y = int(seeds[s][0]), int(seeds[s][1])
        if check[x, y] != 1:
            # last resort full map for independent algorithm
            return [(0, nx, 0, ny)]
    return list(selected)


def _min_clear_rect(content, P, sx, sy):
    """Minimal claim-unaware clearing rectangle for seed (sx, sy).

    Tries expanding-square and mild/greedy strip-grow; returns the
    smallest-area rect with sum >= 0.  Does NOT consult soft claims — used
    by residual stage to decide *which* soft blocks must be dissolved.
    """
    content = np.asarray(content, dtype=float)
    nx, ny = content.shape
    cands = []
    sq = _expanding_square_clear(P, sx, sy, nx, ny)
    if sq is not None and _rect_sum(P, *sq) >= 0.0:
        cands.append(sq)
    dummy = np.zeros((nx, ny), dtype=bool)
    for mode in ("mild", "greedy"):
        g = _grow_rectangle(content, dummy, sx, sy, mode=mode, respect_claimed=False)
        if _rect_sum(P, *g) >= 0.0:
            cands.append(tuple(g))
    if not cands:
        if float(content.sum()) >= 0.0:
            return (0, nx, 0, ny)
        raise RuntimeError(
            "no clearing rect for seed (%d,%d); total integral negative" % (sx, sy)
        )

    def area(r):
        return (r[1] - r[0]) * (r[3] - r[2])

    return min(cands, key=area)


def _soft_strip_expand(content, claimed=None, merged_blocks=None):
    """Soft strip-expand only (exclusive claim into free bins).

    Optional ``claimed`` / ``merged_blocks`` continue from a prior partial
    partition (residual stage re-soft after local dissolve).

    Returns (merged_blocks, claimed, n_neg_before, n_residual_free_neg).
    Residual free negatives may remain (walled by earlier claims).
    """
    content = np.asarray(content, dtype=float)
    nx, ny = content.shape
    n_neg_before = int(np.sum(content < 0))
    if claimed is None:
        claimed = np.zeros((nx, ny), dtype=bool)
    else:
        claimed = np.asarray(claimed, dtype=bool).copy()
    if merged_blocks is None:
        merged_blocks = []
    else:
        merged_blocks = list(merged_blocks)

    def free_neg_coords():
        return list(zip(*np.where((content < 0.0) & (~claimed))))

    def try_claim(ix, iy, mode):
        if claimed[ix, iy]:
            return False
        ix0, ix1, iy0, iy1 = _grow_rectangle(
            content, claimed, ix, iy, mode=mode, respect_claimed=True
        )
        if claimed[ix0:ix1, iy0:iy1].any():
            return False
        sm = float(content[ix0:ix1, iy0:iy1].sum())
        if sm < 0.0:
            return False
        claimed[ix0:ix1, iy0:iy1] = True
        merged_blocks.append((ix0, ix1, iy0, iy1))
        return True

    for mode, order in (
        ("mild", "easy"),
        ("mild", "hard"),
        ("greedy", "hard"),
    ):
        seeds = free_neg_coords()
        if order == "easy":
            seeds.sort(key=lambda p: content[p[0], p[1]], reverse=True)
        else:
            seeds.sort(key=lambda p: content[p[0], p[1]])
        for ix, iy in seeds:
            if claimed[ix, iy] or content[ix, iy] >= 0.0:
                continue
            try_claim(ix, iy, mode)

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for ix, iy in free_neg_coords():
        if claimed[ix, iy]:
            continue
        best = None
        for di, dj in dirs:
            ni, nj = ix + di, iy + dj
            if ni < 0 or nj < 0 or ni >= nx or nj >= ny:
                continue
            if claimed[ni, nj]:
                continue
            sm = float(content[ix, iy] + content[ni, nj])
            if sm < 0.0:
                continue
            key = (float(content[ni, nj]), ni, nj)
            if best is None or key < best[0]:
                best = (key, ni, nj)
        if best is None:
            continue
        _k, ni, nj = best
        ix0, ix1 = min(ix, ni), max(ix, ni) + 1
        iy0, iy1 = min(iy, nj), max(iy, nj) + 1
        if claimed[ix0:ix1, iy0:iy1].any():
            continue
        if float(content[ix0:ix1, iy0:iy1].sum()) < 0.0:
            continue
        claimed[ix0:ix1, iy0:iy1] = True
        merged_blocks.append((ix0, ix1, iy0, iy1))

    for ix, iy in free_neg_coords():
        if claimed[ix, iy]:
            continue
        try_claim(ix, iy, "greedy")

    n_residual = len(free_neg_coords())
    return merged_blocks, claimed, n_neg_before, n_residual


def _bipartition_nonneg(content, rect, max_keep):
    """Recursively split a sum≥0 rect into parts each of area ≤ max_keep.

    At each step pick a horizontal or vertical cut that keeps *both* children
    non-negative and minimises the larger child area.  Falls back to
    :func:`_strip_split_nonneg` when no bipartition cut exists.  If a part is
    still larger than ``max_keep`` and indecomposable, it is returned as-is
    (caller may WARN); such cases are rare on DY maps.
    """
    content = np.asarray(content, dtype=float)
    ix0, ix1, iy0, iy1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    area = (ix1 - ix0) * (iy1 - iy0)
    if area <= max_keep:
        return [(ix0, ix1, iy0, iy1)]
    tot = float(content[ix0:ix1, iy0:iy1].sum())
    if tot < -1e-12:
        return [(ix0, ix1, iy0, iy1)]

    best = None  # (score, kind, idx)
    for i in range(ix0 + 1, ix1):
        s1 = float(content[ix0:i, iy0:iy1].sum())
        s2 = tot - s1
        if s1 >= -1e-12 and s2 >= -1e-12:
            sc = max((i - ix0) * (iy1 - iy0), (ix1 - i) * (iy1 - iy0))
            cand = (sc, "v", i)
            if best is None or cand[0] < best[0]:
                best = cand
    for j in range(iy0 + 1, iy1):
        s1 = float(content[ix0:ix1, iy0:j].sum())
        s2 = tot - s1
        if s1 >= -1e-12 and s2 >= -1e-12:
            sc = max((ix1 - ix0) * (j - iy0), (ix1 - ix0) * (iy1 - j))
            cand = (sc, "h", j)
            if best is None or cand[0] < best[0]:
                best = cand

    if best is not None:
        _sc, kind, idx = best
        if kind == "v":
            left = (ix0, idx, iy0, iy1)
            right = (idx, ix1, iy0, iy1)
        else:
            left = (ix0, ix1, iy0, idx)
            right = (ix0, ix1, idx, iy1)
        # Only accept the cut if it actually reduces the larger child vs parent
        # (always true for a proper cut) and recurse.
        return _bipartition_nonneg(content, left, max_keep) + _bipartition_nonneg(
            content, right, max_keep
        )

    # No bipartition: try strip-split into non-neg slabs, then recurse.
    parts = _strip_split_nonneg(content, (ix0, ix1, iy0, iy1))
    if parts and len(parts) > 1:
        max_p = max((p[1] - p[0]) * (p[3] - p[2]) for p in parts)
        if max_p < area:
            out = []
            for p in parts:
                out.extend(_bipartition_nonneg(content, p, max_keep))
            return out
    return [(ix0, ix1, iy0, iy1)]


def _build_grid_residual_local(content, error, x_edges, y_edges):
    """Soft + residual stage with a hard mega-cell ban.

    HARD_MAX = max(400, n_fine/20) ≈ 5% of the map.  No single frozen (or
    soft residual) cell is ever allowed to exceed HARD_MAX.

    Stage 1: soft strip-expand (exclusive free-bin claim).
    Stage 2: clear residual free negatives (small expanding-square first):
      1. exclusive strip-grow + freeze if area ≤ HARD_MAX
      2. dissolve soft in expanding-square + re-try grow / freeze square
      3. **unfreeze** residual frozen blockers that intersect the min-clear
         region (cascade bbox expand), dissolve soft, then commit via
         non-neg bipartition into parts each ≤ HARD_MAX
      4. ring-expand with the same unfreeze+bipartition commit
    Stage 3: post-pass bipartition any leftover oversize soft/frozen block.
    """
    content = np.asarray(content, dtype=float)
    nx, ny = content.shape
    ntot = nx * ny
    P = _prefix_2d(content)
    soft_blocks, claimed, n_neg_before, n_res0 = _soft_strip_expand(content)
    frozen_blocks = []
    n_force_dissolves = 0
    n_unfreezes = 0
    force_iters = 0
    HARD_MAX = max(400, ntot // 20)
    print(
        "[grid_merge] soft done: merge_blocks=%d  residual_free_neg=%d  "
        "HARD_MAX=%d (%.1f%% of map)"
        % (len(soft_blocks), n_res0, HARD_MAX, 100.0 * HARD_MAX / ntot),
        flush=True,
    )

    def rebuild_claimed():
        nonlocal claimed
        claimed = _rebuild_claimed(soft_blocks + frozen_blocks, nx, ny)

    def free_neg_coords():
        return list(zip(*np.where((content < 0.0) & (~claimed))))

    def area(r):
        return (r[1] - r[0]) * (r[3] - r[2])

    def as_rect(R):
        return (int(R[0]), int(R[1]), int(R[2]), int(R[3]))

    def dissolve_soft_intersecting(rects):
        nonlocal soft_blocks, n_force_dissolves
        if not rects:
            return 0
        kept = []
        nd = 0
        for b in soft_blocks:
            if any(_rects_intersect(b, r) for r in rects):
                nd += 1
            else:
                kept.append(b)
        if nd:
            n_force_dissolves += nd
            soft_blocks = kept
            rebuild_claimed()
        return nd

    def strip_hits_frozen(srect):
        return any(_rects_intersect(b, srect) for b in frozen_blocks)

    def freeze_rect(R):
        """Freeze one rect if free, sum≥0, and area ≤ HARD_MAX."""
        nonlocal frozen_blocks
        R = as_rect(R)
        if area(R) > HARD_MAX:
            return False
        ix0, ix1, iy0, iy1 = R
        if claimed[ix0:ix1, iy0:iy1].any():
            return False
        if float(content[ix0:ix1, iy0:iy1].sum()) < 0.0:
            return False
        frozen_blocks.append(R)
        claimed[ix0:ix1, iy0:iy1] = True
        return True

    def unfreeze_intersecting(R):
        """Remove frozen blocks that intersect R; return the stolen list."""
        nonlocal frozen_blocks, n_unfreezes
        R = as_rect(R)
        steal = [f for f in frozen_blocks if _rects_intersect(f, R)]
        if not steal:
            return []
        steal_set = set(steal)
        frozen_blocks = [f for f in frozen_blocks if f not in steal_set]
        n_unfreezes += len(steal)
        rebuild_claimed()
        return steal

    def unfreeze_bbox_expand(R0):
        """Unfreeze frozen blockers intersecting R, expanding bbox until stable.

        Soft blocks intersecting the final U are dissolved.  Returns U.
        """
        U = as_rect(R0)
        for _ in range(nx + ny + 5):
            steal = unfreeze_intersecting(U)
            if not steal:
                break
            for f in steal:
                U = (
                    min(U[0], f[0]),
                    max(U[1], f[1]),
                    min(U[2], f[2]),
                    max(U[3], f[3]),
                )
        dissolve_soft_intersecting([U])
        return U

    def commit_region(U, sx, sy):
        """Commit sum≥0 region U by bipartition into freezes ≤ HARD_MAX.

        Unfreezes any residual frozen still inside parts, dissolves soft,
        freezes every free non-neg part with area ≤ HARD_MAX.  Oversized
        indecomposable parts are kept as soft (post-pass will re-try).
        Returns True if seed (sx,sy) is no longer a free negative.
        """
        nonlocal soft_blocks, frozen_blocks
        U = as_rect(U)
        if float(content[U[0] : U[1], U[2] : U[3]].sum()) < 0.0:
            return False
        dissolve_soft_intersecting([U])
        if area(U) <= HARD_MAX and not claimed[U[0] : U[1], U[2] : U[3]].any():
            if freeze_rect(U):
                return True
        parts = _bipartition_nonneg(content, U, HARD_MAX)
        # Clear frozen leftovers that bipartition parts may still hit
        for p in parts:
            unfreeze_intersecting(p)
            dissolve_soft_intersecting([p])
        for p in parts:
            p = as_rect(p)
            if claimed[p[0] : p[1], p[2] : p[3]].any():
                continue
            sm = float(content[p[0] : p[1], p[2] : p[3]].sum())
            if sm < -1e-12:
                continue
            if area(p) <= HARD_MAX:
                freeze_rect(p)
            else:
                # rare indecomposable oversize: hold as soft for post-pass
                soft_blocks.append(p)
                claimed[p[0] : p[1], p[2] : p[3]] = True
        rebuild_claimed()
        return claimed[sx, sy] or content[sx, sy] >= 0.0

    def grow_freeze_capped(sx, sy):
        """Strip-grow dissolving soft; abort if area would exceed HARD_MAX."""
        ix0, ix1, iy0, iy1 = sx, sx + 1, sy, sy + 1
        dissolve_soft_intersecting([(ix0, ix1, iy0, iy1)])
        for _ in range(nx + ny + 5):
            sm = float(content[ix0:ix1, iy0:iy1].sum())
            if sm >= 0.0:
                break
            if (ix1 - ix0) * (iy1 - iy0) >= HARD_MAX:
                return None
            opts = []
            if ix0 > 0:
                srect = (ix0 - 1, ix0, iy0, iy1)
                if not strip_hits_frozen(srect):
                    new_a = (ix1 - (ix0 - 1)) * (iy1 - iy0)
                    if new_a <= HARD_MAX:
                        opts.append(
                            (float(content[ix0 - 1, iy0:iy1].sum()), "L", srect)
                        )
            if ix1 < nx:
                srect = (ix1, ix1 + 1, iy0, iy1)
                if not strip_hits_frozen(srect):
                    new_a = ((ix1 + 1) - ix0) * (iy1 - iy0)
                    if new_a <= HARD_MAX:
                        opts.append((float(content[ix1, iy0:iy1].sum()), "R", srect))
            if iy0 > 0:
                srect = (ix0, ix1, iy0 - 1, iy0)
                if not strip_hits_frozen(srect):
                    new_a = (ix1 - ix0) * (iy1 - (iy0 - 1))
                    if new_a <= HARD_MAX:
                        opts.append(
                            (float(content[ix0:ix1, iy0 - 1].sum()), "D", srect)
                        )
            if iy1 < ny:
                srect = (ix0, ix1, iy1, iy1 + 1)
                if not strip_hits_frozen(srect):
                    new_a = (ix1 - ix0) * ((iy1 + 1) - iy0)
                    if new_a <= HARD_MAX:
                        opts.append((float(content[ix0:ix1, iy1].sum()), "U", srect))
            if not opts:
                return None
            opts.sort(key=lambda o: (0 if sm + o[0] >= 0 else 1, -o[0]))
            _add, d, srect = opts[0]
            dissolve_soft_intersecting([srect])
            if d == "L":
                ix0 -= 1
            elif d == "R":
                ix1 += 1
            elif d == "D":
                iy0 -= 1
            else:
                iy1 += 1
        ix0, ix1, iy0, iy1 = _tighten_rect(content, ix0, ix1, iy0, iy1, sx, sy)
        R = as_rect((ix0, ix1, iy0, iy1))
        dissolve_soft_intersecting([R])
        if float(content[R[0] : R[1], R[2] : R[3]].sum()) < 0.0:
            return None
        if freeze_rect(R):
            return R
        return None

    def try_soft_freeze(sx, sy):
        if claimed[sx, sy] or content[sx, sy] >= 0.0:
            return None
        for mode in ("mild", "greedy"):
            ix0, ix1, iy0, iy1 = _grow_rectangle(
                content, claimed, sx, sy, mode=mode, respect_claimed=True
            )
            R = as_rect((ix0, ix1, iy0, iy1))
            if claimed[R[0] : R[1], R[2] : R[3]].any():
                continue
            if float(content[R[0] : R[1], R[2] : R[3]].sum()) < 0.0:
                continue
            if area(R) > HARD_MAX:
                continue
            if freeze_rect(R):
                return R
        return None

    def resoft():
        nonlocal soft_blocks, claimed
        rebuild_claimed()
        soft_blocks, claimed, _, n_left = _soft_strip_expand(
            content, claimed=claimed, merged_blocks=soft_blocks
        )
        # Bipartition any soft that drifted above HARD_MAX
        new_soft = []
        for b in soft_blocks:
            b = as_rect(b)
            if area(b) <= HARD_MAX:
                new_soft.append(b)
                continue
            parts = _bipartition_nonneg(content, b, HARD_MAX)
            for p in parts:
                p = as_rect(p)
                if float(content[p[0] : p[1], p[2] : p[3]].sum()) >= -1e-12:
                    new_soft.append(p)
        soft_blocks = new_soft
        rebuild_claimed()
        return int(np.sum((content < 0.0) & (~claimed)))

    def clear_one_seed(sx, sy):
        if claimed[sx, sy] or content[sx, sy] >= 0.0:
            return "already_clear", None
        # 1) capped exclusive grow
        g = grow_freeze_capped(sx, sy)
        if g is not None:
            return "freeze_grow", g
        # 2) expanding square + soft dissolve
        sq = _expanding_square_clear(P, sx, sy, nx, ny)
        if sq is not None:
            dissolve_soft_intersecting([sq])
            g0 = try_soft_freeze(sx, sy)
            if g0 is not None:
                return "soft_freeze", g0
            g = grow_freeze_capped(sx, sy)
            if g is not None:
                return "freeze_grow", g
            if freeze_rect(sq):
                return "freeze_sq", as_rect(sq)
            # frozen blockers inside sq → unfreeze + bipartition commit
            U = unfreeze_bbox_expand(sq)
            if commit_region(U, sx, sy):
                return "unfreeze_sq", U
            g = grow_freeze_capped(sx, sy)
            if g is not None:
                return "freeze_grow", g
        # 3) min-clear region with unfreeze cascade
        R = _min_clear_rect(content, P, sx, sy)
        U = unfreeze_bbox_expand(R)
        if commit_region(U, sx, sy):
            return "unfreeze_minclear", U
        g = grow_freeze_capped(sx, sy)
        if g is not None:
            return "freeze_grow", g
        # 4) ring expand + unfreeze
        for r in range(1, max(nx, ny) + 1):
            ring = (
                max(0, sx - r),
                min(nx, sx + r + 1),
                max(0, sy - r),
                min(ny, sy + r + 1),
            )
            U = unfreeze_bbox_expand(ring)
            if commit_region(U, sx, sy):
                return "ring_unfreeze", U
            g = grow_freeze_capped(sx, sy)
            if g is not None:
                return "ring_grow", g
            g0 = try_soft_freeze(sx, sy)
            if g0 is not None:
                return "ring_soft", g0
            if area(ring) > HARD_MAX * 8:
                break
        raise RuntimeError(
            "mega-free residual cannot clear seed (%d,%d) content=%.6g "
            "HARD_MAX=%d" % (sx, sy, float(content[sx, sy]), HARD_MAX)
        )

    # ---- Stage 2: clear residuals, prefer small seeds ----
    max_outer = max(60, n_res0 + 20)
    while force_iters < max_outer:
        free = free_neg_coords()
        if not free:
            break
        force_iters += 1
        n_before = len(free)
        n_diss_before = n_force_dissolves
        n_unf_before = n_unfreezes
        ranked = []
        for sx, sy in free:
            sq = _expanding_square_clear(P, sx, sy, nx, ny)
            a = area(sq) if sq is not None else nx * ny
            ranked.append((a, float(content[sx, sy]), sx, sy))
        ranked.sort(key=lambda t: (t[0], t[1]))

        n_cleared = 0
        max_frozen_this = 0
        for _a, _v, sx, sy in ranked:
            if claimed[sx, sy] or content[sx, sy] >= 0.0:
                continue
            _action, rect = clear_one_seed(sx, sy)
            n_cleared += 1
            if rect is not None:
                max_frozen_this = max(max_frozen_this, area(as_rect(rect)))

        n_left = resoft()
        max_now = max((area(r) for r in soft_blocks + frozen_blocks), default=0)
        print(
            "[grid_merge] residual pass %d: cleared=%d free %d→%d  "
            "soft=%d frozen=%d diss+=%d unfreeze+=%d max_cell_n=%d"
            % (
                force_iters,
                n_cleared,
                n_before,
                n_left,
                len(soft_blocks),
                len(frozen_blocks),
                n_force_dissolves - n_diss_before,
                n_unfreezes - n_unf_before,
                max_now,
            ),
            flush=True,
        )
        if n_left == 0:
            break
        if n_cleared == 0:
            raise RuntimeError("mega-free residual made no progress (%d free)" % n_left)

    n_left = len(free_neg_coords())
    if n_left:
        raise RuntimeError(
            "mega-free residual did not finish (%d free neg left)" % n_left
        )

    # Post-pass: bipartition any soft/frozen still over HARD_MAX
    n_split = 0

    def split_list(blocks, as_frozen):
        nonlocal n_split
        out = []
        for b in blocks:
            b = as_rect(b)
            if area(b) <= HARD_MAX:
                out.append(b)
                continue
            n_split += 1
            parts = _bipartition_nonneg(content, b, HARD_MAX)
            for p in parts:
                p = as_rect(p)
                if float(content[p[0] : p[1], p[2] : p[3]].sum()) < -1e-12:
                    continue
                if area(p) <= HARD_MAX:
                    out.append(p)
                else:
                    # last-resort strip then accept
                    sp = _strip_split_nonneg(content, p) or [p]
                    for q in sp:
                        q = as_rect(q)
                        if float(content[q[0] : q[1], q[2] : q[3]].sum()) >= -1e-12:
                            out.append(q)
        return out

    soft_blocks = split_list(soft_blocks, as_frozen=False)
    frozen_blocks = split_list(frozen_blocks, as_frozen=True)

    merged_blocks = soft_blocks + frozen_blocks
    claimed = _rebuild_claimed(merged_blocks, nx, ny)
    n_left = int(np.sum((content < 0.0) & (~claimed)))
    if n_left:
        # free negs created by aggressive split of a soft mega — re-clear once
        soft_blocks = [b for b in soft_blocks]
        frozen_blocks = list(frozen_blocks)
        rebuild_claimed()
        for _guard in range(n_left + 20):
            free = free_neg_coords()
            if not free:
                break
            free.sort(key=lambda p: float(content[p[0], p[1]]))
            sx, sy = free[0]
            clear_one_seed(sx, sy)
        resoft()
        merged_blocks = soft_blocks + frozen_blocks
        claimed = _rebuild_claimed(merged_blocks, nx, ny)
        n_left = int(np.sum((content < 0.0) & (~claimed)))
        if n_left:
            raise RuntimeError("after mega-free post-pass, %d free neg remain" % n_left)

    # Final hard ban: bipartition any remaining mega; refuse to emit one.
    max_all = max((area(as_rect(r)) for r in merged_blocks), default=0)
    if max_all > HARD_MAX:
        fixed = []
        for r in merged_blocks:
            r = as_rect(r)
            if area(r) <= HARD_MAX:
                fixed.append(r)
                continue
            n_split += 1
            parts = _bipartition_nonneg(content, r, HARD_MAX)
            for p in parts:
                p = as_rect(p)
                if float(content[p[0] : p[1], p[2] : p[3]].sum()) >= -1e-12:
                    fixed.append(p)
        merged_blocks = fixed
        claimed = _rebuild_claimed(merged_blocks, nx, ny)
        n_left = int(np.sum((content < 0.0) & (~claimed)))
        if n_left:
            raise RuntimeError("mega-free final split left %d free neg" % n_left)
        max_all = max((area(as_rect(r)) for r in merged_blocks), default=0)

    print(
        "[grid_merge] residual local done: passes=%d  soft_dissolves=%d  "
        "unfreezes=%d  total_merges=%d  max_cell_n=%d  HARD_MAX=%d  n_split=%d"
        % (
            force_iters,
            n_force_dissolves,
            n_unfreezes,
            len(merged_blocks),
            max_all,
            HARD_MAX,
            n_split,
        ),
        flush=True,
    )
    if max_all > HARD_MAX:
        print(
            "[grid_merge] WARN: max_cell_n=%d still exceeds HARD_MAX=%d"
            % (max_all, HARD_MAX),
            flush=True,
        )
    return merged_blocks, n_neg_before, force_iters, n_force_dissolves


def _strip_split_nonneg(content, rect):
    """Partition rect into horizontal strips each with sum >= 0.

    Requires total sum(rect) >= 0.  Greedy: extend each strip until
    non-negative, then start the next.  Falls back to vertical strips if
    a horizontal pass is impossible (shouldn't happen when total sum >= 0).
    """
    content = np.asarray(content, dtype=float)
    ix0, ix1, iy0, iy1 = rect
    if float(content[ix0:ix1, iy0:iy1].sum()) < 0.0:
        return [rect]

    def h_split():
        parts = []
        j = iy0
        while j < iy1:
            j2 = j + 1
            while j2 <= iy1 and float(content[ix0:ix1, j:j2].sum()) < 0.0:
                j2 += 1
            if j2 > iy1:
                # cannot finish last strip alone — fold into previous only if
                # the combined sum stays non-negative
                if parts:
                    a, b, c, _d = parts[-1]
                    cand = (a, b, c, iy1)
                    if float(content[a:b, c:iy1].sum()) >= 0.0:
                        parts[-1] = cand
                        return parts
                return None
            parts.append((ix0, ix1, j, j2))
            j = j2
        return parts

    def v_split():
        parts = []
        i = ix0
        while i < ix1:
            i2 = i + 1
            while i2 <= ix1 and float(content[i:i2, iy0:iy1].sum()) < 0.0:
                i2 += 1
            if i2 > ix1:
                if parts:
                    a, _b, c, d = parts[-1]
                    cand = (a, ix1, c, d)
                    if float(content[a:ix1, c:d].sum()) >= 0.0:
                        parts[-1] = cand
                        return parts
                return None
            parts.append((i, i2, iy0, iy1))
            i = i2
        return parts

    hp = h_split()
    vp = v_split()
    # prefer the finer partition (more parts → smaller cells)
    cands = [p for p in (hp, vp) if p]
    if not cands:
        return [rect]
    return max(cands, key=len)


def _repartition_large_rect(content, rect, max_keep=500):
    """Split an oversized non-negative merge into parts each ≤ max_keep.

    Prefer recursive non-neg bipartition; fall back to strip-split, then soft
    repartition.  Every returned part has content sum ≥ 0 (up to float noise)
    when possible.
    """
    content = np.asarray(content, dtype=float)
    ix0, ix1, iy0, iy1 = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    rect = (ix0, ix1, iy0, iy1)
    area = (ix1 - ix0) * (iy1 - iy0)
    if area <= max_keep:
        return [rect]
    if float(content[ix0:ix1, iy0:iy1].sum()) < -1e-12:
        return [rect]

    parts = _bipartition_nonneg(content, rect, max_keep)
    max_p = max((p[1] - p[0]) * (p[3] - p[2]) for p in parts)
    if max_p <= max_keep:
        return parts

    # Bipartition left indecomposable oversize → soft repartition attempt
    final = []
    for p in parts:
        pa = (p[1] - p[0]) * (p[3] - p[2])
        sm = float(content[p[0] : p[1], p[2] : p[3]].sum())
        if sm < -1e-12:
            continue
        if pa <= max_keep:
            final.append(p)
            continue
        try:
            sub = _soft_repartition_sub(content, p, max_keep=max_keep)
            for q in sub:
                qa = (q[1] - q[0]) * (q[3] - q[2])
                if float(content[q[0] : q[1], q[2] : q[3]].sum()) >= -1e-12:
                    if qa <= max_keep:
                        final.append(q)
                    else:
                        final.extend(_bipartition_nonneg(content, q, max_keep))
        except Exception as exc:
            print(
                "[grid_merge] WARN: soft_repartition failed (%s); "
                "keeping large cell n=%d" % (exc, pa),
                flush=True,
            )
            final.append(p)
    return final if final else [rect]


def _soft_repartition_sub(content, rect, max_keep=500):
    """Soft-expand a large non-neg rect; freeze residual free negs locally."""
    content = np.asarray(content, dtype=float)
    ix0, ix1, iy0, iy1 = rect
    sub = np.ascontiguousarray(content[ix0:ix1, iy0:iy1], dtype=float)
    snx, sny = sub.shape
    soft, claimed, _nb, _nres = _soft_strip_expand(sub)
    frozen = []

    def rebuild():
        nonlocal claimed
        claimed = _rebuild_claimed(soft + frozen, snx, sny)

    def free_neg():
        return list(zip(*np.where((sub < 0.0) & (~claimed))))

    guard = 0
    while free_neg() and guard < snx * sny + 50:
        guard += 1
        seeds = free_neg()
        seeds.sort(key=lambda p: float(sub[p[0], p[1]]))
        sx, sy = seeds[0]
        # exclusive grow from seed (soft+frozen claimed)
        rebuild()
        g = _grow_rectangle(sub, claimed, sx, sy, mode="mild", respect_claimed=True)
        g = tuple(g)
        if float(sub[g[0] : g[1], g[2] : g[3]].sum()) < 0.0:
            g = _grow_rectangle(
                sub, claimed, sx, sy, mode="greedy", respect_claimed=True
            )
            g = tuple(g)
        if float(sub[g[0] : g[1], g[2] : g[3]].sum()) < 0.0:
            # dissolve soft in expanding square and retry grow
            P = _prefix_2d(sub)
            sq = _expanding_square_clear(P, sx, sy, snx, sny)
            if sq is None:
                raise RuntimeError("soft_repartition cannot clear (%d,%d)" % (sx, sy))
            soft = [b for b in soft if not _rects_intersect(b, sq)]
            rebuild()
            g = _grow_rectangle(
                sub, claimed, sx, sy, mode="greedy", respect_claimed=True
            )
            g = tuple(g)
            if float(sub[g[0] : g[1], g[2] : g[3]].sum()) < 0.0:
                # place expanding square after full soft dissolve in it
                soft = [b for b in soft if not _rects_intersect(b, sq)]
                rebuild()
                if claimed[sq[0] : sq[1], sq[2] : sq[3]].any():
                    # blocked by frozen only — leave as last-resort sq freeze
                    # by not absorbing frozen: skip this seed's neighbors
                    # grow only into unclaimed
                    pass
                if float(sub[sq[0] : sq[1], sq[2] : sq[3]].sum()) >= 0.0:
                    if not claimed[sq[0] : sq[1], sq[2] : sq[3]].any():
                        g = sq
                    else:
                        raise RuntimeError(
                            "soft_repartition stuck at (%d,%d)" % (sx, sy)
                        )
                else:
                    raise RuntimeError("soft_repartition stuck at (%d,%d)" % (sx, sy))
        # dissolve soft overlapping g, freeze g (must not hit frozen)
        if any(_rects_intersect(f, g) for f in frozen):
            # shrink not possible easily — skip freeze absorb; try next seed
            # by freezing seed+neighbour free bins only
            soft = [b for b in soft if not _rects_intersect(b, g)]
            rebuild()
            if claimed[sx, sy]:
                continue
            raise RuntimeError("soft_repartition frozen block at (%d,%d)" % (sx, sy))
        soft = [b for b in soft if not _rects_intersect(b, g)]
        rebuild()
        if claimed[g[0] : g[1], g[2] : g[3]].any():
            soft = [b for b in soft if not _rects_intersect(b, g)]
            rebuild()
        if claimed[g[0] : g[1], g[2] : g[3]].any():
            raise RuntimeError("soft_repartition claim conflict at (%d,%d)" % (sx, sy))
        frozen.append(g)
        rebuild()

    parts = soft + frozen
    claimed = _rebuild_claimed(parts, snx, sny)
    for i in range(snx):
        for j in range(sny):
            if not claimed[i, j]:
                if sub[i, j] < 0.0:
                    raise RuntimeError("soft_repartition free neg (%d,%d)" % (i, j))
                parts.append((i, i + 1, j, j + 1))
    out = [(ix0 + a, ix0 + b, iy0 + c, iy0 + d) for a, b, c, d in parts]
    # recurse large parts
    parent_a = (ix1 - ix0) * (iy1 - iy0)
    final = []
    for p in out:
        pa = (p[1] - p[0]) * (p[3] - p[2])
        if (
            pa > max_keep
            and pa < parent_a
            and float(content[p[0] : p[1], p[2] : p[3]].sum()) >= 0.0
        ):
            final.extend(_repartition_large_rect(content, p, max_keep=max_keep))
        else:
            final.append(p)
    return final


def _build_grid_force_clear_legacy(content, error, x_edges, y_edges):
    """LEGACY strip-expand + bbox force-clear (mega-cell cascade). Do not use."""
    content = np.asarray(content, dtype=float).copy()
    error = np.asarray(error, dtype=float).copy()
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)
    nx, ny = content.shape
    merged_blocks, claimed, n_neg_before, _n_res = _soft_strip_expand(content)
    n_force_dissolves = 0

    def free_neg_coords():
        return list(zip(*np.where((content < 0.0) & (~claimed))))

    max_force = max(n_neg_before * 3, 100)
    force_iters = 0
    while force_iters < max_force:
        free = free_neg_coords()
        if not free:
            break
        force_iters += 1
        free.sort(key=lambda p: content[p[0], p[1]])
        ix, iy = free[0]
        dummy = np.zeros((nx, ny), dtype=bool)
        ix0, ix1, iy0, iy1 = _grow_rectangle(
            content, dummy, ix, iy, mode="greedy", respect_claimed=False
        )
        rect = (ix0, ix1, iy0, iy1)
        changed = True
        while changed:
            changed = False
            for b in merged_blocks:
                if _rects_intersect(b, rect):
                    u = (
                        min(rect[0], b[0]),
                        max(rect[1], b[1]),
                        min(rect[2], b[2]),
                        max(rect[3], b[3]),
                    )
                    if u != rect:
                        rect = u
                        changed = True
        ix0, ix1, iy0, iy1 = rect
        for _ in range(nx + ny + 5):
            sm = float(content[ix0:ix1, iy0:iy1].sum())
            if sm >= 0.0:
                break
            opts = []
            if ix0 > 0:
                opts.append((float(content[ix0 - 1, iy0:iy1].sum()), "L"))
            if ix1 < nx:
                opts.append((float(content[ix1, iy0:iy1].sum()), "R"))
            if iy0 > 0:
                opts.append((float(content[ix0:ix1, iy0 - 1].sum()), "D"))
            if iy1 < ny:
                opts.append((float(content[ix0:ix1, iy1].sum()), "U"))
            if not opts:
                break
            opts.sort(key=lambda o: -o[0])
            d = opts[0][1]
            if d == "L":
                ix0 -= 1
            elif d == "R":
                ix1 += 1
            elif d == "D":
                iy0 -= 1
            else:
                iy1 += 1
            rect = (ix0, ix1, iy0, iy1)
            for b in merged_blocks:
                if _rects_intersect(b, rect):
                    rect = (
                        min(rect[0], b[0]),
                        max(rect[1], b[1]),
                        min(rect[2], b[2]),
                        max(rect[3], b[3]),
                    )
                    ix0, ix1, iy0, iy1 = rect
        sm = float(content[ix0:ix1, iy0:iy1].sum())
        if sm < 0.0:
            if float(content.sum()) >= 0.0:
                ix0, ix1, iy0, iy1 = 0, nx, 0, ny
                rect = (ix0, ix1, iy0, iy1)
            else:
                raise RuntimeError(
                    "cannot clear negatives: total fine integral is negative "
                    "(%.6g)" % float(content.sum())
                )
        kept = []
        dissolved = 0
        for b in merged_blocks:
            if _rects_intersect(b, rect):
                dissolved += 1
            else:
                kept.append(b)
        if dissolved:
            n_force_dissolves += dissolved
        merged_blocks = kept
        merged_blocks.append(rect)
        claimed = _rebuild_claimed(merged_blocks, nx, ny)

    if free_neg_coords():
        raise RuntimeError(
            "force-clear did not finish (%d free neg left)" % len(free_neg_coords())
        )
    return merged_blocks, n_neg_before, force_iters, n_force_dissolves


# Back-compat alias
def _build_grid_force_clear(content, error, x_edges, y_edges):
    """Deprecated alias → residual_local (no bbox cascade)."""
    return _build_grid_residual_local(content, error, x_edges, y_edges)


def build_grid_from_arrays(
    content,
    error,
    x_edges,
    y_edges,
    *,
    meta=None,
    algorithm="residual_local",
):
    """Build a rectangular partition of the fine TH2.

    algorithm:
      'residual_local' / 'force_clear' (default) — soft strip-expand + local
        residual dissolve (no bbox cascade; no residual negatives).
      'soft' — soft strip-expand only; residual free negatives remain as
        singleton cells (for inspection / residual R&D).
      'force_clear_legacy' — old bbox-union second stage (mega-cells; A/B only).
      'independent' — alias of residual_local (legacy name).
    """
    content = np.asarray(content, dtype=float).copy()
    error = np.asarray(error, dtype=float).copy()
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)
    nx, ny = content.shape
    assert x_edges.shape == (nx + 1,)
    assert y_edges.shape == (ny + 1,)

    n_neg_before = int(np.sum(content < 0))
    if n_neg_before > 0 and float(content.sum()) < 0.0:
        raise RuntimeError(
            "cannot clear negatives: total fine integral is negative (%.6g)"
            % float(content.sum())
        )

    n_cand_total = 0
    force_iters = 0
    n_force_dissolves = 0
    allow_residual_neg = False

    # Normalise aliases
    if algorithm in ("force_clear", "independent", None):
        algorithm = "residual_local"

    if algorithm == "soft":
        merged_blocks, claimed, n_neg_before, n_res = _soft_strip_expand(content)
        algo_name = "soft_strip_expand"
        allow_residual_neg = True
        print(
            "[grid_merge] soft phase only: merge_blocks=%d  residual_free_neg=%d"
            % (len(merged_blocks), n_res),
            flush=True,
        )
    elif algorithm == "force_clear_legacy":
        merged_blocks, n_neg_before, force_iters, n_force_dissolves = (
            _build_grid_force_clear_legacy(content, error, x_edges, y_edges)
        )
        algo_name = "strip_expand_force_clear_legacy_bbox"
        claimed = _rebuild_claimed(merged_blocks, nx, ny)
    elif algorithm == "residual_local":
        merged_blocks, n_neg_before, force_iters, n_force_dissolves = (
            _build_grid_residual_local(content, error, x_edges, y_edges)
        )
        algo_name = "soft_residual_local_dissolve"
        claimed = _rebuild_claimed(merged_blocks, nx, ny)
    else:
        raise ValueError(
            "unknown algorithm %r (use residual_local|soft|force_clear_legacy)"
            % algorithm
        )

    cells = []
    cid = 0
    for ix0, ix1, iy0, iy1 in merged_blocks:
        csum = float(content[ix0:ix1, iy0:iy1].sum())
        if csum < 0.0 and not allow_residual_neg:
            raise RuntimeError(
                "selected merge has negative sum %.6g at [%d:%d]x[%d:%d]"
                % (csum, ix0, ix1, iy0, iy1)
            )
        e2 = _bbox_err2(error, ix0, ix1, iy0, iy1)
        n_fine = (ix1 - ix0) * (iy1 - iy0)
        cells.append(
            Cell(
                id=cid,
                ix0=ix0,
                ix1=ix1,
                iy0=iy0,
                iy1=iy1,
                xmin=float(x_edges[ix0]),
                xmax=float(x_edges[ix1]),
                ymin=float(y_edges[iy0]),
                ymax=float(y_edges[iy1]),
                content=csum,
                error=math.sqrt(e2) if e2 > 0 else 0.0,
                merged=(n_fine > 1),
            )
        )
        cid += 1

    for ix in range(nx):
        for iy in range(ny):
            if claimed[ix, iy]:
                continue
            c = float(content[ix, iy])
            if c < 0.0 and not allow_residual_neg:
                raise RuntimeError(
                    "singleton fine bin (%d,%d) still negative (%.6g)" % (ix, iy, c)
                )
            e = float(error[ix, iy])
            cells.append(
                Cell(
                    id=cid,
                    ix0=ix,
                    ix1=ix + 1,
                    iy0=iy,
                    iy1=iy + 1,
                    xmin=float(x_edges[ix]),
                    xmax=float(x_edges[ix + 1]),
                    ymin=float(y_edges[iy]),
                    ymax=float(y_edges[iy + 1]),
                    content=c,
                    error=e if e > 0 else 0.0,
                    merged=False,
                )
            )
            cid += 1

    n_neg_after = sum(1 for c in cells if c.content < 0)
    if n_neg_after > 0 and not allow_residual_neg:
        raise RuntimeError(
            "no-negative grid failed: %d cells still have content < 0" % n_neg_after
        )

    integral = float(sum(c.content for c in cells))
    n_fine_cov = sum(c.n_fine for c in cells)
    n_merged = sum(1 for c in cells if c.merged)
    max_merge_area = max((c.n_fine for c in cells if c.merged), default=0)
    max_merge_yield = max((c.content for c in cells if c.merged), default=0.0)
    out_meta = {
        "algorithm": algo_name,
        "nx_fine": int(nx),
        "ny_fine": int(ny),
        "n_neg_before": int(n_neg_before),
        "n_neg_after": int(n_neg_after),
        "n_merge_blocks": int(len(merged_blocks)),
        "n_merged_cells": int(n_merged),
        "max_merge_n_fine": int(max_merge_area),
        "max_merge_yield": float(max_merge_yield),
        "n_candidate_rects_total": int(n_cand_total),
        "n_force_iters": int(force_iters),
        "n_force_dissolves": int(n_force_dissolves),
        "integral": integral,
        "integral_fine": float(content.sum()),
        "n_fine_covered": int(n_fine_cov),
    }
    if meta:
        out_meta.update(meta)
    if abs(integral - float(content.sum())) > 1e-4 * max(
        1.0, abs(float(content.sum()))
    ):
        raise RuntimeError(
            "integral mismatch cells=%.6g fine=%.6g (overlap/miss)"
            % (integral, float(content.sum()))
        )
    if n_fine_cov != nx * ny:
        raise RuntimeError("fine coverage %d != %d" % (n_fine_cov, nx * ny))
    grid = Grid(
        cells,
        x_edges,
        y_edges,
        meta=out_meta,
        fine_content=content,
        fine_error=error,
    )
    vstats = validate_grid(grid, fine_content=content)
    grid.meta["validation"] = vstats
    return grid


def build_grid_from_th2(h, algorithm="residual_local", **meta_kw):
    content, error, x_edges, y_edges = th2_to_arrays(h)
    g = build_grid_from_arrays(content, error, x_edges, y_edges, algorithm=algorithm)
    g.meta.update(meta_kw)
    return g


def build_grid_from_file(
    path, hist_path="plots_2d/DY_2d", algorithm="residual_local", **meta_kw
):
    h = load_th2(path, hist_path)
    meta_kw.setdefault("source_file", os.path.abspath(path))
    meta_kw.setdefault("hist", hist_path)
    return build_grid_from_th2(h, algorithm=algorithm, **meta_kw)


# --------------------------------------------------------------------------- plot


def plot_grid(grid, output_pdf, title=None):
    """Multi-page matplotlib PDF of the adaptive grid.

    Every cell is an axis-aligned rectangle of fine bins (validated).  Large
    empty-ish regions that look "non-rectangular" when many adjacent boxes
    share edges are still unions of true rectangles — page 2/3 draw each
    merged cell as one Rectangle patch to make that explicit.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except Exception as exc:
        print(
            "[grid_merge] matplotlib unavailable (%s); ROOT plot" % exc,
            flush=True,
        )
        return _plot_grid_root(grid, output_pdf, title=title)

    # Re-validate before drawing (hard error if partition is broken)
    validate_grid(grid)

    nx = int(grid.meta["nx_fine"])
    ny = int(grid.meta["ny_fine"])
    xe = grid.x_edges
    ye = grid.y_edges
    if grid.fine_content is not None:
        fine = np.asarray(grid.fine_content, dtype=float)
    else:
        # Reconstruct average density (fallback)
        fine = np.zeros((nx, ny))
        for c in grid.cells:
            fine[c.ix0 : c.ix1, c.iy0 : c.iy1] = c.content / max(c.n_fine, 1)

    Z = fine.T  # pcolormesh: rows=y, cols=x
    merged = [c for c in grid.cells if c.merged]
    residual_neg = [c for c in grid.cells if c.content < 0]
    n_merged = len(merged)
    n_sing = sum(1 for c in grid.cells if not c.merged)
    ttl = title or "grid merge"

    # Ownership map: 0=singleton, 1=merged, 2=residual negative
    own = np.zeros((nx, ny), dtype=np.int32)
    for c in grid.cells:
        if c.content < 0:
            own[c.ix0 : c.ix1, c.iy0 : c.iy1] = 2
        elif c.merged:
            own[c.ix0 : c.ix1, c.iy0 : c.iy1] = 1
        else:
            own[c.ix0 : c.ix1, c.iy0 : c.iy1] = 0

    parent = os.path.dirname(os.path.abspath(output_pdf))
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Symmetric scale for signed content
    abs_vals = np.abs(fine[np.isfinite(fine)])
    vmax = float(np.percentile(abs_vals, 99)) if abs_vals.size else 1.0
    vmax = max(vmax, 1e-6)

    with PdfPages(output_pdf) as pdf:
        # ---- Page 1: original fine TH2 content (no outlines) ----
        fig, ax = plt.subplots(figsize=(11, 8))
        pcm = ax.pcolormesh(
            xe, ye, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat"
        )
        fig.colorbar(pcm, ax=ax, label="fine bin content")
        ax.set_xlabel("x (DNN)")
        ax.set_ylabel("y (HME)")
        ax.set_title(
            "%s — original fine TH2\n"
            "nx×ny=%d×%d  n_neg_before=%s  integral=%.4g"
            % (
                ttl,
                nx,
                ny,
                grid.meta.get("n_neg_before"),
                grid.meta.get("integral_fine", fine.sum()),
            )
        )
        ax.set_aspect("auto")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2: ownership (singleton / merged / residual neg) ----
        fig, ax = plt.subplots(figsize=(11, 8))
        cmap = ListedColormap(["#dddddd", "#4C78A8", "#E45756"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = BoundaryNorm(bounds, cmap.N)
        pcm = ax.pcolormesh(xe, ye, own.T, cmap=cmap, norm=norm, shading="flat")
        cbar = fig.colorbar(pcm, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["singleton (≥0)", "merged block", "residual neg"])
        # Draw each merged cell as a true Rectangle outline (no fill)
        patches = []
        for c in merged:
            patches.append(
                Rectangle(
                    (c.xmin, c.ymin),
                    c.xmax - c.xmin,
                    c.ymax - c.ymin,
                )
            )
        if patches:
            pc = PatchCollection(
                patches,
                facecolor="none",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
            )
            ax.add_collection(pc)
        # Per-cell yield text: size from cell box AND string length so the
        # label never spills outside the rectangle; clip_path as a hard guard.
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
        ax.set_xlabel("x (DNN)")
        ax.set_ylabel("y (HME)")
        ax.set_title(
            "%s — partition ownership + yield labels\n"
            "cells=%d  merged=%d (black outlines)  "
            "singleton=%d  residual_neg=%d  "
            "(yield text fits inside each cell; zoom PDF to read fine bins)"
            % (ttl, grid.n_cells, n_merged, n_sing, len(residual_neg))
        )
        fig.tight_layout()
        # Real axes size in points (after layout) for accurate font scaling
        fig.canvas.draw()
        ax_bbox = ax.get_window_extent()
        dpi = float(fig.dpi) or 100.0
        ax_w_pt = float(ax_bbox.width) * 72.0 / dpi
        ax_h_pt = float(ax_bbox.height) * 72.0 / dpi
        xspan = float(xe[-1] - xe[0]) or 1.0
        yspan = float(ye[-1] - ye[0]) or 1.0

        def _fmt_yield(v):
            av = abs(float(v))
            if av == 0.0:
                return "0"
            if av >= 100.0:
                return "%.0f" % v
            if av >= 10.0:
                return "%.1f" % v
            if av >= 1.0:
                return "%.2f" % v
            if av >= 0.1:
                return "%.2f" % v
            if av >= 0.01:
                return "%.3f" % v
            # compact sci notation (fewer chars → less overflow)
            return "%.0e" % v

        # char-width / line-height factors for proportional sans (points)
        _CHAR_W = 0.60  # average glyph width ≈ 0.60 × fontsize
        _CHAR_H = 1.05  # line height ≈ 1.05 × fontsize
        _PAD = 0.80  # stay inside ~80% of the cell box

        for c in grid.cells:
            dx = float(c.xmax - c.xmin)
            dy = float(c.ymax - c.ymin)
            if dx <= 0.0 or dy <= 0.0:
                continue
            label = _fmt_yield(c.content)
            nch = max(len(label), 1)
            w_pt = (dx / xspan) * ax_w_pt
            h_pt = (dy / yspan) * ax_h_pt
            # fit full string width AND height into the padded cell
            fs = min(
                (_PAD * w_pt) / (_CHAR_W * nch),
                (_PAD * h_pt) / _CHAR_H,
                7.0,
            )
            # still emit sub-pt labels so fine bins are readable when zoomed
            fs = max(0.25, fs)
            cx = 0.5 * (c.xmin + c.xmax)
            cy = 0.5 * (c.ymin + c.ymax)
            if c.content < 0.0:
                color = "#5c0000"
            elif c.merged:
                color = "#061a33"
            else:
                color = "#222222"
            # hard clip to this cell's data rectangle
            clip_patch = Rectangle(
                (c.xmin, c.ymin),
                dx,
                dy,
                transform=ax.transData,
            )
            t = ax.text(
                cx,
                cy,
                label,
                ha="center",
                va="center",
                fontsize=fs,
                color=color,
                clip_on=True,
                fontfamily="sans-serif",
                zorder=5,
            )
            t.set_clip_path(clip_patch)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 3: merged rectangles only (empty canvas + boxes) ----
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.set_facecolor("#f7f7f7")
        # light fine grid for context
        ax.pcolormesh(
            xe,
            ye,
            np.zeros((ny, nx)),
            cmap="Greys",
            vmin=0,
            vmax=1,
            shading="flat",
            alpha=0.15,
        )
        if merged:
            # colour by log10(n_fine)
            sizes = np.array([c.n_fine for c in merged], dtype=float)
            logn = np.log10(np.maximum(sizes, 1.0))
            patches = []
            for c in merged:
                patches.append(
                    Rectangle(
                        (c.xmin, c.ymin),
                        c.xmax - c.xmin,
                        c.ymax - c.ymin,
                    )
                )
            pc = PatchCollection(
                patches,
                cmap="viridis",
                array=logn,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.55,
            )
            ax.add_collection(pc)
            cbar = fig.colorbar(pc, ax=ax)
            cbar.set_label("log10(n_fine bins in block)")
        # residual negatives as red empty rectangles
        for c in residual_neg:
            ax.add_patch(
                Rectangle(
                    (c.xmin, c.ymin),
                    c.xmax - c.xmin,
                    c.ymax - c.ymin,
                    fill=False,
                    edgecolor="red",
                    linewidth=0.8,
                )
            )
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
        ax.set_xlabel("x (DNN)")
        ax.set_ylabel("y (HME)")
        ax.set_title(
            "%s — merged blocks only (each patch is one axis-aligned rectangle)\n"
            "n_merged=%d  residual_neg singletons=%d (red)  "
            "validation: no overlap, full coverage" % (ttl, n_merged, len(residual_neg))
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 4: residual negatives on original content ----
        fig, ax = plt.subplots(figsize=(11, 8))
        pcm = ax.pcolormesh(
            xe, ye, Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat"
        )
        fig.colorbar(pcm, ax=ax, label="fine bin content")
        for c in residual_neg:
            ax.add_patch(
                Rectangle(
                    (c.xmin, c.ymin),
                    c.xmax - c.xmin,
                    c.ymax - c.ymin,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.0,
                )
            )
        ax.set_xlabel("x (DNN)")
        ax.set_ylabel("y (HME)")
        ax.set_title(
            "%s — residual negative singletons (lime) on fine content\n"
            "n_neg_after=%d  (could not form sum≥0 rectangle without dissolving prior merges)"
            % (ttl, len(residual_neg))
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 5: size + content histograms ----
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        if merged:
            axes[0].hist(
                [c.n_fine for c in merged],
                bins=40,
                histtype="stepfilled",
                color="C0",
                alpha=0.7,
            )
        axes[0].set_xlabel("n_fine per merged block")
        axes[0].set_ylabel("count")
        axes[0].set_yscale("log")
        axes[0].set_title("merged block sizes")
        vals = np.array([c.content for c in grid.cells])
        axes[1].hist(vals, bins=80, histtype="step", color="C0", label="all")
        if merged:
            axes[1].hist(
                [c.content for c in merged],
                bins=40,
                histtype="step",
                color="C2",
                label="merged",
            )
        axes[1].axvline(0, color="k", ls="--", lw=0.8)
        axes[1].set_xlabel("cell content")
        axes[1].set_ylabel("count")
        axes[1].set_yscale("log")
        axes[1].legend()
        axes[1].set_title("cell content")
        fig.suptitle(ttl)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(
        "[grid_merge] plot: %d pages, %d rectangular merged patches, "
        "validation OK (cover=1 everywhere)" % (5, n_merged),
        flush=True,
    )
    return output_pdf


def _plot_grid_root(grid, output_pdf, title=None):
    """ROOT fallback if matplotlib is unavailable."""
    validate_grid(grid)
    nx = grid.meta["nx_fine"]
    ny = grid.meta["ny_fine"]
    xe = grid.x_edges
    ye = grid.y_edges
    fine = grid.fine_content
    h = ROOT.TH2D(
        "h_grid",
        (title or "grid") + ";x;y",
        nx,
        np.array(xe, dtype=np.float64),
        ny,
        np.array(ye, dtype=np.float64),
    )
    h.SetDirectory(0)
    if fine is not None:
        for ix in range(nx):
            for iy in range(ny):
                h.SetBinContent(ix + 1, iy + 1, float(fine[ix, iy]))
    else:
        for c in grid.cells:
            for ix in range(c.ix0, c.ix1):
                for iy in range(c.iy0, c.iy1):
                    h.SetBinContent(ix + 1, iy + 1, c.content / max(c.n_fine, 1))
    cnv = ROOT.TCanvas("c_grid", "grid", 1200, 900)
    h.Draw("COLZ")
    boxes = []
    for cell in grid.cells:
        if not cell.merged:
            continue
        b = ROOT.TBox(cell.xmin, cell.ymin, cell.xmax, cell.ymax)
        b.SetFillStyle(0)
        b.SetLineColor(ROOT.kBlack)
        b.SetLineWidth(1)
        b.Draw("same")
        boxes.append(b)
    cnv.SaveAs(output_pdf)
    return output_pdf


# --------------------------------------------------------------------------- JSON


def write_grid_json(grid, path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    d = grid.to_dict()
    # recompute n_merged/n_singleton into meta for to_dict already
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)
    return path


# --------------------------------------------------------------------------- CLI


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Input ROOT file")
    p.add_argument("--hist", default="plots_2d/DY_2d", help="TH2 path")
    p.add_argument("--output-json", required=True, help="Output grid JSON")
    p.add_argument(
        "--output-pdf",
        default=None,
        help="Output inspection PDF (default: alongside JSON)",
    )
    p.add_argument("--mx", type=int, default=None, help="Mass label for metadata")
    p.add_argument("--category", default="res2b")
    p.add_argument("--process", default="DY")
    p.add_argument("--title", default=None)
    p.add_argument(
        "--algorithm",
        choices=(
            "residual_local",
            "force_clear",
            "soft",
            "force_clear_legacy",
            "independent",
        ),
        default="residual_local",
        help=(
            "Merge algorithm (default residual_local = soft + local residual "
            "dissolve, no bbox cascade). soft = stage1 only. "
            "force_clear_legacy = old bbox cascade (mega-cells)."
        ),
    )
    p.add_argument(
        "--allow-residual-neg",
        action="store_true",
        help="Exit 0 even if residual negatives remain (implied by --algorithm soft).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.input):
        sys.exit("[ERROR] input not found: %s" % args.input)

    print(
        "[grid_merge] loading %s:%s  algorithm=%s"
        % (args.input, args.hist, args.algorithm),
        flush=True,
    )
    grid = build_grid_from_file(
        args.input,
        args.hist,
        mx=args.mx,
        category=args.category,
        process=args.process,
        algorithm=args.algorithm,
    )
    grid.meta["n_merged"] = sum(1 for c in grid.cells if c.merged)
    grid.meta["n_singleton"] = sum(1 for c in grid.cells if not c.merged)
    try:
        vstats = validate_grid(grid)
    except GridValidationError as exc:
        print("[grid_merge] VALIDATION FAILED:\n%s" % exc, flush=True)
        sys.exit(1)
    print(
        "[grid_merge] algorithm=%s  fine=%dx%d  n_neg_before=%d  n_neg_after=%d  "
        "cells=%d  merged=%d  singletons=%d  integral=%.6g  "
        "max_merge_n_fine=%s  max_merge_yield=%.4g  "
        "n_force_iters=%s  n_force_dissolves=%s  "
        "validation=OK (cover=%d..%d)"
        % (
            grid.meta.get("algorithm"),
            grid.meta["nx_fine"],
            grid.meta["ny_fine"],
            grid.meta["n_neg_before"],
            grid.meta["n_neg_after"],
            grid.n_cells,
            grid.meta["n_merged"],
            grid.meta["n_singleton"],
            grid.meta["integral"],
            grid.meta.get("max_merge_n_fine"),
            float(grid.meta.get("max_merge_yield") or 0),
            grid.meta.get("n_force_iters"),
            grid.meta.get("n_force_dissolves"),
            vstats["cover_min"],
            vstats["cover_max"],
        ),
        flush=True,
    )

    write_grid_json(grid, args.output_json)
    print("[grid_merge] wrote %s" % args.output_json, flush=True)

    pdf = args.output_pdf
    if pdf is None:
        base, _ = os.path.splitext(args.output_json)
        pdf = base + ".pdf"
    ttl = args.title or (
        "mX=%s %s [%s]"
        % (
            args.mx,
            args.process,
            grid.meta.get("algorithm", args.algorithm),
        )
        if args.mx is not None
        else "%s [%s]" % (args.hist, grid.meta.get("algorithm", args.algorithm))
    )
    plot_grid(grid, pdf, title=ttl)
    print("[grid_merge] wrote %s" % pdf, flush=True)

    allow_res = args.allow_residual_neg or args.algorithm == "soft"
    if grid.meta["n_neg_after"] > 0:
        msg = "[grid_merge] residual negative cells remain: %d  algorithm=%s" % (
            grid.meta["n_neg_after"],
            grid.meta.get("algorithm"),
        )
        if allow_res:
            print(msg + "  (allowed)", flush=True)
            sys.exit(0)
        print("[grid_merge] ERROR: " + msg, flush=True)
        sys.exit(2)
    print(
        "[grid_merge] no residual negatives  algorithm=%s  n_candidates=%s"
        % (
            grid.meta.get("algorithm"),
            grid.meta.get("n_candidate_rects_total"),
        ),
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
