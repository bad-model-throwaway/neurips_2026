"""Compile multi-panel figures from individual SVG panels."""
from configs import *
from visualization.svgtools import *

import os
import shutil

# 6.4 x 4.8 inches at 72 dpi
PANEL_WIDTH = 460.8
PANEL_HEIGHT = 345.6


def _combine_row(panels, output):
    """Combine a list of panel files horizontally into output."""
    if len(panels) == 1:
        shutil.copy(panels[0], output)
        return

    # Combine first two
    tmp = output.replace('.svg', '_tmp.svg')
    combine_svgs_horizontal(panels[0], panels[1], tmp if len(panels) > 2 else output)

    # Add remaining panels
    for i, panel in enumerate(panels[2:], start=2):
        src = tmp
        dst = output if i == len(panels) - 1 else tmp.replace('_tmp', f'_tmp{i}')
        combine_svgs_horizontal(src, panel, dst)
        if os.path.exists(src) and src != dst:
            os.remove(src)
        tmp = dst


def _combine_grid(rows, output):
    """Combine a list of row files vertically into output."""
    if len(rows) == 1:
        shutil.copy(rows[0], output)
        return

    # Combine first two
    tmp = output.replace('.svg', '_tmp.svg')
    combine_svgs_vertical(rows[0], rows[1], tmp if len(rows) > 2 else output)

    # Add remaining rows
    for i, row in enumerate(rows[2:], start=2):
        src = tmp
        dst = output if i == len(rows) - 1 else tmp.replace('_tmp', f'_tmp{i}')
        combine_svgs_vertical(src, row, dst)
        if os.path.exists(src) and src != dst:
            os.remove(src)
        tmp = dst


def _add_grid_labels(svg_in, svg_out, labels, col_widths=None, row_heights=None):
    """Add panel labels at grid positions.

    labels: 2D list of labels, row-major order. None entries are skipped.
    col_widths: width of each column. Defaults to PANEL_WIDTH for all.
    row_heights: height of each row. Defaults to PANEL_HEIGHT for all.

    If every entry is None, `svg_in` is copied to `svg_out` unchanged so
    downstream finalize steps have a valid input.
    """
    n_rows = len(labels)
    n_cols = max(len(row) for row in labels)

    if col_widths is None:
        col_widths = [PANEL_WIDTH] * n_cols
    if row_heights is None:
        row_heights = [PANEL_HEIGHT] * n_rows

    current = svg_in
    for r, row in enumerate(labels):
        y = sum(row_heights[:r]) + 20
        for c, label in enumerate(row):
            if label is None:
                continue
            x = sum(col_widths[:c]) + 10
            add_text_to_svg(current, svg_out, label, x=x, y=y, font_size=14 * 0.75 * SCALE_TEXT)
            current = svg_out

    if current is svg_in:
        shutil.copy(svg_in, svg_out)


def _finalize(labeled, final, panel_files, intermediate_files, cleanup):
    """Scale to final width and optionally clean up."""
    scale_svg(labeled, final, FIG_WIDTH=FIG_WIDTH)

    if cleanup:
        for f in panel_files + intermediate_files:
            if os.path.exists(f):
                os.remove(f)


def _compile_grid(fig_num, layout, labels, cleanup=FIG_CLEANUP, col_widths=None,
                   row_heights=None, prefix=None):
    """Generic grid compilation.

    fig_num: figure number (used to build prefix if prefix is None)
    layout: 2D list of panel labels (e.g., [['A','B','C'], ['D','E','F']])
    labels: 2D list of display labels (can differ from layout for custom labeling)
    prefix: file prefix override; defaults to 'fig{fig_num}'
    """
    prefix = prefix or f'fig{fig_num}'

    # Build panel file paths
    panel_files = []
    for row in layout:
        for label in row:
            if label is not None:
                panel_files.append(FIGURES_DIR + f'{prefix}_{label}' + FIG_FMT)

    # Intermediate files
    row_files = [FIGURES_DIR + f'{prefix}_row{r}.svg' for r in range(len(layout))]
    combined = FIGURES_DIR + f'{prefix}_combined.svg'
    labeled_file = FIGURES_DIR + f'{prefix}_labeled.svg'
    final = FIGURES_DIR + f'{prefix}_final.svg'

    # Build each row
    for r, row in enumerate(layout):
        row_panels = [FIGURES_DIR + f'{prefix}_{label}' + FIG_FMT for label in row if label is not None]
        _combine_row(row_panels, row_files[r])

    # Combine rows
    _combine_grid(row_files, combined)

    # Add labels
    _add_grid_labels(combined, labeled_file, labels, col_widths, row_heights)

    # Finalize
    intermediate = row_files + [combined, labeled_file]
    _finalize(labeled_file, final, panel_files, intermediate, cleanup)


def compile_figure_diagnostics(cleanup=FIG_CLEANUP):
    """Compile the diagnostics figure (not in manuscript): performance vs mismatch (2x3 grid)."""
    _compile_grid(
        fig_num=None,
        layout=[['A', 'B', 'C'], ['D', 'E', 'F']],
        labels=[['A', 'B', 'C'], ['D', 'E', 'F']],
        cleanup=cleanup,
        prefix='figdiag',
    )


def compile_figure_2():
    """Compile Figure 2 → PDF.

    `figures.figure_2` already produces fig2_final.svg as a single-figure
    composition (GridSpec 3×4) — no SVG stitching is needed here. This
    function just wraps the SVG→PDF conversion so run.py mirrors
    compile_figure_3's shape.
    """
    final_svg = FIGURES_DIR + 'fig2_final.svg'
    pdf_file  = FIGURES_DIR + 'fig2_final.pdf'
    try:
        svg_to_pdf(final_svg, pdf_file)
    except (ImportError, OSError) as e:
        print(f"compile_figure_2: PDF conversion skipped ({e}); SVG is available at {final_svg}")


def compile_supplement_fig2():
    """Compile figS1, figS2, figS3 → PDFs alongside the SVGs.

    The supplement panels in `visualization.supplement_fig2` are produced
    as standalone matplotlib figures (no SVG stitching needed); we just
    wrap the SVG→PDF conversion so the manuscript can `\\includegraphics`
    the .pdf files like fig2_final.pdf.
    """
    for stem in ('figS1', 'figS2', 'figS3'):
        final_svg = FIGURES_DIR + stem + '.svg'
        pdf_file  = FIGURES_DIR + stem + '.pdf'
        if not os.path.exists(final_svg):
            continue
        try:
            svg_to_pdf(final_svg, pdf_file)
        except (ImportError, OSError) as e:
            print(f'compile_supplement_fig2: PDF conversion skipped for '
                  f'{stem} ({e}); SVG is at {final_svg}')


def compile_figure_3():
    """Compile Figure 3 → PDF.

    `figures.figure_3` already produces fig3_final.svg as a single-figure
    composition (3×3 stack of compute schedule, success rate, mid-switch).
    This function just wraps the SVG→PDF conversion so run.py mirrors
    compile_figure_2's shape.
    """
    final_svg = FIGURES_DIR + 'fig3_final.svg'
    pdf_file  = FIGURES_DIR + 'fig3_final.pdf'
    try:
        svg_to_pdf(final_svg, pdf_file)
    except (ImportError, OSError) as e:
        print(f"compile_figure_3: PDF conversion skipped ({e}); SVG is available at {final_svg}")


def compile_supplement_fig3():
    """Compile Figure 3 supplement (cumulative cost/s) → PDF."""
    final_svg = FIGURES_DIR + 'fig3_supp.svg'
    pdf_file  = FIGURES_DIR + 'fig3_supp.pdf'
    if not os.path.exists(final_svg):
        print('compile_supplement_fig3: fig3_supp.svg not found, skipped.')
        return
    try:
        svg_to_pdf(final_svg, pdf_file)
    except (ImportError, OSError) as e:
        print(f'compile_supplement_fig3: PDF conversion skipped ({e}); SVG is at {final_svg}')


def compile_supplement_rob():
    """Compile figS4, figS5 (robustness supplements) → PDFs alongside SVGs."""
    for stem in ('figS4', 'figS5'):
        final_svg = FIGURES_DIR + stem + '.svg'
        pdf_file  = FIGURES_DIR + stem + '.pdf'
        if not os.path.exists(final_svg):
            continue
        try:
            svg_to_pdf(final_svg, pdf_file)
        except (ImportError, OSError) as e:
            print(f'compile_supplement_rob: PDF conversion skipped for '
                  f'{stem} ({e}); SVG is at {final_svg}')


def compile_supplement_1(cleanup=FIG_CLEANUP):
    """Compile Supplement 1: single panel."""
    prefix = 'suppl1'
    panel_file = FIGURES_DIR + f'{prefix}_A' + FIG_FMT
    labeled = FIGURES_DIR + f'{prefix}_labeled.svg'
    final = FIGURES_DIR + f'{prefix}_final.svg'

    _add_grid_labels(panel_file, labeled, [['A']])
    _finalize(labeled, final, [panel_file], [labeled], cleanup)


def compile_supplement_2(cleanup=FIG_CLEANUP):
    """Compile Supplement 2: heatmaps (2 full-width rows stacked)."""
    prefix = 'suppl2'
    panel_files = [FIGURES_DIR + f'{prefix}_{p}' + FIG_FMT for p in ['A', 'B']]

    combined = FIGURES_DIR + f'{prefix}_combined.svg'
    labeled = FIGURES_DIR + f'{prefix}_labeled.svg'
    final = FIGURES_DIR + f'{prefix}_final.svg'

    _combine_grid(panel_files, combined)

    import xml.etree.ElementTree as ET
    tree = ET.parse(combined)
    root = tree.getroot()
    total_height = float(root.get('height').replace('pt', ''))
    row_height = total_height / 2

    _add_grid_labels(
        combined, labeled,
        [['A'], ['B']],
        row_heights=[row_height] * 2
    )

    intermediate = [combined, labeled]
    _finalize(labeled, final, panel_files, intermediate, cleanup)


def compile_supplement_3(cleanup=FIG_CLEANUP):
    """Compile Supplement 3: adaptive analysis (3x2 grid)."""
    _compile_grid(
        fig_num=None,
        layout=[['A1', 'A2'], ['B1', 'B2'], ['C1', 'C2']],
        labels=[['A', None], ['B', None], ['C', None]],
        cleanup=cleanup,
        prefix='suppl3'
    )


def compile_supplement_4(cleanup=FIG_CLEANUP):
    """Compile Supplement 4: cartpole (A) + walker (B) perturbation panels.

    Stacks the two panels vertically — horizontal stacking crushes the
    three-axis time-series width. Letters are drawn inside each matplotlib
    figure by `supplement_4`, so the grid-label overlay is a no-op.
    """
    _compile_grid(
        fig_num=None,
        layout=[['A'], ['B']],
        labels=[[None], [None]],
        cleanup=cleanup,
        prefix='suppl4',
    )

    final_svg = FIGURES_DIR + 'suppl4_final.svg'
    pdf_file  = FIGURES_DIR + 'suppl4_final.pdf'
    try:
        svg_to_pdf(final_svg, pdf_file)
    except (ImportError, OSError) as e:
        print(f"compile_supplement_4: PDF conversion skipped ({e}); SVG is available at {final_svg}")

