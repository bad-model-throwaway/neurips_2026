"""Publication-quality matplotlib style, applied once on import.

Tuned for high-impact journals (Nature, Science, NeurIPS): colorblind-safe
Okabe-Ito palette, TrueType-embedded fonts for Illustrator/Inkscape editing,
thin hairline axes, outward ticks, no top/right spines, vector SVG output.

Base font size is scaled by SCALE_TEXT because panels are rendered at
native size then downscaled when composed to FIG_WIDTH. Change rcParams
here — do not call rcParams.update in plotting modules.
"""

from cycler import cycler
import matplotlib.pyplot as plt

from configs import SCALE_TEXT


# Okabe-Ito: eight-color palette distinguishable under common color-vision
# deficiencies. Standard in scientific publishing.
OKABE_ITO = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#F0E442',  # yellow
    '#000000',  # black
]


PUBLICATION_RCPARAMS = {
    # Fonts: sans-serif, TrueType so glyphs stay editable in vector editors.
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':        8 * SCALE_TEXT,
    'axes.titlesize':   9 * SCALE_TEXT,
    'axes.labelsize':   8 * SCALE_TEXT,
    'xtick.labelsize':  7 * SCALE_TEXT,
    'ytick.labelsize':  7 * SCALE_TEXT,
    'legend.fontsize':  7 * SCALE_TEXT,
    'figure.titlesize': 9 * SCALE_TEXT,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
    'svg.fonttype': 'none',
    'text.usetex':  False,

    # Axes: hairline spines, no top/right, light subtle grid off by default.
    'axes.prop_cycle':    cycler('color', OKABE_ITO),
    'axes.linewidth':     0.8,
    'axes.edgecolor':     '#222222',
    'axes.labelcolor':    '#222222',
    'axes.titleweight':   'regular',
    'axes.titlepad':      4.0,
    'axes.labelpad':      3.0,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          False,
    'axes.axisbelow':     True,

    # Ticks: outward, short, hairline.
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   3.0,
    'ytick.major.size':   3.0,
    'xtick.minor.size':   1.5,
    'ytick.minor.size':   1.5,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.minor.width':  0.6,
    'ytick.minor.width':  0.6,
    'xtick.color':        '#222222',
    'ytick.color':        '#222222',
    'xtick.major.pad':    2.5,
    'ytick.major.pad':    2.5,

    # Grid: when turned on per-axes, light and subtle.
    'grid.color':     '#B0B0B0',
    'grid.linestyle': '-',
    'grid.linewidth': 0.4,
    'grid.alpha':     0.4,

    # Lines and markers.
    'lines.linewidth':       1.25,
    'lines.markersize':      4.5,
    'lines.markeredgewidth': 0.8,
    'patch.linewidth':       0.6,

    # Legend: frameless, tight.
    'legend.frameon':       False,
    'legend.borderpad':     0.3,
    'legend.labelspacing':  0.3,
    'legend.handlelength':  1.6,
    'legend.handletextpad': 0.5,
    'legend.columnspacing': 1.0,

    # Figure and savefig: white background, vector-friendly defaults.
    'figure.facecolor':  'white',
    'figure.edgecolor':  'white',
    'figure.dpi':        120,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
    'savefig.transparent': False,

    # Image defaults (heatmaps).
    'image.cmap':        'viridis',
    'image.interpolation': 'nearest',

    # Errorbar: match marker sizing.
    'errorbar.capsize': 2.5,
}


def apply():
    """Apply the publication rcParams to the global matplotlib state."""
    plt.rcParams.update(PUBLICATION_RCPARAMS)


# Apply on import so any `from visualization import ...` picks up the style.
apply()
plt.ion()
