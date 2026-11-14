import sys

import numpy as np
import matplotlib
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.path import Path
try:
    # Matplotlib >= 3.5 (any of PyQt6, PySide6, PyQt5, PySide2)
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except (ImportError, ValueError):
    # older Matplotlib (Qt5 only)
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.backends.qt_compat import QtWidgets, QtGui

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['cmr10']
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.unicode_minus'] = False

t = np.linspace(0, 2 * np.pi, 200)
astroid = Path(np.column_stack([np.cos(t)**3, np.sin(t)**3]), closed=True)

letters_dict = {
    'a': [False, True, False, False, True, False, True, False, True],
    'e': [True, True, True, False, False, False, True, True, True],
    'l': [False, False, True, False, False, False, True, True, True],
    'k': [True, False, True, False, False, False, True, True, True],
    'h': [True, False, True, False, False, False, True, False, True],
    'x': [True, False, True, False, True, False, True, False, True],
    's': [True, False, False, False, True, False, False, False, True],
    'n': [True, True, True, False, False, False, True, False, True],
    'm': [True, False, True, True, True, True, True, False, True],
    'd': [False, True, False, False, False, False, True, True, True],
    'f': [True, True, False, False, False, False, True, True, True],
    'c': [True, False, True, False, False, False, False, True, False],
    'v': [True, False, True, False, True, False, False, False, False],
    'w': [True, False, True, True, True, True, False, True, False],
    'p': [True, False, False, False, True, False, True, True, True],
    'r': [True, False, False, False, True, True, True, True, True],
    'g': [True, False, True, False, False, True, False, True, False],
    'o': [True, True, True, True, False, True, True, True, True],
    'q': [True, True, True, True, False, True, True, True, False],
    'u': [True, False, True, True, False, True, True, True, True],
    'b': [True, False, True, False, True, False, True, True, True],
    'i': [False, False, False, True, True, True, False, False, False],
    't': [True, False, False, True, True, True, True, False, False],
    'z': [True, False, True, True, True, True, True, False, True],
    'y': [True, False, False, False, True, True, True, False, False],
    'j': [False, False, False, True, True, True, False, False, True],
    'A': [False, True, False, False, True, False, True, False, True],
    'E': [True, True, True, False, False, False, True, True, True],
    'L': [False, False, True, False, False, False, True, True, True],
    'K': [True, False, True, False, False, False, True, True, True],
    'H': [True, False, True, False, False, False, True, False, True],
    'X': [True, False, True, False, True, False, True, False, True],
    'S': [True, False, False, False, True, False, False, False, True],
    'N': [True, True, True, False, False, False, True, False, True],
    'M': [True, False, True, True, True, True, True, False, True],
    'D': [False, True, False, False, False, False, True, True, True],
    'F': [True, True, False, False, False, False, True, True, True],
    'C': [True, False, True, False, False, False, False, True, False],
    'V': [True, False, True, False, True, False, False, False, False],
    'W': [True, False, True, True, True, True, False, True, False],
    'P': [True, False, False, False, True, False, True, True, True],
    'R': [True, False, False, False, True, True, True, True, True],
    'G': [True, False, True, False, False, True, False, True, False],
    'O': [True, True, True, True, False, True, True, True, True],
    'Q': [True, True, True, True, False, True, True, True, False],
    'U': [True, False, True, True, False, True, True, True, True],
    'B': [True, False, True, False, True, False, True, True, True],
    'I': [False, False, False, True, True, True, False, False, False],
    'T': [True, False, False, True, True, True, True, False, False],
    'Z': [True, False, True, True, True, True, True, False, True],
    'Y': [True, False, False, False, True, True, True, False, False],
    'J': [False, False, False, True, True, True, False, False, True],
}

length = 0.15                # edge length
check_radius = 0.15          # minimum distance between nodes
branch_prob = 0.5            # chance the next letter shares the current letter's start node
max_letters_per_node = 3     # max letters starting from one node
branch_turn = (60, 150)      # clockwise turn (degrees) of a branch relative to the previous letter
grid_spacing = 0.2           # letter grid spacing, as a fraction of edge length


def segments_cross(p1, p2, p3, p4):
    """True if segment p1-p2 properly crosses segment p3-p4."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (orient(p3, p4, p1) * orient(p3, p4, p2) < 0 and
            orient(p1, p2, p3) * orient(p1, p2, p4) < 0)


def random_free_angle(rng):
    cosphi = 0.3 + 0.7 * rng.random()
    cossign = rng.choice([-1, 1])
    sinsign = rng.choice([-1, 1])
    return np.arctan2(sinsign * np.sqrt(1 - cosphi**2), cossign * cosphi)


def try_layout(word, rng, max_tries=300):
    """Returns (nodes, edges) or None if it got stuck. Letter i is drawn on edges[i]."""
    nodes = [0.25 + 0.5 * rng.random(2)]
    edges = []
    letters_from = [0]

    for i in range(len(word)):
        branch = False
        if i == 0:
            start = 0
        else:
            prev_start, prev_end = edges[-1]
            if letters_from[prev_start] < max_letters_per_node and rng.random() < branch_prob:
                branch = True
                start = prev_start
            else:
                start = prev_end

        for _ in range(max_tries):
            if branch:
                ps, pe = edges[-1]
                dx, dy = nodes[pe] - nodes[ps]
                angle = np.arctan2(dy, dx) - np.radians(rng.uniform(*branch_turn))
            else:
                angle = random_free_angle(rng)
            new = nodes[start] + length * np.array([np.cos(angle), np.sin(angle)])

            if any(np.linalg.norm(new - n) < check_radius for n in nodes):
                continue
            if any(segments_cross(nodes[start], new, nodes[a], nodes[b])
                   for a, b in edges if start not in (a, b)):
                continue
            break
        else:
            return None

        nodes.append(new)
        letters_from.append(0)
        letters_from[start] += 1
        edges.append((start, len(nodes) - 1))

    return np.array(nodes), edges


def make_layout(word, rng):
    for _ in range(1000):
        layout = try_layout(word, rng)
        if layout is not None:
            return layout
    raise RuntimeError("Couldn't find a layout for this word.")


def draw_word(ax, word, rng):
    ax.clear()
    nodes, edges = make_layout(word, rng)

    along = 0.5 + grid_spacing * np.array([-1, 0, 1] * 3)
    across = grid_spacing * np.array([1] * 3 + [0] * 3 + [-1] * 3)

    for letter, (a, b) in zip(word, edges):
        p1, p2 = nodes[a], nodes[b]
        if p2[0] < p1[0] or (p2[0] == p1[0] and p2[1] < p1[1]):
            p1, p2 = p2, p1
        d = p2 - p1
        normal = np.array([-d[1], d[0]])
        pts = p1 + np.outer(along, d) + np.outer(across, normal)
        mask = np.array(letters_dict[letter])
        ax.scatter(pts[mask, 0], pts[mask, 1], color='black', marker=astroid,
                   s=150, edgecolor='white', zorder=100)

    for a, b in edges:
        ax.plot(nodes[[a, b], 0], nodes[[a, b], 1], color='black', zorder=1)
    ax.plot(nodes[:, 0], nodes[:, 1], linestyle='', color='black', marker=astroid,
            markersize=35, markeredgecolor='white', markeredgewidth=2, zorder=2)
    ax.plot(nodes[0, 0], nodes[0, 1], linestyle='', marker='o', markersize=50,
            markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.5,
            zorder=3, clip_on=False)

    ax.set_aspect('equal')
    ax.margins(0.1)
    ax.axis('off')


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Astronymic')
        self.resize(1000, 650)

        # left: matplotlib canvas + toolbar
        self.fig = Figure(figsize=(7, 6))
        self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvasQTAgg(self.fig)
        plot_box = QtWidgets.QVBoxLayout()
        plot_box.addWidget(NavigationToolbar2QT(self.canvas, self))
        plot_box.addWidget(self.canvas)

        # right: controls
        self.word_edit = QtWidgets.QLineEdit('tamara')
        self.seed_edit = QtWidgets.QLineEdit('0')
        self.seed_edit.setValidator(QtGui.QIntValidator(0, 2_000_000_000))
        random_btn = QtWidgets.QPushButton('Random')
        draw_btn = QtWidgets.QPushButton('Draw')
        self.status = QtWidgets.QLabel('')
        self.status.setStyleSheet('color: firebrick')
        self.status.setWordWrap(True)

        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(self.seed_edit)
        seed_row.addWidget(random_btn)

        panel = QtWidgets.QVBoxLayout()
        panel.addWidget(QtWidgets.QLabel('Word'))
        panel.addWidget(self.word_edit)
        panel.addSpacing(10)
        panel.addWidget(QtWidgets.QLabel('Seed'))
        panel.addLayout(seed_row)
        panel.addSpacing(10)
        panel.addWidget(draw_btn)
        panel.addWidget(self.status)
        panel.addStretch()

        panel_widget = QtWidgets.QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setFixedWidth(220)

        main = QtWidgets.QHBoxLayout()
        main.addLayout(plot_box, stretch=1)
        main.addWidget(panel_widget)
        central = QtWidgets.QWidget()
        central.setLayout(main)
        self.setCentralWidget(central)

        draw_btn.clicked.connect(self.redraw)
        random_btn.clicked.connect(self.random_seed)
        self.word_edit.returnPressed.connect(self.redraw)
        self.seed_edit.returnPressed.connect(self.redraw)

        self.redraw()

    def random_seed(self):
        self.seed_edit.setText(str(np.random.randint(0, 1_000_000)))
        self.redraw()

    def redraw(self):
        word = ''.join(c for c in self.word_edit.text().lower() if c in letters_dict)
        if not word:
            self.status.setText('Enter a word with letters a–z.')
            return
        try:
            seed = int(self.seed_edit.text())
        except ValueError:
            self.status.setText('Seed must be a whole number.')
            return
        try:
            draw_word(self.ax, word, np.random.default_rng(seed))
        except RuntimeError as err:
            self.status.setText(str(err))
            return
        self.status.setText('')
        self.canvas.draw()


if __name__ == '__main__':
    qapp = QtWidgets.QApplication(sys.argv)
    window = App()
    window.showMaximized()
    # PySide2 and older PyQt5 only have exec_()
    run = getattr(qapp, 'exec', None) or qapp.exec_
    sys.exit(run())
