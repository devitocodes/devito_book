import os
import sys

sys.path.insert(0, os.path.join(os.pardir, "src-vib"))
from math import pi, sin

import numpy as np

from vib import solver, visualize

beta_values = [0.005, 0.05, 0.2]
beta_values = [0.00005]
gamma_values = [5, 1.5, 1.1, 1]
for i, beta in enumerate(beta_values):
    for gamma in gamma_values:
        u, t = solver(
            I=1,
            V=0,
            m=1,
            b=2 * beta,
            s=lambda u: u,
            F=lambda t: sin(gamma * t),
            dt=2 * pi / 60,
            T=2 * pi * 20,
            damping="quadratic",
        )
        visualize(u, t, title=f"gamma={gamma:g}", filename=f"tmp_{gamma}")
        print(gamma, "max u amplitude:", np.abs(u).max())
    for ext in "png", "pdf":
        files = " ".join([f"tmp_{gamma}." + ext for gamma in gamma_values])
        output = "resonance%d.%s" % (i + 1, ext)
        cmd = "montage %s -tile 2x2 -geometry +0+0 %s" % (files, output)
        os.system(cmd)
raw_input()
