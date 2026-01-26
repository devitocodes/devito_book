from numpy import arcsin as asin
from numpy import linspace, pi
from scitools.std import hold, plot, savefig


def tilde_w(w, dt):
    return (2.0 / dt) * asin(w * dt / 2.0)


def plot_phase_error():
    w = 1  # relevant value in a scaled problem
    m = linspace(1, 101, 101)
    period = 2 * pi / w
    dt_values = [
        period / num_timesteps_per_period for num_timesteps_per_period in (4, 8, 16, 32)
    ]
    for dt in dt_values:
        e = m * 2 * pi * (1.0 / w - 1 / tilde_w(w, dt))
        plot(
            m,
            e,
            "-",
            title="peak location error (phase error)",
            xlabel="no of periods",
            ylabel="phase error",
        )
        hold("on")
    savefig("phase_error.png")


plot_phase_error()
