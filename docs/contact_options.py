from dataclasses import dataclass

from fe_jax.helper import *
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex":True,
        # "font.family":"serif",
        # "font.sans-serif":["Helvetica"],
        # 'font.size':10,
        'lines.linewidth':2,
        # 'legend.fontsize':9,
        # 'xtick.labelsize':8,
        # 'ytick.labelsize':8
    }
)

DEFAULT_CONTACT_STIFFNESS_MODELS = (
    (linear_elasticity.__contact_stiffness_piecewise_linear, "piecewise linear"),
    (linear_elasticity.__contact_stiffness_piecewise_quadratic, "piecewise quadratic"),
    (linear_elasticity.__contact_stiffness_exponential, "exponential"),
    (linear_elasticity.__contact_stiffness_tanh, "tanh"),
    # (linear_elasticity.__contact_stiffness_linear, "linear"),
)


@dataclass(frozen=True)
class ContactStiffnessPlotValues:
    E: float
    A: float
    radius1: float
    radius2: float
    physical_contact: float
    hard_contact: float
    ramp_up_distance: float
    search_radius: float
    contact_E_c: float
    contact_E_min: float
    contact_material_params: np.ndarray


def contact_stiffness_plot_values(
    contact_options,
    E=1,
    A=1,
    radius1=0.5,
    radius2=0.5,
):
    if contact_options.contact_search_alpha <= contact_options.M_to_D_ratio:
        raise ValueError(
            "contact_search_alpha must be greater than M_to_D_ratio."
        )
    if contact_options.M_to_D_ratio <= contact_options.C_to_D_ratio:
        raise ValueError(
            "M_to_D_ratio must be greater than C_to_D_ratio."
        )

    physical_contact = radius1 + radius2
    contact_E_c = contact_options.D_stiffness_to_E_ratio * E
    contact_E_min = contact_options.M_stiffness_to_E_ratio * E

    return ContactStiffnessPlotValues(
        E=E,
        A=A,
        radius1=radius1,
        radius2=radius2,
        physical_contact=physical_contact,
        hard_contact=contact_options.C_to_D_ratio * physical_contact,
        ramp_up_distance=contact_options.M_to_D_ratio * physical_contact,
        search_radius=contact_options.contact_search_alpha * physical_contact,
        contact_E_c=contact_E_c,
        contact_E_min=contact_E_min,
        contact_material_params=np.array([
            contact_E_c,
            A,
            radius1,
            radius2,
            contact_options.M_to_D_ratio,
            contact_options.C_to_D_ratio,
            contact_options.contact_search_alpha,
            contact_E_min,
        ]),
    )


def _ticks_in_limits(ticks, limits):
    lower, upper = sorted(limits)
    tol = 1e-12 * max(1.0, abs(lower), abs(upper))
    return [
        (value, label)
        for value, label in ticks
        if lower - tol <= value <= upper + tol
    ]


def _set_visible_ticks(ax, values):
    x_ticks = _ticks_in_limits(
        [
            (values.hard_contact, 'hard\ncontact'),
            (values.ramp_up_distance, 'ramp\nup'),
            (values.search_radius, 'search\nradius'),
        ],
        ax.get_xlim(),
    )
    y_ticks = _ticks_in_limits(
        [
            (values.contact_E_c, '$E_c$'),
            (values.E, '$E$'),
            (values.contact_E_min, '$E_{min}$'),
        ],
        ax.get_ylim(),
    )

    if x_ticks:
        ax.set_xticks(
            [value for value, _ in x_ticks],
            [label for _, label in x_ticks],
        )
    if y_ticks:
        ax.set_yticks(
            [value for value, _ in y_ticks],
            [label for _, label in y_ticks],
        )


def style_plot(
    ax,
    values,
    show_physical_contact=False,
    label_physical_contact=False,
):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if show_physical_contact:
        ax.axvspan(0, values.physical_contact, color='gray', alpha=0.3, zorder=0)
        if label_physical_contact:
            label_x0 = max(xlim[0], 0)
            label_x1 = min(xlim[1], values.physical_contact)
            if label_x1 > label_x0:
                ax.text(
                    label_x1,
                    (ylim[0] + ylim[1])/2 ,
                    "physical contact",
                    ha="right",
                    va="center",
                    rotation = 90,
                    color="0.3",
                    fontsize=9,
                )
                # ax.text(
                #     0.5 * (label_x0 + label_x1),
                #     ylim[0] + 0.9 * (ylim[1] - ylim[0]),
                #     "physical contact",
                #     ha="center",
                #     va="top",
                #     color="0.3",
                #     fontsize=9,
                # )

    _set_visible_ticks(ax, values)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, which="major", color="0.75", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _normalize_models(models):
    if models is None:
        return DEFAULT_CONTACT_STIFFNESS_MODELS

    normalized_models = []
    for model in models:
        if isinstance(model, tuple):
            normalized_models.append(model)
        else:
            normalized_models.append(
                (model, getattr(model, "__name__", "contact stiffness"))
            )
    return normalized_models


def _annotate_key_points(ax, values):
    points = [
        (values.hard_contact, values.contact_E_c, r"($c$, $E_c$)"),
        (values.ramp_up_distance, values.contact_E_min, r"($m$, $E_{min}$)"),
    ]
    for x, y, label in points:
        ax.plot(x, y, marker="o", color="black", markersize=4, linestyle="none")
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(6, 6),
            textcoords="offset points",
            ha="left",
            va="bottom",
        )


def plot_contact_stiffness_model(
    contact_options,
    E=1,
    A=1,
    radius1=0.5,
    radius2=0.5,
    ax=None,
    models=None,
    xlim=None,
    ylim=None,
    show_legend=False,
    show_physical_contact=False,
    label_physical_contact=False,
    annotate_key_points=False,
    n_points=401,
):
    if ax is None:
        ax = plt.gca()

    values = contact_stiffness_plot_values(
        contact_options,
        E=E,
        A=A,
        radius1=radius1,
        radius2=radius2,
    )
    if xlim is None:
        xlim = (0, values.search_radius)

    d = np.linspace(xlim[0], xlim[1], n_points)
    for model, label in _normalize_models(models):
        stiffness = model(d, values.contact_material_params)
        ax.plot(d, np.asarray(stiffness), label=label)

    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    style_plot(
        ax,
        values,
        show_physical_contact=show_physical_contact,
        label_physical_contact=label_physical_contact,
    )
    if annotate_key_points:
        _annotate_key_points(ax, values)
    if show_legend:
        ax.legend()

    return ax


def plot_contact_stiffness_model_summary(
    contact_options,
    E=1,
    A=1,
    radius1=0.5,
    radius2=0.5,
    models=None,
    figsize=(8, 6),
):
    values = contact_stiffness_plot_values(
        contact_options,
        E=E,
        A=A,
        radius1=radius1,
        radius2=radius2,
    )
    fig, ax = plt.subplot_mosaic(
        [["top", "top"], ["bottom_left", "bottom_right"]],
        figsize=figsize,
        constrained_layout=True,
    )

    common_plot_options = dict(
        contact_options=contact_options,
        E=E,
        A=A,
        radius1=radius1,
        radius2=radius2,
        models=models,
    )
    plot_contact_stiffness_model(
        ax=ax["top"],
        xlim=(0, values.search_radius),
        ylim=(0, 2 * values.contact_E_c),
        show_legend=True,
        show_physical_contact=True,
        label_physical_contact=True,
        annotate_key_points=True,
        **common_plot_options,
    )
    plot_contact_stiffness_model(
        ax=ax["bottom_left"],
        xlim=(0, values.ramp_up_distance),
        ylim=(0, values.contact_E_c + values.contact_E_min),
        show_physical_contact=True,
        label_physical_contact=True,
        **common_plot_options,
    )
    plot_contact_stiffness_model(
        ax=ax["bottom_right"],
        xlim=(values.ramp_up_distance, values.search_radius),
        ylim=(0, values.contact_E_min),
        **common_plot_options,
    )

    return fig, ax


contact_params=ContactParams(
    self_adjacency_block    = 10000,
    contact_constitutive_model = elastic_contact_truss_piecewise_linear,
    D_stiffness_to_E_ratio  = 4.0,
    # M_stiffness_to_E_ratio  = 1e-6,
    M_stiffness_to_E_ratio  = 0.05,
    M_to_D_ratio            = 1.10,
    C_to_D_ratio            = 0.5,
    contact_search_alpha    = 4.,
)

if __name__ == "__main__":
    fig, ax = plot_contact_stiffness_model_summary(contact_params)
    plt.savefig('docs/contact_stffness_models.pdf')
    plt.show()
