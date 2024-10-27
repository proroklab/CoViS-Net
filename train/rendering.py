import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import torch_geometric
from matplotlib.patches import ConnectionPatch, Arc, Ellipse
import roma


def polar(angle, dist):
    return torch.stack([angle.sin(), angle.cos()]) * dist


def plot_marker(ax, p, p_var, heading, heading_var, color):
    fov = 120
    dist = 0.75
    fov_half = fov / 2
    var_linestyle = "--"
    fov_half_rad = np.deg2rad(fov_half)
    p_fov_l = p + polar(torch.Tensor([heading + fov_half_rad])[0], dist)
    ax.plot([p[0], p_fov_l[0]], [p[1], p_fov_l[1]], c=color)
    p_fov_r = p + polar(torch.Tensor([heading - fov_half_rad])[0], dist)
    ax.plot([p[0], p_fov_r[0]], [p[1], p_fov_r[1]], c=color)

    if heading_var is not None:
        heading_std = heading_var.sqrt()
        p_var_l = p + polar(torch.Tensor([heading + 2 * heading_std])[0], dist)
        ax.plot(
            [p[0], p_var_l[0]], [p[1], p_var_l[1]], c=color, linestyle=var_linestyle
        )
        p_var_r = p + polar(torch.Tensor([heading - 2 * heading_std])[0], dist)
        ax.plot(
            [p[0], p_var_r[0]], [p[1], p_var_r[1]], c=color, linestyle=var_linestyle
        )

    arc = Arc(
        p,
        dist * 2,
        dist * 2,
        angle=np.rad2deg(-heading + np.pi / 2),
        theta1=-fov_half,
        theta2=fov_half,
        color=color,
    )
    ax.add_patch(arc)

    if p_var is not None:
        p_std = p_var.sqrt()
        arc_var = Ellipse(
            p,
            2 * p_std[1],
            2 * p_std[0],
            angle=np.rad2deg(0.0),
            color=color,
            fc="none",
            linestyle=var_linestyle,
        )
        ax.add_patch(arc_var)


def render_single(img, bev_label, bev_pred, edge_index, edge_label, edge_pred):
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    n_nodes = img.shape[0]

    fig, axs = plt.subplot_mosaic(
        [
            [f"img_{j}" for j in range(n_nodes)],
            [f"gt_{j}" for j in range(n_nodes)],
            [f"pred_{j}" for j in range(n_nodes)],
        ],
        figsize=[12, 7],
    )
    axs["img_0"].set_ylabel("Image\n\n")
    axs["gt_0"].set_ylabel("Ground Truth\ny [m]")
    axs["pred_0"].set_ylabel("Prediction\ny [m]")

    for i in range(n_nodes):
        for ax_label in ["pred", "gt"]:
            ax = axs[f"{ax_label}_{i}"]
            ax.set_aspect("equal")
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.grid()
            plot_marker(ax, torch.zeros(2), None, 0.0, None, colors[i])

    for i in range(n_nodes):
        ax = axs[f"img_{i}"]
        ax.imshow(img[i].permute(1, 2, 0))

        # Colored frame
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(colors[i])
            spine.set_linewidth(2)

        axs[f"gt_{i}"].imshow(
            bev_label[i][0],
            vmin=0,
            vmax=1,
            cmap="gray",
            alpha=0.5,
            extent=[-3, 3, -3, 3],
            interpolation="nearest",
        )
        if bev_pred is not None:
            axs[f"pred_{i}"].imshow(
                bev_pred[i][0],
                vmin=0,
                vmax=1,
                cmap="gray",
                alpha=0.5,
                extent=[-3, 3, -3, 3],
                interpolation="nearest",
            )
        if i > 0:
            axs[f"pred_{i}"].set_yticklabels([])
            axs[f"gt_{i}"].set_yticklabels([])
        axs[f"gt_{i}"].set_xticklabels([])
        axs[f"pred_{i}"].set_xlabel("x [m]")

    for i, j, p, q in zip(
        edge_index[1],
        edge_index[0],
        edge_label["pos"],
        edge_label["rot"],
    ):
        ax = axs[f"gt_{j}"]
        heading = roma.unitquat_to_rotvec(q)[1]
        pos = torch.Tensor([-p[0], p[2]])
        plot_marker(ax, pos, None, heading, None, colors[i])

    if not any([v is None for v in edge_pred.values()]):
        for i, j, p_pred, q_pred, p_var, q_var in zip(
            edge_index[1],
            edge_index[0],
            edge_pred["pos"],
            edge_pred["rot"],
            edge_pred["pos_var"],
            edge_pred["rot_var"],
        ):
            ax = axs[f"pred_{j}"]
            heading = roma.unitquat_to_rotvec(q_pred)[1]
            pos = torch.Tensor([-p_pred[0], p_pred[2]])
            pos_var = torch.Tensor([p_var[2], p_var[0]])
            if pos_var[0] < 1.5 or pos_var[1] < 1.5:
                plot_marker(ax, pos, pos_var, heading, q_var, colors[i])

    return fig


def unbatch_dict(data, indexes):
    if data is None:
        return None

    unbatched_dict = {}
    for key, value in data.items():
        if value is None:
            unbatched_dict[key] = None
        else:
            unbatched_dict[key] = torch_geometric.utils.unbatch(value, indexes)
    return unbatched_dict


def render_batch(
    datas_batched, labels_batched, edge_index, edge_batch, edge_preds, node_preds
):
    n = 16

    labels_pos = labels_batched["pos"].flatten(0, 1)
    pos_sink = labels_pos[edge_index[1]]
    pos_source = labels_pos[edge_index[0]]

    labels_rot = labels_batched["rot_quat"].flatten(0, 1)
    angle_sink = labels_rot[edge_index[1]]
    angle_source = labels_rot[edge_index[0]]

    rel_pos = pos_source - pos_sink
    rel_pos_source = roma.RotationUnitQuat(
        linear=roma.quat_inverse(angle_source)
    ).apply(rel_pos)
    angle_between_nodes = roma.quat_product(angle_source, roma.quat_inverse(angle_sink))
    edge_labels = {
        "pos": rel_pos_source,
        "rot": angle_between_nodes,
    }
    edge_labels_unbatched = unbatch_dict(edge_labels, edge_batch[edge_index[1]])
    edge_preds_unbatched = unbatch_dict(edge_preds, edge_batch[edge_index[1]])
    edge_index_unbatched = torch_geometric.utils.unbatch_edge_index(
        edge_index, edge_batch
    )

    if edge_preds_unbatched is None:
        edge_preds_unbatched = {
            "pos": [None] * len(edge_labels_unbatched),
            "rot": [None] * len(edge_labels_unbatched),
            "pos_var": [None] * len(edge_labels_unbatched),
            "rot_var": [None] * len(edge_labels_unbatched),
        }

    if node_preds is None:
        node_preds = [None] * len(datas_batched["img_raw"])

    renderings = []
    for (
        i,
        img,
        edge_index,
        bev_label,
        edge_label_pos,
        edge_label_rot,
        bev_pred,
        edge_pred_pos,
        edge_pred_rot,
        edge_pred_pos_var,
        edge_pred_rot_var,
    ) in zip(
        range(n),
        datas_batched["img_raw"],
        edge_index_unbatched,
        labels_batched["topdowns_agents"],
        edge_labels_unbatched["pos"],
        edge_labels_unbatched["rot"],
        node_preds,
        edge_preds_unbatched["pos"],
        edge_preds_unbatched["rot"],
        edge_preds_unbatched["pos_var"],
        edge_preds_unbatched["rot_var"],
    ):
        edge_label = {"pos": edge_label_pos, "rot": edge_label_rot}
        edge_pred = {
            "pos": edge_pred_pos,
            "rot": edge_pred_rot,
            "pos_var": edge_pred_pos_var,
            "rot_var": edge_pred_rot_var,
        }
        fig = render_single(img, bev_label, bev_pred, edge_index, edge_label, edge_pred)
        renderings.append(wandb.Image(fig))
        plt.close()

    return renderings
