import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import torch_geometric


def polar(angle, dist):
    return torch.stack([angle.sin(), angle.cos()]) * dist


def render_bev(dataset, datas, labels, preds):
    # mpl.use("agg")
    n = 16
    batch_size, n_nodes = datas["img_raw"].shape[:2]

    renderings = []
    for i, img, topdown, pos, topdown_pixel_pos, rotations, label, pred in zip(
        range(n),
        datas["img_raw"],
        labels["topdown_global"],
        labels["pos"],
        labels["pixel_pos"],
        labels["rot"],
        labels["topdowns_agents"],
        preds,
    ):
        fig, axs = plt.subplot_mosaic(
            [
                [f"img_{j}" for j in range(n_nodes)],
                ["m"] * n_nodes,
                [f"label_{j}" for j in range(n_nodes)],
                [f"pred_{j}" for j in range(n_nodes)],
            ],
            layout="constrained",
            figsize=[12, 5],
        )
        axs["m"].imshow(topdown)
        for j in range(n_nodes):
            ax = axs[f"img_{j}"]
            ax.imshow(img[j].permute(1, 2, 0))
            ax.axis("off")
            axs["m"].scatter([topdown_pixel_pos[j][0]], [topdown_pixel_pos[j][1]])
            ax.add_artist(
                patches.ConnectionPatch(
                    axesA=ax,
                    xyA=(0.5, 0),
                    coordsA="axes fraction",
                    axesB=axs["m"],
                    coordsB="data",
                    xyB=topdown_pixel_pos[j],
                    color="blue",
                )
            )

            l = polar(torch.Tensor([rotations[j][1] - np.pi])[0], 20.0)
            axs["m"].arrow(
                topdown_pixel_pos[j][0],
                topdown_pixel_pos[j][1],
                l[0],
                l[1],
                width=0.1,
            )

        axs["m"].set_zorder(-1)

        for j in range(n_nodes):
            axs[f"label_{j}"].imshow(label[j, 0], vmin=0, vmax=1)
        for j in range(n_nodes):
            axs[f"pred_{j}"].imshow(pred[j, 0], vmin=0, vmax=1)

        fig.canvas.draw()

        renderings.append(wandb.Image(fig))

        plt.close()

    return renderings


def plot_marker(ax, p, heading, color):
    ax.scatter([p[0]], [p[1]], marker="o", facecolors="none", edgecolors=color)

    p_heading = p + polar(torch.Tensor([heading])[0], 0.5)
    ax.plot([p[0], p_heading[0]], [p[1], p_heading[1]], c=color)


def plot_edge_data(axs, edge_index, edge_data, edge_var, color):
    for j, label, var in zip(edge_index[0], edge_data, edge_var):
        angle_head_j_to_k = label[3]
        p_k = torch.Tensor([-label[0], label[2]])

        plot_marker(axs[j], p_k, angle_head_j_to_k, color)
        if var.shape[0] > 0:
            axs[j].text(
                p_k[0],
                p_k[1],
                f"p {var[0]:.2f} {var[1]:.2f} {var[2]:.2f} r {var[3]:.2f}",
                color=color,
                fontsize="x-small",
            )


def render_localize(dataset, datas, labels, preds, edge_feature_labels):
    # mpl.use("agg")
    n = 128
    batch_size, n_nodes = datas["img_raw"].shape[:2]
    edge_preds, edge_vars, edge_indexes, edge_batch = preds

    edge_indexes_batched = torch_geometric.utils.unbatch_edge_index(
        edge_indexes, edge_batch
    )
    edge_preds_batched = torch_geometric.utils.unbatch(
        edge_preds, edge_batch[edge_indexes[1]]
    )
    edge_vars_batched = torch_geometric.utils.unbatch(
        edge_vars, edge_batch[edge_indexes[1]]
    )
    edge_labels_batched = torch_geometric.utils.unbatch(
        edge_feature_labels, edge_batch[edge_indexes[1]]
    )

    renderings = []
    for (
        i,
        img,
        topdown,
        pos,
        topdown_pixel_pos,
        rotations,
        edge_index,
        edge_label,
        edge_pred,
        edge_var,
    ) in zip(
        range(n),
        datas["img_raw"],
        labels["topdown_global"],
        datas["pos"],
        labels["pixel_pos"],
        labels["rot"],
        edge_indexes_batched,
        edge_labels_batched,
        edge_preds_batched,
        edge_vars_batched,
    ):
        fig, axs = plt.subplot_mosaic(
            [
                [f"img_{j}" for j in range(n_nodes)],
                ["m"] * n_nodes,
                [f"pos_{j}" for j in range(n_nodes)],
            ],
            figsize=[12, 5],
        )
        axs["m"].imshow(topdown)
        for j in range(n_nodes):
            ax = axs[f"img_{j}"]
            ax.imshow(img[j].permute(1, 2, 0))
            ax.axis("off")
            axs["m"].scatter([topdown_pixel_pos[j][0]], [topdown_pixel_pos[j][1]])
            ax.add_artist(
                patches.ConnectionPatch(
                    axesA=ax,
                    xyA=(0.5, 0),
                    coordsA="axes fraction",
                    axesB=axs["m"],
                    coordsB="data",
                    xyB=topdown_pixel_pos[j],
                    color="blue",
                )
            )

            l = polar(torch.Tensor([rotations[j][1] - np.pi])[0], 20.0)
            axs["m"].arrow(
                topdown_pixel_pos[j][0],
                topdown_pixel_pos[j][1],
                l[0],
                l[1],
                width=0.1,
            )

        axs["m"].set_zorder(-1)

        axs_pos = [axs[f"pos_{j}"] for j in range(n_nodes)]
        for ax in axs_pos:  # own pose (origin)
            plot_marker(ax, torch.zeros(2), 0.0, "black")
        plot_edge_data(axs_pos, edge_index, edge_pred, edge_var, color="r")
        empty_edge_var = torch.tensor([[]] * len(edge_var))
        plot_edge_data(axs_pos, edge_index, edge_label, empty_edge_var, color="b")

        red_patch = patches.Patch(color="r", label="Prediction")
        blue_patch = patches.Patch(color="b", label="Ground Truth")
        axs_pos[0].legend(
            handles=[red_patch, blue_patch], bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc=3
        )

        # Make axis labels amongst all axis in bottom similar
        limits = []
        for j in range(n_nodes):
            ax = axs[f"pos_{j}"]
            ax.grid()
            limits.append([ax.get_xlim(), ax.get_ylim()])
        limits = np.array(limits)
        for j in range(n_nodes):
            ax = axs[f"pos_{j}"]
            ax.set_xlim(
                [np.array(limits)[:, 0, 0].min(), np.array(limits)[:, 0, 1].max()]
            )
            ax.set_ylim(
                [np.array(limits)[:, 1, 0].min(), np.array(limits)[:, 1, 1].max()]
            )
            ax.set_aspect("equal")
        fig.canvas.draw()
        renderings.append(wandb.Image(fig))
        plt.close()

    return renderings
