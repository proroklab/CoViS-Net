import torch
import roma
import wandb
from pathlib import Path
import shutil
import os
from typing import Tuple
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch import seed_everything
import torchmetrics
import torch_geometric
import pandas as pd
from torchmetrics.classification import BinaryJaccardIndex

from .rendering import render_batch
from .dataloader import RelPosDataModule
from .dice_loss import DiceLoss


def to_device(obj, device):
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple([to_device(v, device) for v in obj])
    elif torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    elif isinstance(obj, str):
        return obj
    else:
        raise NotImplementedError


def quat_norm_diff(q_a, q_b):
    # https://github.com/utiasSTARS/bingham-rotation-learning/blob/master/quaternions.py
    assert q_a.shape == q_b.shape
    assert q_a.shape[-1] == 4
    if q_a.dim() < 2:
        q_a = q_a.unsqueeze(0)
        q_b = q_b.unsqueeze(0)
    return torch.min((q_a - q_b).norm(dim=1), (q_a + q_b).norm(dim=1)).squeeze()


def quat_chordal_squared_loss(q, q_target, reduce=True):
    # https://github.com/utiasSTARS/bingham-rotation-learning/blob/master/losses.py
    assert q.shape == q_target.shape
    d = quat_norm_diff(q, q_target)
    losses = 2 * d * d * (4.0 - d * d)
    loss = losses.mean() if reduce else losses
    return loss


def quat_gaussian_nll(q, q_target, var, eps=1e-5):
    l = quat_chordal_squared_loss(q, q_target, reduce=False)
    var_eps = torch.maximum(var, torch.tensor(eps))
    return ((var_eps.log() + l / var_eps) / 2.0).mean()


def quat_gaussian_nll_beta(q, q_target, var, eps=1e-5, beta=0.5):
    l = quat_chordal_squared_loss(q, q_target, reduce=False)
    var_eps = torch.maximum(var, torch.tensor(eps))
    return var_eps.detach() ** beta * ((var_eps.log() + l / var_eps) / 2.0).mean()


class TrainerModuleLocalize(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        initial_learning_rate: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-4,
        max_learning_rate: float = 0.05,
        warmup_epochs: int = 5,
        num_epochs: int = 100,
        seed: int = 43,
    ):
        super().__init__()

        self.model = model
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss("binary")
        self.iou = BinaryJaccardIndex()

        self.initial_learning_rate = initial_learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.max_learning_rate = max_learning_rate
        self.warmup_epochs = warmup_epochs
        self.num_epochs = num_epochs

        self.eval_preds = {}

        self.save_hyperparameters(ignore=["model"])

    def export_preds(self, inputs, outputs, meta, bev=False):
        (edge_preds, node_preds), (edge_index, edge_batch) = outputs

        e_idx_b = torch_geometric.utils.unbatch_edge_index(edge_index, edge_batch)

        p_gt_b = torch_geometric.utils.unbatch(
            meta["rel_pos"], edge_batch[edge_index[1]]
        )
        p_p_b = torch_geometric.utils.unbatch(
            edge_preds["pos"], edge_batch[edge_index[1]]
        )
        p_v_b = torch_geometric.utils.unbatch(
            edge_preds["pos_var"], edge_batch[edge_index[1]]
        )

        q_gt_b = torch_geometric.utils.unbatch(
            meta["rel_rot"], edge_batch[edge_index[1]]
        )
        q_p_b = torch_geometric.utils.unbatch(
            edge_preds["rot"], edge_batch[edge_index[1]]
        )
        q_v_b = torch_geometric.utils.unbatch(
            edge_preds["rot_var"], edge_batch[edge_index[1]]
        )

        preds = []
        for b_idx, (d_idx, e_idx, ps_gt, ps_p, ps_v, qs_gt, qs_p, qs_v) in enumerate(
            zip(inputs["idx"], e_idx_b, p_gt_b, p_p_b, p_v_b, q_gt_b, q_p_b, q_v_b)
        ):
            preds.append(
                pd.DataFrame(
                    {
                        "node_source": e_idx[1].cpu().numpy(),
                        "node_sink": e_idx[0].cpu().numpy(),
                        "px_pred": ps_p[:, 0].cpu().numpy(),
                        "py_pred": ps_p[:, 1].cpu().numpy(),
                        "pz_pred": ps_p[:, 2].cpu().numpy(),
                        "px_var": ps_v[:, 0].cpu().numpy(),
                        "py_var": ps_v[:, 1].cpu().numpy(),
                        "pz_var": ps_v[:, 2].cpu().numpy(),
                        "qx_pred": qs_p[:, 0].cpu().numpy(),
                        "qy_pred": qs_p[:, 1].cpu().numpy(),
                        "qz_pred": qs_p[:, 2].cpu().numpy(),
                        "qw_pred": qs_p[:, 3].cpu().numpy(),
                        "q_var": qs_v[:, 0].cpu().numpy(),
                        "px_gt": ps_gt[:, 0].cpu().numpy(),
                        "py_gt": ps_gt[:, 1].cpu().numpy(),
                        "pz_gt": ps_gt[:, 2].cpu().numpy(),
                        "qx_gt": qs_gt[:, 0].cpu().numpy(),
                        "qy_gt": qs_gt[:, 1].cpu().numpy(),
                        "qz_gt": qs_gt[:, 2].cpu().numpy(),
                        "qw_gt": qs_gt[:, 3].cpu().numpy(),
                        "d_idx": d_idx.item(),
                        "b_idx": b_idx,
                    }
                )
            )
        return preds, node_preds.cpu()

    def loss_function(self, outputs, labels):
        (edge_preds, node_preds), (edge_index, _) = outputs
        labels_pos = labels["pos"].flatten(0, 1)
        pos_sink = labels_pos[edge_index[1]]
        pos_source = labels_pos[edge_index[0]]

        labels_rot = labels["rot_quat"].flatten(0, 1)
        angle_sink = labels_rot[edge_index[1]]
        angle_source = labels_rot[edge_index[0]]

        rel_pos = pos_source - pos_sink
        rel_pos_source = roma.RotationUnitQuat(
            linear=roma.quat_inverse(angle_source)
        ).apply(rel_pos)
        angle_between_nodes = roma.quat_product(
            angle_source, roma.quat_inverse(angle_sink)
        )

        dist_to_other = torch.norm(rel_pos, p=2, dim=1)

        dist_metric = torch.norm(edge_preds["pos"] - rel_pos_source, dim=1)
        dist_loss = torch.nn.GaussianNLLLoss()(
            edge_preds["pos"], rel_pos_source, edge_preds["pos_var"]
        )

        angle_between_loss = quat_gaussian_nll(
            edge_preds["rot"], angle_between_nodes, edge_preds["rot_var"][:, 0]
        )

        angle_between_node_gt = roma.unitquat_to_rotvec(angle_between_nodes.float())[
            :, 1
        ]
        angle_between_m = roma.unitquat_geodesic_distance(
            edge_preds["rot"], angle_between_nodes
        )

        bce_loss = self.bce_loss(node_preds, labels["topdowns_agents"])
        dice_loss = self.dice_loss(node_preds, labels["topdowns_agents"])
        bev_loss = 0.25 * bce_loss + 0.75 * dice_loss

        return (
            dist_loss,
            angle_between_loss,
            bev_loss,
            {
                "angle_between_m": angle_between_m,
                "angle_between_nodes_gt": angle_between_node_gt,
                "dist_metric": dist_metric,
                "dist": dist_to_other,
                "dist_pred_var": edge_preds["pos_var"],
                "angle_between_nodes_pred_var": edge_preds["rot_var"][:, 0],
                "bev_bce": bce_loss,
                "bev_dice": dice_loss,
                "rel_pos": rel_pos_source,
                "rel_rot": angle_between_nodes,
            },
        )

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        dist_loss, angle_between_loss, bev_loss, meta = self.loss_function(
            outputs, labels
        )
        loss = dist_loss + angle_between_loss + 1.5 * bev_loss

        self.log_dict(
            {
                "train_loss": loss,
                "train_dist_loss": meta["dist_metric"].mean(),
                "train_angle_between_loss": meta["angle_between_m"].mean(),
                "train_dist_pred_var": meta["dist_pred_var"].mean(),
                "train_angle_between_nodes_pred_var": meta[
                    "angle_between_nodes_pred_var"
                ].mean(),
                "train_bce": meta["bev_bce"],
                "train_dice": meta["bev_dice"],
            },
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        outputs = self.model(inputs)
        dist_loss, angle_between_loss, bev_loss, meta = self.loss_function(
            outputs, labels
        )
        loss = dist_loss + angle_between_loss + bev_loss

        fov = 120
        min_overlap = 10
        overlap_threshold = torch.deg2rad(torch.tensor(fov - min_overlap))
        visible_gt = meta["angle_between_nodes_gt"].abs() < overlap_threshold.item()

        iou = self.iou(outputs[0][1].sigmoid(), labels["topdowns_agents"])

        metrics = {
            "val_loss": meta["dist_metric"].mean() + meta["angle_between_m"].mean(),
            "val_dist_loss": meta["dist_metric"].mean(),
            "val_angle_between_loss": meta["angle_between_m"].mean(),
            "val_dist_pred_var": meta["dist_pred_var"].mean(),
            "val_angle_between_nodes_pred_var": meta[
                "angle_between_nodes_pred_var"
            ].mean(),
            "val_bce": meta["bev_bce"],
            "val_dice": meta["bev_dice"],
            "val_iou": iou,
        }
        self.log_dict(metrics, sync_dist=True)

        if (
            False
        ):  # Set to true if you want to save and load on validation end. Also update the exported model id below.
            if dataloader_idx not in self.eval_preds.keys():
                self.eval_preds[dataloader_idx] = {"e": [], "n": []}
            edge_preds, node_preds = self.export_preds(inputs, outputs, meta)
            self.eval_preds[dataloader_idx]["e"] += edge_preds
            self.eval_preds[dataloader_idx]["n"].append(node_preds)

    def on_validation_epoch_end(self):
        for d_idx, eval_preds in self.eval_preds.items():
            model_id = "checkpoint_epoch"  # Update for export
            df_edge = pd.concat([ep for ep in eval_preds["e"]])
            df_edge.to_pickle(f"eval_camera_ready/eval_e_{model_id}_{d_idx}.pkl")
            node_preds = torch.cat([ep for ep in eval_preds["n"]], dim=0)
            torch.save(node_preds, f"eval_camera_ready/eval_n_{model_id}_{d_idx}.pkl")

        test_dataloader = self.trainer.datamodule.test_dataloader()
        render_idx = 0
        device = next(self.parameters()).device
        for dataloader_idx, test_dataloader in enumerate(
            self.trainer.datamodule.test_dataloader()
        ):
            for inputs, labels in test_dataloader:
                test_input_dev_model = {}
                for k, v in inputs.items():
                    test_input_dev_model[k] = v.to(device) if torch.is_tensor(v) else v
                (edge_preds, node_preds), (edge_index, edge_batch) = self.model(
                    test_input_dev_model
                )

                cpu = torch.device("cpu")
                renderings = render_batch(
                    to_device(inputs, cpu),
                    to_device(labels, cpu),
                    to_device(edge_index, cpu),
                    to_device(edge_batch, cpu),
                    to_device(edge_preds, cpu),
                    to_device(node_preds, cpu),
                )

                if self.logger is not None:
                    self.logger.log_image(
                        key=f"render_{dataloader_idx}",
                        images=renderings,
                    )
                    for img in renderings:
                        try:
                            os.remove(img._path)
                        except FileNotFoundError:
                            pass
                else:
                    Path("./train/render").mkdir(parents=False, exist_ok=True)
                    for img in renderings:
                        shutil.move(
                            img._path,
                            f"./train/render/render_{dataloader_idx}_{render_idx:04d}.png",
                        )
                        render_idx += 1
                break

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            betas=self.betas,
            lr=self.initial_learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_learning_rate,
            steps_per_epoch=int(
                len(self.trainer.datamodule.train_dataloader())
                / torch.cuda.device_count()
            ),
            pct_start=self.warmup_epochs / self.num_epochs,
            epochs=self.num_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--resume_run_id", default="", type=str, help="W&B run ID to resume from"
        )
        parser.link_arguments(
            "resume_run_id",
            "trainer.logger.init_args.resume",
            compute_fn=lambda x: "must" if x else "never",
        )

    def before_instantiate_classes(self):
        subcommand = self.config.subcommand
        c = self.config[subcommand]
        run_id = None

        seed_everything(c.model.seed, workers=True)

        if not c.trainer.logger:
            return

        if c.resume_run_id:
            run_id = c.resume_run_id
            api = wandb.Api()
            c.ckpt_path = None
            for callback in c.trainer.callbacks:
                if callback.class_path == "lightning.pytorch.callbacks.ModelCheckpoint":
                    c.ckpt_path = str(
                        Path(callback.init_args.dirpath) / run_id / "last.ckpt"
                    )
            assert c.ckpt_path is not None

        else:
            run_id = wandb.util.generate_id()

        # also make sure that ModelCheckpoints go to the right place
        c.trainer.logger.init_args.id = run_id
        for callback in c.trainer.callbacks:
            if callback.class_path == "lightning.pytorch.callbacks.ModelCheckpoint":
                callback.init_args.dirpath = str(
                    Path(callback.init_args.dirpath) / run_id
                )


def main_cli():
    torch.set_float32_matmul_precision("high")
    MyLightningCLI(TrainerModuleLocalize, RelPosDataModule, save_config_callback=None)


if __name__ == "__main__":
    main_cli()
