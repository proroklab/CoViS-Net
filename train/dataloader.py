from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from typing import List, Optional
from scipy.spatial.transform import Rotation
import warnings
from torchvision import transforms
from torchvision.transforms import functional as F
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from io import BytesIO
from .parallelzipfile import ParallelZipFile as ZipFile
import scipy
import numpy as np
import json
import hashlib
import pickle


def load_topdowns(data_path: Path):
    topdowns = {}
    try:
        with open(data_path / "meta.json", "r") as json_file:
            meta_csvs = [Path(p) for p in json.load(json_file)]
    except FileNotFoundError:
        meta_csvs = sorted(list(data_path.glob("*/data.csv")))

    for meta_path in meta_csvs:
        id = meta_path.parent.stem
        topdowns[id] = {}
        for topdown_path in sorted((data_path / id).glob("topdown_*.png")):
            topdown_img = Image.open(topdown_path).convert("L")
            topdowns[id][int(topdown_path.stem.split("_")[1])] = topdown_img
    return topdowns


def load_metas(data_path: Path):
    metas = []
    try:
        with open(data_path / "meta.json", "r") as json_file:
            meta_csvs = [Path(p) for p in json.load(json_file)]
    except FileNotFoundError:
        meta_csvs = sorted(list(data_path.glob("*/data.csv")))

    for meta_path in meta_csvs:
        meta = pd.read_csv(meta_path)
        id = meta_path.parent.stem
        meta["dataset_id"] = id
        metas.append(meta)
    return pd.concat(metas, ignore_index=True)


class RandomCenterCrop(torch.nn.Module):
    def __init__(self, max_p=0.5):
        super().__init__()
        self.max_p = max_p

    def forward(self, img):
        img_h, _ = F.get_image_size(img)
        crop_size = int(img_h * self.max_p * torch.rand(1))
        return F.center_crop(img, img_h - crop_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RelPosDataset(Dataset):
    def __init__(
        self,
        data_path_str: str,
        splits: List[float] = [0.0, 1.0],
        nodes_per_sample: int = 6,
        data_frac: float = 0.05,
        n_samples: int = 10,
        distance_upper_bound: float = 15.0,
        quat_diff_max: Optional[float] = 0.1,
        input_resolution: int = 224,
        **kwargs,
    ):
        self.data_path = Path(data_path_str)
        self.nodes_per_sample = nodes_per_sample
        self.d_left_right = 3.0
        self.d_back = 3.0
        self.d_front = 3.0
        self.inp_img_crop = 336
        self.inp_resolution = input_resolution

        self.topdowns = load_topdowns(self.data_path)

        meta = load_metas(self.data_path)
        meta_dataset = meta[int(len(meta) * splits[0]) : int(len(meta) * splits[1])]

        self.samples = None

        cfg = {
            "data_frac": data_frac,
            "n_samples": n_samples,
            "distance_upper_bound": distance_upper_bound,
            "nodes_per_sample": nodes_per_sample,
            "quat_diff_max": quat_diff_max,
            "dataset_ids": sorted(list(meta_dataset["dataset_id"].unique())),
        }
        cache_dir = self.data_path / "sample_cache"
        cache_dir.mkdir(exist_ok=True)
        cfg_hash = self.hash_dict(cfg)

        for hashed_sample_path in cache_dir.glob("*.pkl"):
            if hashed_sample_path.stem == cfg_hash:
                with open(hashed_sample_path, "rb") as file:
                    self.samples, _ = pickle.load(file)
                print("Loaded cached dataset samples")

        if self.samples is None:
            print("No cached samples found, generating...")
            self.samples = self.generate_samples(cfg, meta)
            with open(cache_dir / f"{cfg_hash}.pkl", "wb") as file:
                pickle.dump((self.samples, cfg), file)

        self.transforms = [
            transforms.ColorJitter(
                brightness=(0.75, 1.25), hue=0.05, contrast=0.05, saturation=0.05
            ),
            transforms.GaussianBlur(7, sigma=(5.0, 10.0)),
            # RandomCenterCrop(max_p=0.5),
        ]

        self.rgb_archives = {}
        for dataset_id in self.topdowns.keys():
            archive = ZipFile(str(self.data_path / dataset_id / "rgb.zip"))
            self.rgb_archives[dataset_id] = archive

    def hash_dict(self, dictionary):
        serialized_dict = json.dumps(dictionary, sort_keys=True).encode("utf-8")
        hash_object = hashlib.sha256(serialized_dict)
        return hash_object.hexdigest()

    def generate_samples(self, cfg, meta):
        samples = []
        for i, dataset_id in enumerate(cfg["dataset_ids"]):
            meta_id = meta[meta["dataset_id"] == dataset_id]
            for meta_index, meta_row in meta_id.sample(
                frac=cfg["data_frac"], random_state=i * cfg["n_samples"]
            ).iterrows():
                dist = (
                    meta_id[["pos_x", "pos_z"]].values
                    - meta_row[["pos_x", "pos_z"]].values
                )
                norm = np.linalg.norm(dist.astype(float), axis=1)
                meta_query_raw = meta_id[
                    (norm < cfg["distance_upper_bound"])
                    & (meta_id.topdown_id == meta_row.topdown_id)
                ]

                if cfg["quat_diff_max"] is not None:
                    queried_quats = meta_query_raw[
                        ["quat_x", "quat_y", "quat_z", "quat_w"]
                    ].values
                    rand_row = meta_query_raw.sample(random_state=i * cfg["n_samples"])
                    rand_quat = rand_row[
                        ["quat_x", "quat_y", "quat_z", "quat_w"]
                    ].values
                    quat_dist = 1.0 - (
                        (rand_quat.reshape(1, 4) * queried_quats).sum(axis=1) ** 2
                    )
                    meta_query = meta_query_raw.iloc[quat_dist < cfg["quat_diff_max"]]
                else:
                    meta_query = meta_query_raw

                if meta_query.shape[0] < cfg["nodes_per_sample"] * 2:
                    continue

                for j in range(cfg["n_samples"]):
                    sample = meta_query.sample(
                        cfg["nodes_per_sample"], random_state=i * cfg["n_samples"] + j
                    )
                    samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs = []
        imgs_normalized = []
        embeddings = []
        poss = []
        pixel_poss = []
        rots_rotvec = []
        rots_quat = []
        topdowns = []

        transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                *self.transforms,
                transforms.Resize(
                    self.inp_resolution,
                    antialias=True,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(self.inp_resolution),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
        norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        samples = self.samples[idx]

        for _, meta in samples.iterrows():
            rgb_path = f"rgb/{meta.image_id:05d}.jpg"
            img_path = self.data_path / meta.dataset_id / rgb_path
            if img_path.is_file():
                img = Image.open(img_path)
            else:
                tfile = self.rgb_archives[meta.dataset_id]
                data = tfile.read(Path(rgb_path).name)
                img = Image.open(BytesIO(data))

            img_transformed = transform(img)
            imgs.append(img_transformed)

            img_normalized = norm(img_transformed)
            imgs_normalized.append(img_normalized)

            pos = torch.tensor(
                meta[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float),
                dtype=torch.float32,
            )
            pixel_pos = torch.tensor(
                meta[["pixel_pos_x", "pixel_pos_y"]].to_numpy(dtype=float),
                dtype=torch.float32,
            )
            rot_quat = torch.tensor(
                meta[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy(dtype=float),
                dtype=torch.float32,
            )
            rot_obj = Rotation.from_quat(rot_quat)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Gimbal lock detected.")
                rot_euler = torch.tensor(rot_obj.as_rotvec(), dtype=torch.float32)

            topdown = self.topdowns[samples.iloc[0].dataset_id][meta.topdown_id]
            origin = torch.tensor(topdown.size) // 2
            shift = pixel_pos - origin
            topdown_shift = topdown.transform(
                topdown.size, Image.Transform.AFFINE, (1, 0, shift[0], 0, 1, shift[1])
            )
            topdown_rot = topdown_shift.rotate(-rot_euler[1] * 180 / torch.pi)

            topdown_clip = topdown_rot.crop(
                (
                    origin[0].item() - int(self.d_left_right / 0.1),
                    origin[1].item() - int(self.d_front / 0.1),
                    origin[0].item() + int(self.d_left_right / 0.1),
                    origin[1].item() + int(self.d_back / 0.1),
                )
            )
            topdown_tensor = transforms.functional.pil_to_tensor(topdown_clip)

            embedding = torch.Tensor()

            embeddings.append(embedding)
            poss.append(pos)
            pixel_poss.append(pixel_pos)
            rots_rotvec.append(rot_euler)
            rots_quat.append(rot_quat)
            topdowns.append(topdown_tensor)

        topdowns = torch.stack(topdowns, dim=0)
        poss = torch.stack(poss, dim=0)
        pixel_poss = torch.stack(pixel_poss, dim=0)
        rots_rotvec = torch.stack(rots_rotvec, dim=0)
        rots_quat = torch.stack(rots_quat, dim=0)

        imgs = torch.stack(imgs, dim=0)
        imgs_normalized = torch.stack(imgs_normalized, dim=0)
        embeddings = torch.stack(embeddings, dim=0)

        origin = torch.tensor(topdown.size) // 2
        pixel_pos_mean = pixel_poss.mean(dim=0).int()
        shift = pixel_pos_mean - origin
        topdown_shift = topdown.transform(
            topdown.size, Image.Transform.AFFINE, (1, 0, shift[0], 0, 1, shift[1])
        )

        origin_shift = torch.tensor(topdown_shift.size) // 2
        topdown_shift_view_size = int(4.0 / 0.1)
        topdown_shift_crop = topdown_shift.crop(
            (
                origin_shift[0].item() - topdown_shift_view_size,
                origin_shift[1].item() - topdown_shift_view_size,
                origin_shift[0].item() + topdown_shift_view_size,
                origin_shift[1].item() + topdown_shift_view_size,
            )
        )
        origin_shift_crop = torch.tensor(topdown_shift_crop.size) // 2
        pixel_poss_shift_crop = pixel_poss - pixel_pos_mean + origin_shift_crop

        topdown_tensor = transforms.functional.pil_to_tensor(topdown_shift_crop)

        data = {
            "idx": idx,
            "t": 0,
            "img_raw": imgs,
            "pos": poss,
            "rot": rots_rotvec,
            "rot_quat": rots_quat,
            "img_norm": imgs_normalized,
            "embeddings": embeddings,
        }
        labels = {
            "pos": poss,
            "rot": rots_rotvec,
            "rot_quat": rots_quat,
            "pixel_pos": pixel_poss_shift_crop,
            "topdown_global": topdown_tensor.squeeze(0) / 255.0,
            "topdowns_agents": topdowns / 255.0,
        }
        return data, labels


class RealRelPosDataset(RelPosDataset):
    def __init__(
        self,
        data_path_str: str,
        splits: List[float] = [0.0, 1.0],
        nodes_per_sample: int = 2,
        data_frac: float = 0.4,
        n_samples: int = 6,
        distance_upper_bound: float = 1.0,
        input_resolution: int = 224,
        quat_diff_max: float = 0.1,
        **kwargs,
    ):
        self.data_path = Path(data_path_str)
        self.nodes_per_sample = nodes_per_sample
        self.inp_resolution = input_resolution

        self.topdowns = {}

        converters = {
            "ranges": lambda x: [float(el) for el in x.strip("[]").split(", ")]
        }
        samples = []
        idx = 0
        for scene_path in sorted(self.data_path.glob("*")):
            scene_path = Path(scene_path)
            df_scene = pd.read_csv(
                scene_path / "sensor_2" / "meta.csv",
                converters=converters,
                index_col="index",
            )
            df_scene["file"] = df_scene["file"].apply(lambda x: str(scene_path / x))
            df_scene["scene"] = scene_path.name

            # Dataset D
            # Assume uniform distribution of relative positions with distances ranging from 0 to 4 meters with n bins
            # Choose bin with least number of samples and sample an initial random sample from D
            # Restrict the possible relative poses to the ones that are within a certain distance of the initial sample
            # Sample a random relative pose from the restricted set
            # Repeat the process until the desired number of samples is reached

            n_bins = 10
            bins = np.ones(n_bins)
            bin_ranges = np.linspace(0, 2, n_bins + 1)
            while bins.sum() < 5000:
                sample_0 = df_scene.sample(1).reset_index()
                df_sample = df_scene[df_scene.index != sample_0.iloc[0]["index"]]
                sub_samples = [sample_0]
                while len(sub_samples) < self.nodes_per_sample:
                    inverse_weights = 1 / (bins + 1e-3)
                    normalized_weights = inverse_weights / np.sum(inverse_weights)
                    bin_idx = np.random.choice(np.arange(n_bins), p=normalized_weights)
                    bin_range = bin_ranges[bin_idx : bin_idx + 2]
                    dists = np.linalg.norm(
                        df_sample[["x", "y"]].values - sample_0[["x", "y"]].values,
                        axis=1,
                    )
                    df_restricted = df_sample[
                        (dists > bin_range[0]) & (dists <= bin_range[1])
                    ]
                    if len(df_restricted) < 10:
                        continue

                    sample_i = df_restricted.sample(1).reset_index()
                    df_subs = pd.concat(sub_samples)
                    dists = np.linalg.norm(
                        (df_subs[["x", "y"]] - sample_i[["x", "y"]]).values, axis=1
                    )
                    if (dists < 0.05).any():
                        continue
                    sub_samples.append(sample_i)
                    bin_idxs = np.argmax(dists[:, None] < bin_ranges, axis=1) - 1
                    bins[bin_idxs] += 1
                sub_samples = pd.concat(sub_samples)
                sub_samples["idx"] = idx
                idx += 1
                samples.append(sub_samples)
            print(scene_path.name, bins)

        self.df = pd.concat(samples)

        env_resolution_m = 0.1
        p_to_pix = 1 / env_resolution_m
        env_size_m = [4.0, 4.0]
        self.env_size_pix = (np.array(env_size_m) * p_to_pix).astype(int)
        self.df["pixel_pos_x"] = (self.df["x"] * p_to_pix).astype(int) + int(
            self.env_size_pix[0] / 2
        )
        self.df["pixel_pos_y"] = (self.df["y"] * p_to_pix).astype(int) + int(
            self.env_size_pix[1] / 2
        )

        self.transforms = [
            # RandomCenterCrop(max_p=0.5),
        ]

    def __len__(self):
        return self.df.idx.max()

    def __getitem__(self, idx):
        transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                *self.transforms,
                transforms.Resize(
                    self.inp_resolution,
                    antialias=True,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(self.inp_resolution),
                transforms.ConvertImageDtype(torch.float),
            ]
        )
        norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        imgs_raw = []
        imgs_normalized = []
        poss = []
        poss_pixel = []
        rots_euler = []
        rots_quat = []
        files = []
        sample = self.df[self.df.idx == idx]

        for _, sensor in sample.iterrows():
            files.append(sensor.file)
            img = Image.open(sensor.file)
            img_transformed = transform(img)
            img_normalized = norm(img_transformed)
            pos = torch.tensor([sensor.x, 0.0, sensor.y])
            pos_pixel = torch.tensor([sensor.pixel_pos_x, sensor.pixel_pos_y])

            rot_euler = torch.tensor([0.0, sensor.theta, 0.0])
            rot_obj = Rotation.from_rotvec(rot_euler)
            rot_quat = torch.from_numpy(rot_obj.as_quat())

            rots_euler.append(rot_euler)
            rots_quat.append(rot_quat)
            poss.append(pos)
            poss_pixel.append(pos_pixel)
            imgs_raw.append(img_transformed)
            imgs_normalized.append(img_normalized)

        imgs_raw = torch.stack(imgs_raw, dim=0)
        imgs_normalized = torch.stack(imgs_normalized, dim=0)
        poss = torch.stack(poss, dim=0)
        poss_pixel = torch.stack(poss_pixel, dim=0)
        rots_quat = torch.stack(rots_quat, dim=0).to(torch.float32)
        rots_rotvec = torch.stack(rots_euler, dim=0).to(torch.float32)

        data = {
            "file": files,
            "idx": idx,
            "img_raw": imgs_raw,
            "pos": poss,
            "pixel_pos": poss_pixel,
            "rot": rots_rotvec,
            "rot_quat": rots_quat,
            "t": torch.tensor(sample.iloc[0].t_msg_sync),
            "img_norm": imgs_normalized,
        }
        labels = {
            "pos": poss,
            "pixel_pos": poss_pixel,
            "rot": rots_rotvec,
            "rot_quat": rots_quat,
            "lidar": sample.iloc[0].ranges,
            "topdown_global": torch.zeros(self.env_size_pix[1], self.env_size_pix[0]),
            "topdowns_agents": torch.ones(3, 1, 60, 60),
        }

        return data, labels


class RelPosDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "dataset",
        data_test_dir: str = "dataset",
        batch_size: int = 128,
        num_workers: int = 8,
        nodes_per_sample: int = 2,
        quat_diff_max: Optional[float] = None,
        distance_upper_bound: float = 1.0,
        input_resolution: int = 224,
        n_samples: int = 10,
        data_frac: float = 1.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.data_test_dir = data_test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nodes_per_sample = nodes_per_sample
        self.quat_diff_max = quat_diff_max
        self.distance_upper_bound = distance_upper_bound
        self.input_resolution = input_resolution
        self.n_samples = n_samples
        self.data_frac = data_frac

        self.save_hyperparameters()

    def setup(self, stage: str):
        if "real" in self.data_dir:
            self.dataset_eval = [
                RealRelPosDataset(
                    self.data_dir,
                    splits=[0.0, 1.0],
                    nodes_per_sample=3,
                    input_resolution=self.input_resolution,
                )
            ]
            self.dataset_train = self.dataset_eval[0]
            self.dataset_test = self.dataset_eval
        else:
            self.dataset_train = RelPosDataset(
                self.data_dir,
                splits=[0.0, 0.8],
                nodes_per_sample=self.nodes_per_sample,
                data_frac=self.data_frac,
                n_samples=self.n_samples,
                distance_upper_bound=self.distance_upper_bound,
                input_resolution=self.input_resolution,
                quat_diff_max=self.quat_diff_max,
            )
            self.dataset_eval = [
                RelPosDataset(
                    self.data_dir,
                    splits=[0.8, 0.99],
                    nodes_per_sample=self.nodes_per_sample,
                    data_frac=self.data_frac,
                    n_samples=self.n_samples,
                    distance_upper_bound=self.distance_upper_bound,
                    input_resolution=self.input_resolution,
                    quat_diff_max=self.quat_diff_max,
                )
            ]
            if self.data_test_dir is not None:
                self.dataset_eval.append(
                    RealRelPosDataset(
                        self.data_test_dir,
                        splits=[0.0, 1.0],
                        nodes_per_sample=3,
                        input_resolution=self.input_resolution,
                    )
                )
            self.dataset_test = [
                RelPosDataset(
                    self.data_dir,
                    splits=[0.99, 1.0],
                    nodes_per_sample=self.nodes_per_sample,
                    data_frac=self.data_frac,
                    n_samples=self.n_samples,
                    distance_upper_bound=self.distance_upper_bound,
                    input_resolution=self.input_resolution,
                    quat_diff_max=self.quat_diff_max,
                )
            ]
            if self.data_test_dir is not None:
                self.dataset_test.append(
                    RealRelPosDataset(
                        self.data_test_dir,
                        splits=[0.0, 1.0],
                        nodes_per_sample=3,
                        input_resolution=self.input_resolution,
                    )
                )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size * 2,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
            )
            for dataset in self.dataset_eval
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=8,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False,
            )
            for dataset in self.dataset_test
        ]


def visualize(dataset_path):
    import torch_geometric
    from .rendering import render_batch

    torch.manual_seed(1)

    batch_size = 2
    nodes_per_sample = 5
    datamodule = RelPosDataModule(
        dataset_path,
        None,  # "datasets/dataset_real_5_231024",
        nodes_per_sample=nodes_per_sample,
        num_workers=0,
        batch_size=batch_size,
        quat_diff_max=None,
        distance_upper_bound=1.0,
        input_resolution=224,
        n_samples=10,
        data_frac=0.01,
    )
    datamodule.setup("")
    data_batch, label_batch = next(iter(datamodule.train_dataloader()))

    pos = data_batch["pos"][:, :, :2].reshape(batch_size * nodes_per_sample, 2)
    edge_batch = torch.repeat_interleave(
        torch.arange(batch_size), nodes_per_sample, dim=0
    )

    edge_index = torch_geometric.nn.pool.radius_graph(
        pos, batch=edge_batch, r=4.0, loop=False
    )

    # fully connected minus self-loops
    n_d = batch_size * (nodes_per_sample**2 - nodes_per_sample)

    # Predictions should come from model, but in this case it's random test data
    edge_preds = {
        "pos": (torch.rand(n_d, 3) - 0.5) * 2.0,
        "rot": torch.ones(n_d, 4),
        "pos_var": torch.rand(n_d, 3) * 3,
        "rot_var": torch.rand(n_d, 1),
    }
    node_preds = np.ones_like(label_batch["topdowns_agents"])
    render_batch(
        data_batch,
        label_batch,
        edge_index,
        edge_batch,
        edge_preds,
        node_preds,
        show=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    args = parser.parse_args()

    visualize(args.dataset_path)
