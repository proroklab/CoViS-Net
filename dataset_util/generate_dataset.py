import shutil
from tqdm import tqdm
import imageio
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import habitat_sim
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageEnhance


def make_simple_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    if settings["color_sensor"]:
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.position = [0.0, 0.0, 0.0]
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        color_sensor_spec.hfov = settings["fov"]

        agent_cfg.sensor_specifications.append(color_sensor_spec)

    if settings["depth_sensor"]:
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, 0.0, 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        depth_sensor_spec.hfov = settings["fov"]

        agent_cfg.sensor_specifications.append(depth_sensor_spec)

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def sample_random_points(
    sim, max_samples=float("inf"), volume_sample_fac=1.0, significance_threshold=0.2
):
    scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    scene_volume = scene_bb.size().product()
    n_samples = min(int(scene_volume * volume_sample_fac), max_samples)
    points = np.array(
        [sim.pathfinder.get_random_navigable_point() for _ in range(n_samples)]
    )

    hist, bin_edges = np.histogram(points[:, 1], bins="auto")
    significant_bins = (hist / len(points)) > significance_threshold
    l_bin_edges = bin_edges[:-1][significant_bins]
    r_bin_edges = bin_edges[1:][significant_bins]
    points_floors = {}
    for l_edge, r_edge in zip(l_bin_edges, r_bin_edges):
        points_floor = points[(points[:, 1] >= l_edge) & (points[:, 1] <= r_edge)]
        height = points_floor[:, 1].mean()
        points_floors[height] = points_floor
    return points_floors


def get_obs_from_random_pose(sim, agent, cfg, pos, yaw):
    agent_state = agent.get_state()
    pos_offset = np.random.uniform(low=cfg["pos_range"][0], high=cfg["pos_range"][1])
    rot = np.random.uniform(
        low=cfg["rot_range"][0], high=cfg["rot_range"][1]
    )  # x pitch, y yaw, z roll
    agent_state.position = pos + pos_offset
    agent_state.rotation = R.from_euler(
        "xyz", rot + np.array([0.0, yaw, 0.0])
    ).as_quat()
    agent.set_state(agent_state)
    return sim.get_sensor_observations(), agent_state


def create_meta_and_save(sim, agent, sim_cfg, obs, topdown_idx, img_idx, out_path):
    if "depth_sensor" in obs:
        depth_image = Image.fromarray(100 * obs["depth_sensor"], mode="RGB")
        depth_image.save(out_path / "depth" / f"{img_idx:05d}.jpg")

    if "color_sensor" in obs:
        rgb_image = Image.fromarray(obs["color_sensor"], mode="RGBA")
        rgb_image.convert("RGB").save(out_path / "rgb" / f"{img_idx:05d}.jpg")

    agent_state = agent.get_state()
    bounds = sim.pathfinder.get_bounds()
    return pd.DataFrame(
        {
            "walk_id": img_idx,
            "image_id": img_idx,
            "topdown_id": topdown_idx,
            "pos_x": agent_state.position[0],
            "pos_y": agent_state.position[1],
            "pos_z": agent_state.position[2],
            "quat_w": agent_state.rotation.w,
            "quat_x": agent_state.rotation.x,
            "quat_y": agent_state.rotation.y,
            "quat_z": agent_state.rotation.z,
            "cam_fov": sim_cfg["fov"],
            "pixel_pos_x": (
                (agent_state.position[0] - bounds[0][0]) / sim_cfg["meters_per_pixel"]
            ).astype(np.int32),
            "pixel_pos_y": (
                (agent_state.position[2] - bounds[0][2]) / sim_cfg["meters_per_pixel"]
            ).astype(np.int32),
        },
        index=[img_idx],
    )


def save_topdown(sim, height, meters_per_pixel, filename):
    topdown = sim.pathfinder.get_topdown_view(
        meters_per_pixel=meters_per_pixel, height=height
    ).astype(np.uint8)
    td_img = Image.fromarray(255 * topdown, mode="L")
    td_img.convert("RGB").save(filename)


def generate(scene_idx, store_depth, hm3d_data_path, dataset_out_path):
    tmp_path = Path("/tmp/covisnet_render")
    for scene in hm3d_data_path.glob(f"{scene_idx:05d}-*/*.basis.glb"):
        shutil.rmtree(tmp_path, ignore_errors=True)

        (tmp_path / "rgb").mkdir(parents=True, exist_ok=True)
        if store_depth:
            (tmp_path / "depth").mkdir(parents=True, exist_ok=True)

        out_path = dataset_out_path / scene.parent.stem
        out_path.mkdir(parents=True, exist_ok=True)

        sim_settings = {
            "color_sensor": True,
            "depth_sensor": store_depth,
            "scene": str(scene),  # Scene path
            "default_agent": 0,  # Index of the default agent
            "width": 448,  # Spatial resolution of the observations
            "height": 448,
            # "fov": 48,
            "fov": 120,
            "meters_per_pixel": 0.1,
        }

        sampling_cfg = {
            "pos_range": (
                [-0.25, 0.1, -0.25],
                [0.25, 1.9, 0.25],
            ),  # top/right, up/down, top/right
            "rot_range": (
                [-np.pi / 4, -np.pi / 4, -np.pi / 4],
                [np.pi / 4, np.pi / 4, np.pi / 4],
            ),  # pitch up/down, yaw left/right (?), tilt l/r,
            #'rot_range': ([0.0, 0.0, 0.0], [0.0, math.pi * 2, 0.0]), # pitch up/down, yaw left/right (?), tilt l/r,
            "samples_around_point": 4,
            "max_samples_per_scene": 20000,
        }

        cfg = make_simple_cfg(sim_settings)
        sim = habitat_sim.Simulator(cfg)
        sim.pathfinder.seed(scene_idx)
        np.random.seed(scene_idx)
        agent = sim.initialize_agent(sim_settings["default_agent"])
        heights_points = sample_random_points(
            sim, max_samples=sampling_cfg["max_samples_per_scene"]
        )
        total = (
            sum([len(p) for p in heights_points.values()])
            * sampling_cfg["samples_around_point"]
        )
        img_idx = 0
        metas = []
        with tqdm(total=total) as pbar:
            for floor_idx, (height, points) in enumerate(heights_points.items()):
                save_topdown(
                    sim,
                    height,
                    sim_settings["meters_per_pixel"],
                    out_path / f"topdown_{floor_idx}.png",
                )
                for point in points:
                    for yaw in np.linspace(
                        0, np.pi * 2, sampling_cfg["samples_around_point"] + 1
                    )[:-1]:
                        obs, agent_state = get_obs_from_random_pose(
                            sim, agent, sampling_cfg, point, yaw
                        )

                        meta = create_meta_and_save(
                            sim, agent, sim_settings, obs, floor_idx, img_idx, tmp_path
                        )
                        metas.append(meta)

                        img_idx += 1
                        pbar.update(1)

        pd.concat(metas).to_csv(out_path / "data.csv", index_label="index")

        shutil.make_archive(out_path / "rgb", "zip", tmp_path / "rgb")
        if store_depth:
            shutil.make_archive(out_path / "depth", "zip", tmp_path / "depth")

        sim.close()

    shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_idx", type=int)
    parser.add_argument("hm3d_data_path", type=str)
    parser.add_argument("dataset_out", type=str)
    parser.add_argument("--depth", action="store_true")
    args = parser.parse_args()

    generate(
        args.scene_idx, args.depth, Path(args.hm3d_data_path), Path(args.dataset_out)
    )
