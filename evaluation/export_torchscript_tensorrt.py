import torch
import torch_tensorrt
import time
import numpy as np
from pathlib import Path


out_base_dir = Path("./exported_models")
# model_in_file = Path("artifacts/model-kpqv7omx:v6/model.ckpt")
# model_in_file = Path("models_bev_pose/cklmxq61/epoch=12-step=155025-validation_loss=0.000.ckpt")
# model_in_file = Path("models_bev_pose/ugz0kinn/epoch=11-step=143100-v1.ckpt")
model_prefix = "0kc5po4ee18"  # model_in_file.parent.stem.replace(":", "-")
dtype = torch.half
# dtype = torch.float
type_prefix = str(dtype).split(".")[1]
dev_str = "cuda"
dev = torch.device(dev_str)

bs = 1
n_agents = 5

data = {
    "pos": torch.rand(bs, n_agents, 3, device=dev) * 8,
    "rot": torch.rand(bs, n_agents, 3, device=dev),
    "img_norm": torch.rand(bs, n_agents, 3, 224, 224, device=dev),
}

# model_prefix = model_in_file.parent.stem.replace(":", "-")
f_enc = torch.jit.load(
    out_base_dir / f"{model_prefix}_{type_prefix}_jit_{dev_str}_enc.ts"
)
f_msg = torch.jit.load(
    out_base_dir / f"{model_prefix}_{type_prefix}_jit_{dev_str}_msg.ts"
)
f_post = torch.jit.load(out_base_dir / f"{model_prefix}_float32_jit_{dev_str}_post.ts")
f_bev = torch.jit.load(
    out_base_dir / f"{model_prefix}_{type_prefix}_jit_{dev_str}_bev.ts"
)
f_bevdec = torch.jit.load(
    out_base_dir / f"{model_prefix}_{type_prefix}_jit_{dev_str}_bevdec.ts"
)


def model_decentr_forward(data, edge_index, f_enc, f_msg, f_post, dtype=torch.float):
    out = []
    img_flat = data["img_norm"].flatten(0, 1).to(dtype)
    for i, j in zip(edge_index[1], edge_index[0]):
        img_i = img_flat[i].unsqueeze(0)
        img_j = img_flat[j].unsqueeze(0)
        x_i = f_enc.to(dtype)(img_i)
        x_j = f_enc.to(dtype)(img_j)
        # aggr = f_msg.to(dtype)(x_i.to(dtype), x_j.to(dtype)).sum(dim=1)
        # out.append(f_post.to(dtype)(aggr.to(dtype)))
        out.append(f_msg.to(dtype)(x_i.to(dtype), x_j.to(dtype)))
    return torch.concatenate(out, dim=0)


def eval_outputs(a, b, dtype=torch.float):
    err = a.to(dtype) - b.to(dtype)
    serr = err**2
    abserr = err.abs()
    mse = serr.mean()
    mae = abserr.mean()
    print("mse", mse.item(), "mae", mae.item())
    return mse.item()


def model_trt(data, f_enc, f_msg, f_post, f_bev, f_bevdec, dtype=torch.float):
    out = []
    img_flat = data["img_norm"].to(dtype).flatten(0, 1)
    i, j = 0, 0
    img_i = img_flat[i].unsqueeze(0)
    img_j = img_flat[j].unsqueeze(0)
    x_i = f_enc.to(dtype)(img_i)
    x_j = f_enc.to(dtype)(img_j)
    aggr = f_msg.to(dtype)(x_i, x_j)
    bev = f_bev.to(dtype)(x_i, x_j, aggr)
    bev_out = f_bevdec.to(dtype)(bev)
    f_enc_trt = torch_tensorrt.compile(
        f_enc,
        inputs=[torch_tensorrt.Input(shape=img_i.shape, dtype=dtype)],
        enabled_precisions={dtype},
        truncate_long_and_double=True,
    )
    f_msg_trt = torch_tensorrt.compile(
        f_msg,
        inputs=[
            torch_tensorrt.Input(shape=x_i.shape, dtype=dtype),
            torch_tensorrt.Input(shape=x_j.shape, dtype=dtype),
        ],
        enabled_precisions={dtype},
        truncate_long_and_double=True,
    )
    f_post_trt = torch_tensorrt.compile(
        f_post,
        inputs=[torch_tensorrt.Input(shape=aggr.shape, dtype=torch.float)],
        enabled_precisions={torch.float},
        truncate_long_and_double=True,
    )
    f_bev_trt = torch_tensorrt.compile(
        f_bev,
        inputs=[
            torch_tensorrt.Input(shape=x_i.shape, dtype=dtype),
            torch_tensorrt.Input(shape=x_j.shape, dtype=dtype),
            torch_tensorrt.Input(shape=aggr.shape, dtype=dtype),
        ],
        enabled_precisions={torch.float},
        truncate_long_and_double=True,
    )
    f_bevdec_trt = torch_tensorrt.compile(
        f_bevdec,
        inputs=[torch_tensorrt.Input(shape=bev.shape, dtype=dtype)],
        enabled_precisions={dtype},
        truncate_long_and_double=True,
    )

    return f_enc_trt, f_msg_trt, f_post_trt, f_bev_trt, f_bevdec_trt


# edge_index = tuple(torch.randint(n_agents, (2, 30)))
# model_out = model_decentr_forward(
#    data, edge_index, f_enc, f_msg, f_post, dtype=torch.float
# )


def convert(dtype=torch.float):
    f_enc_trt, f_msg_trt, f_post_trt, f_bev_trt, f_bevdec_trt = model_trt(
        data, f_enc, f_msg, f_post, f_bev, f_bevdec, dtype=dtype
    )

    # model_trt_out = model_decentr_forward(
    #    data, edge_index, f_enc_trt, f_msg_trt, f_post_trt, dtype=dtype
    # )
    # mse = eval_outputs(model_trt_out, model_out, dtype=torch.float)
    # assert mse < 1e-5

    torch.jit.save(f_enc_trt, out_base_dir / f"{model_prefix}_{type_prefix}_trt_enc.ts")
    torch.jit.save(f_msg_trt, out_base_dir / f"{model_prefix}_{type_prefix}_trt_msg.ts")
    torch.jit.save(f_post_trt, out_base_dir / f"{model_prefix}_float32_trt_post.ts")
    torch.jit.save(f_bev_trt, out_base_dir / f"{model_prefix}_{type_prefix}_trt_bev.ts")
    torch.jit.save(
        f_bevdec_trt, out_base_dir / f"{model_prefix}_{type_prefix}_trt_bevdec.ts"
    )


# convert(torch.float)
convert(dtype)
