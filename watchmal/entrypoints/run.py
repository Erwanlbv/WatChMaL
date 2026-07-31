"""
Single worker entry point for the unified WatChMaL core.

This replaces the former per-core workers (run_watchmal.py + run_caverns.py). There is no
longer a watchmal/caverns distinction: one worker builds the model, instantiates the
engine and runs its tasks for every model family (CNN / image, graph / GNN, multi-ring
segmentation). Families differ only inside their engine (train loop, dataset pipeline),
never here.

Uniform engine contract served here:
  - build_model wraps the model in DDP (config-driven find_unused_parameters)
  - engine ctor: (target_key via config, model, rank, device, dump_path, wandb_run, dataset)
  - dump_path is <run_dir>/outputs/ (created here) so every run is analysis-readable
  - configure_amp is called before any task (a no-op unless the engine/loop uses AMP)
  - setup_data_loaders(data, loaders, is_distributed, seed) is the one data entry point
  - a barrier is raised between tasks so train -> restore_best_state -> evaluate is safe
    under DDP
  - MASTER_PORT (optional, defaults to DEFAULT_MASTER_PORT) is offset by the first GPU
    index (collision avoidance on shared nodes)
"""

# hydra imports
from hydra.utils import instantiate
from hydra.core.utils import configure_log

from omegaconf import open_dict

# torch imports
import torch
from torch.distributed import destroy_process_group

# generic imports
import os
import time

# watchmal imports
from watchmal.utils.banner import loading_banner
from watchmal.utils.determinism import configure_determinism
from watchmal.utils.logging_utils import setup_logging
from watchmal.utils.build_utils import build_model
from watchmal.utils.distributed_utils import ddp_setup, restrict_logging_to_rank0

log = setup_logging(__name__)
sleep_time = 5

# Rendez-vous port used when a config does not set MASTER_PORT. The actual port is this
# value offset by the first GPU index (see below).
DEFAULT_MASTER_PORT = 12355


def _engine_label(hydra_config) -> str:
    """Short, human-readable engine name for the banner, e.g. 'graph/classification'.

    Derived from the engine's own `_target_` so it follows any renaming, rather than
    being a second copy of the family list.
    """
    target = str(hydra_config.engine.get("_target_", "")) if "engine" in hydra_config else ""
    parts = [p for p in target.split(".") if p]
    if len(parts) > 3 and parts[0] == "watchmal" and parts[1] == "engine":
        return "/".join(parts[2:-1])
    return parts[-1] if parts else "engine"


def _device_label(gpu_list) -> str:
    """Every device the run uses, e.g. 'cuda:0,1' or 'cuda:2,4,5' - not just this rank's.

    The banner is drawn by rank 0, so showing `self.device` there would read as
    "cuda:0" on an 8-GPU job: technically true of the process doing the drawing, and
    misleading about the run.
    """
    if not gpu_list:
        return "cpu"
    return "cuda:" + ",".join(str(int(gpu)) for gpu in gpu_list)


def run(rank, gpu_list, dataset, wandb_run, hydra_config, global_hydra_config):

    ngpus = len(gpu_list)
    is_distributed = ngpus > 1

    if rank == 0:
        for k in list(os.environ.keys()):
            if k.startswith('WANDB_'):
                log.info(f"wandb env var {k}: {os.getenv(k)}")

    # ---- determinism ---- #
    # Before the device is touched: CUBLAS_WORKSPACE_CONFIG has to be set before the
    # CUDA context exists. A no-op unless `deterministic: True` is in the config.
    configure_determinism(
        enabled=hydra_config.get("deterministic", False),
        seed=hydra_config.get("seed", None),
        warn_only=hydra_config.get("deterministic_warn_only", False),
        rank=rank,
    )

    # ---- device ---- #
    if ngpus == 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{int(gpu_list[rank])}")
        # set_device up front so libraries that allocate on the current device (not the
        # device of the input tensors) bind the right GPU under DDP.
        torch.cuda.set_device(device)

    # ---- distributed init ---- #
    if is_distributed:
        # Offset the port by the first GPU index to avoid collisions between concurrent
        # jobs sharing a node. MASTER_PORT is optional: a config that omits it (every
        # config in the watchmal tree does, and some ship gpu_list with 2+ entries) gets
        # the default rather than an AttributeError at rendez-vous time.
        master_port = int(hydra_config.get("MASTER_PORT", DEFAULT_MASTER_PORT)) + int(gpu_list[0])
        ddp_setup(rank, world_size=ngpus, master_port=master_port, device=device)
        configure_log(global_hydra_config.job_logging, global_hydra_config.verbose)

    wandb_run = wandb_run if rank == 0 else None
    log.info(f"Running worker {rank} on device : {device} with wandb_run : {wandb_run}")

    # Each worker announced itself above; from here on keep INFO/DEBUG on rank 0 only so
    # the shared start-up logs are not duplicated per process.
    restrict_logging_to_rank0(rank)

    # One seed for the whole run, set once. The engine captures it at construction
    # (self.seed = torch.initial_seed()); every rank holds the same value, so sampler
    # ordering and any train/val split agree across ranks.
    torch.manual_seed(hydra_config.seed)

    # ---- model ---- #
    model, nb_params = build_model(
        model_config=hydra_config.model,
        device=device,
        use_ddp=is_distributed,
        find_unused_parameters=hydra_config.get("find_unused_parameters", False),
    )
    if wandb_run is not None:
        wandb_run.log({'nb_params': nb_params})

    # ---- output dir (analysis-compatible location) ---- #
    dump_path = hydra_config.get("dump_path", None) or "./outputs/"
    if not str(dump_path).endswith("/"):
        dump_path = str(dump_path) + "/"
    os.makedirs(dump_path, exist_ok=True)

    # ---- engine + per-task configuration ---- #
    # Everything below is the slow part of a start-up: for several families the engine
    # constructor builds the dataset, and setup_data_loaders reads the split files. The
    # banner animates on rank 0 for exactly that stretch, with the terminal split so
    # the run's own log output keeps scrolling above it. Its status line says what is
    # being waited on; it is drawn only, never logged, so nothing reaches main.log or
    # wandb. On other ranks, and whenever stdout is not a TTY, every call is a no-op.
    with loading_banner(
        engine=_engine_label(hydra_config),
        device=_device_label(gpu_list),
        params=nb_params,
        enabled=(rank == 0),
    ) as banner:

        banner.set_status("instantiating engine")
        engine = instantiate(
            config=hydra_config.engine,
            model=model,
            rank=rank,
            device=device,
            dump_path=dump_path,
            wandb_run=wandb_run,
            dataset=dataset,
        )

        # Stop point 1 (the normal case): the banner ends when the first batch has been
        # through the model, not when configuration finishes. Every family's train loop
        # reports its first step through the tracker, so this needs no per-family code -
        # and it means the banner keeps running through the parts of train() that happen
        # before any compute (loader iteration, first-epoch setup, cuDNN autotune).
        engine.tracker.on_first_step = banner.stop

        banner.set_status("configuring precision")
        engine.configure_amp(hydra_config.get("amp", False))

        for task, task_config in hydra_config.tasks.items():

            with open_dict(task_config):

                if 'data_loaders' in task_config:
                    banner.set_status(f"building data loaders for '{task}'")
                    engine.setup_data_loaders(
                        hydra_config.data,
                        task_config.pop("data_loaders"),
                        is_distributed,
                        hydra_config.seed,
                    )

                if 'optimizers' in task_config:
                    banner.set_status(f"configuring optimizer for '{task}'")
                    engine.configure_optimizers(task_config.pop("optimizers"))

                if 'scheduler' in task_config:
                    banner.set_status(f"configuring scheduler for '{task}'")
                    engine.configure_scheduler(task_config.pop("scheduler"))

                if 'loss' in task_config:
                    banner.set_status(f"configuring loss for '{task}'")
                    engine.configure_loss(task_config.pop("loss"))

                if 'early_stopping' in task_config:
                    banner.set_status(f"configuring early stopping for '{task}'")
                    engine.configure_early_stopping(task_config.pop('early_stopping'))

        # ---- run tasks ---- #
        # Inside the banner's `with`, so it also covers the inter-task barrier - which
        # on a shared node is a real wait when the ranks arrive unevenly.
        for task, task_config in hydra_config.tasks.items():
            if is_distributed:
                banner.set_status(f"waiting for all ranks before '{task}'")
                torch.distributed.barrier()

            # Stop point 2: only `train` produces training steps, so any other task
            # would leave the banner animating forever - end it before dispatching one.
            # (An evaluate-only or multi-ring test-only run never reaches stop point 1.)
            if task == "train":
                banner.set_status("training")
            else:
                banner.stop()

            getattr(engine, task)(**task_config)

    # Stop point 3, the backstop: leaving the `with` stops the banner if neither of the
    # above fired. banner.stop() is idempotent, so the three points cannot conflict.

    # ---- teardown ---- #
    if (rank == 0) and (wandb_run is not None):  # close W&B first
        run_id = wandb_run.id
        log.info(f"run id : {run_id}")
        log.info(f"Calling wandb.finish()")
        time.sleep(sleep_time)
        wandb_run.finish()
        time.sleep(sleep_time)
        log.info(f"Done")

    if is_distributed:  # then tear down DDP
        log.info(f"Calling destroy_process_group()")
        destroy_process_group()
        log.info(f"Finished.")
