"""Training entrypoint for the one-shot residual refiner variant."""

import hydra
from omegaconf import DictConfig

import curriculum_training as base
from oneshot_refiner import enOneShotDynamicMesh, enOneShotDynamicMesh_checkpoint


base.enDynamicMesh = enOneShotDynamicMesh
base.enDynamicMesh_checkpoint = enOneShotDynamicMesh_checkpoint


@hydra.main(config_path="conf", config_name="refine", version_base=None)
def main(cfg: DictConfig):
    return base.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
