"""Training entrypoint for RBP/Anderson recurrent MeshNet."""

import hydra
from omegaconf import DictConfig

import curriculum_training as base
from rbp_meshnet import enRBPMesh, enRBPMesh_checkpoint


base.enDynamicMesh = enRBPMesh
base.enDynamicMesh_checkpoint = enRBPMesh_checkpoint


@hydra.main(config_path="conf", config_name="refine", version_base=None)
def main(cfg: DictConfig):
    # Reuse the refiner switch to route through the experimental model path.
    cfg.model.use_refiner = True
    return base.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
