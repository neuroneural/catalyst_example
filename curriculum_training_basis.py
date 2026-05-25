"""Training entrypoint for basis-kernel adapter MeshNet."""

import hydra
from omegaconf import DictConfig

import curriculum_training as base
from basis_refiner import enBasisDynamicMesh, enBasisDynamicMesh_checkpoint


base.enDynamicMesh = enBasisDynamicMesh
base.enDynamicMesh_checkpoint = enBasisDynamicMesh_checkpoint


@hydra.main(config_path="conf", config_name="basis", version_base=None)
def main(cfg: DictConfig):
    return base.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()
