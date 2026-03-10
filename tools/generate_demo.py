import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from omegaconf import OmegaConf
import hydra
import pathlib

from demo_generation.demogen import DemoGen


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('configs')),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    generator: DemoGen = cls(cfg)
    generator.generate_demo()


if __name__ == "__main__":
    main()