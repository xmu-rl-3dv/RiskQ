from functools import partial
from .macf import MultiAgentCar
from smac.env import MultiAgentEnv, StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper


import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2wrapped"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper) #smacv2
REGISTRY["macf"] = partial(env_fn, env=MultiAgentCar)




if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
