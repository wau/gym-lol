import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='League-v0',
    entry_point='gym_lol.envs:LeagueEnv',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='LeagueRemote-v0',
    entry_point='gym_lol.envs:LeageEnvRemote',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic=True,
)