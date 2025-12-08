from quadruped_mjx_rl.domain_randomization.types import (
    DomainRandomizationFn,
    TerrainMapRandomizationFn,
)
from quadruped_mjx_rl.domain_randomization.debug_randomizer import randomize_minimal
from quadruped_mjx_rl.domain_randomization.randomized_physics import (
    domain_randomize as randomize_physics,
)
from quadruped_mjx_rl.domain_randomization.randomized_obstacles import (
    terrain_randomize as randomize_obstacles,
)
from quadruped_mjx_rl.domain_randomization.randomized_tiles import (
    randomize_tiles,
    color_meaning_fn,
)
