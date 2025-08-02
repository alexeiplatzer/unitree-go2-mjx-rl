from collections.abc import Callable

import random
import mujoco as mj
import numpy as np
from quadruped_mjx_rl.terrain_gen.noise_utils import perlin, edge_slope

ObstacleMaker = Callable[[mj.MjSpec, list[float], ...], None]


def flat(
    spec: mj.MjSpec | None = None, grid_loc: list[float] | None = None, name: str = "flat"
) -> None:
    SQUARE_LENGTH = 2
    BROWN = [0.460, 0.362, 0.216, 1.0]
    THICKNESS = 0.05

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    # Create tile
    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)
    body.add_geom(size=[SQUARE_LENGTH, SQUARE_LENGTH, THICKNESS], rgba=BROWN)


def stripes(
    spec: mj.MjSpec | None = None,
    grid_loc: list[float] | None = None,
    stripe_w: float = 0.10,  # total width of one stripe (m)
    amplitude: float = 0.05,  # extra height for the raised stripes (m)
    base_thickness: float = 0.05,  # thickness of the “flat” stripes (m)
    name: str = "stripes",
) -> None:
    """
    Generates a 2 m × 2 m tile whose surface is made of long, thin
    stripes running along the y-axis.  Every second stripe is raised
    by `amplitude`, creating a simple uneven terrain.
    """
    SQUARE_LENGTH = 2
    BROWN = [0.460, 0.362, 0.216, 1.0]

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)

    # Compute how many stripes we can fit across the 2 m span
    n_stripes = int((2 * SQUARE_LENGTH) / stripe_w)
    x_start = -SQUARE_LENGTH + stripe_w / 2  # centre of first stripe

    for k in range(n_stripes):
        is_high = k % 2 == 1  # alternate low/high
        height = base_thickness + (amplitude if is_high else 0.0)
        z_center = height / 2  # MuJoCo boxes are centred
        x_center = x_start + k * stripe_w

        body.add_geom(
            pos=[x_center, 0, z_center],
            size=[stripe_w / 2, SQUARE_LENGTH, height / 2],
            rgba=BROWN,
        )


def stairs(
    spec: mj.MjSpec | None = None,
    grid_loc: list[float] | None = None,
    num_stairs: int = 4,
    direction: int = 1,
    name: str = "stair",
) -> None:
    SQUARE_LENGTH = 2
    V_SIZE = 0.076
    H_SIZE = 0.12
    H_STEP = H_SIZE * 2
    V_STEP = V_SIZE * 2
    BROWN = [0.460, 0.362, 0.216, 1.0]

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)
    # Offset
    x_beginning, y_end = [-SQUARE_LENGTH + H_SIZE] * 2
    x_end, y_beginning = [SQUARE_LENGTH - H_SIZE] * 2
    # Dimension
    size_one = [H_SIZE, SQUARE_LENGTH, V_SIZE]
    size_two = [SQUARE_LENGTH, H_SIZE, V_SIZE]
    # Geoms positions
    x_pos_l = [x_beginning, 0, direction * V_SIZE]
    x_pos_r = [x_end, 0, direction * V_SIZE]
    y_pos_up = [0, y_beginning, direction * V_SIZE]
    y_pos_down = [0, y_end, direction * V_SIZE]

    for i in range(num_stairs):
        size_one[1] = SQUARE_LENGTH - H_STEP * i
        size_two[0] = SQUARE_LENGTH - H_STEP * i

        x_pos_l[2], x_pos_r[2], y_pos_up[2], y_pos_down[2] = [
            direction * (V_SIZE + V_STEP * i)
        ] * 4

        # Left side
        x_pos_l[0] = x_beginning + H_STEP * i
        body.add_geom(pos=x_pos_l, size=size_one, rgba=BROWN)
        # Right side
        x_pos_r[0] = x_end - H_STEP * i
        body.add_geom(pos=x_pos_r, size=size_one, rgba=BROWN)
        # Top
        y_pos_up[1] = y_beginning - H_STEP * i
        body.add_geom(pos=y_pos_up, size=size_two, rgba=BROWN)
        # Bottom
        y_pos_down[1] = y_end + H_STEP * i
        body.add_geom(pos=y_pos_down, size=size_two, rgba=BROWN)

    # Closing
    size = [SQUARE_LENGTH - H_STEP * num_stairs, SQUARE_LENGTH - H_STEP * num_stairs, V_SIZE]
    pos = [0, 0, direction * (V_SIZE + V_STEP * num_stairs)]
    body.add_geom(pos=pos, size=size, rgba=BROWN)


def debris_with_simple_geoms(
    spec: mj.MjSpec | None = None, grid_loc: list[float] | None = None, name: str = "plane"
) -> None:
    SQUARE_LENGTH = 2
    THICKNESS = 0.05
    BROWN = [0.460, 0.362, 0.216, 1.0]
    RED = [0.6, 0.12, 0.15, 1.0]

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    # Create tile
    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)
    body.add_geom(size=[SQUARE_LENGTH, SQUARE_LENGTH, THICKNESS], rgba=BROWN)

    # Simple Geoms
    x_beginning, y_end = [-SQUARE_LENGTH + THICKNESS] * 2
    x_end, y_beginning = [SQUARE_LENGTH - THICKNESS] * 2

    x_grid = np.linspace(x_beginning, x_end, 10)
    y_grid = np.linspace(y_beginning, y_end, 10)

    for i in range(10):
        x = np.random.choice(x_grid)
        y = np.random.choice(y_grid)

        pos = [grid_loc[0] + x, grid_loc[1] + y, 0.2]

        g_type = None
        size = None
        if random.randint(0, 1):
            g_type = mj.mjtGeom.mjGEOM_BOX
            size = [0.1, 0.1, 0.02]
        else:
            g_type = mj.mjtGeom.mjGEOM_CYLINDER
            size = [0.1, 0.02, 0]

        body = spec.worldbody.add_body(pos=pos, name=f"g{i}_{name}", mass=1)
        body.add_geom(type=g_type, size=size, rgba=RED)
        body.add_freejoint()


def debris(
    spec: mj.MjSpec | None = None, grid_loc: list[float] | None = None, name: str = "debris"
) -> None:
    SQUARE_LENGTH = 2
    THICKNESS = 0.05
    STEP = THICKNESS * 8
    SCALE = 0.1
    BROWN = [0.460, 0.362, 0.216, 1.0]
    RED = [0.6, 0.12, 0.15, 1.0]

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX
    main.mesh.scale = np.array([SCALE] * 3, dtype=np.float64)

    x_beginning = -SQUARE_LENGTH + THICKNESS
    y_beginning = SQUARE_LENGTH - THICKNESS

    # Create tile
    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)
    body.add_geom(size=[SQUARE_LENGTH, SQUARE_LENGTH, THICKNESS], rgba=BROWN)

    # Place debris on the tile
    for i in range(10):
        for j in range(10):
            # draw on xy plane
            drawing = np.random.normal(size=(4, 2))
            drawing /= np.linalg.norm(drawing, axis=1, keepdims=True)
            z = np.zeros((drawing.shape[0], 1))
            # Add z value to drawing
            base = np.concatenate((drawing, z), axis=1)
            # Extrude drawing
            z_extrusion = np.full((drawing.shape[0], 1), THICKNESS * 4)
            top = np.concatenate((drawing, z_extrusion), axis=1)
            # Combine to get a mesh
            mesh = np.vstack((base, top))

            # Create body and add the mesh to the geom of the body
            spec.add_mesh(name=f"d{i}_{j}_{name}", uservert=mesh.flatten())
            pos = [
                grid_loc[0] + x_beginning + i * STEP,
                grid_loc[1] + y_beginning - j * STEP,
                0.2,
            ]

            body = spec.worldbody.add_body(pos=pos, name=f"d{i}_{j}_{name}", mass=1)
            body.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname=f"d{i}_{j}_{name}", rgba=RED)
            body.add_freejoint()


def boxy_terrain(
    spec: mj.MjSpec | None = None,
    grid_loc: list[float] | None = None,
    name: str = "boxy_terrain",
) -> None:
    SQUARE_LENGTH = 2
    CUBE_LENGTH = 0.05
    GRID_SIZE = int(SQUARE_LENGTH / CUBE_LENGTH)
    STEP = CUBE_LENGTH * 2
    BROWN = [0.460, 0.362, 0.216, 1.0]

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    # Create tile
    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)

    x_beginning = -SQUARE_LENGTH + CUBE_LENGTH
    y_beginning = SQUARE_LENGTH - CUBE_LENGTH
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            body.add_geom(
                pos=[
                    x_beginning + i * STEP,
                    y_beginning - j * STEP,
                    random.randint(-1, 1) * CUBE_LENGTH,
                ],
                size=[CUBE_LENGTH] * 3,
                rgba=BROWN,
            )


def box_extrusions(
    spec: mj.MjSpec | None = None,
    grid_loc: list[float] | None = None,
    complex: bool = False,
    name: str = "box_extrusions",
) -> None:
    # Warning! complex sometimes leads to creation of holes
    SQUARE_LENGTH = 2
    CUBE_LENGTH = 0.05
    GRID_SIZE = int(SQUARE_LENGTH / CUBE_LENGTH)
    STEP = CUBE_LENGTH * 2
    BROWN = [0.460, 0.362, 0.216, 1.0]

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    # Create tile
    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)

    x_beginning = -SQUARE_LENGTH + CUBE_LENGTH
    y_beginning = SQUARE_LENGTH - CUBE_LENGTH

    # Create initial grid and store geoms ref
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            ref = body.add_geom(
                pos=[x_beginning + i * STEP, y_beginning - j * STEP, 0],
                size=[CUBE_LENGTH] * 3,
                rgba=BROWN,
            )
            grid[i][j] = ref

    # Extrude or Cut operation using the boxes
    for _ in range(random.randint(4, 50)):
        box = None
        while box is None:
            # Create a box
            start = (random.randint(0, GRID_SIZE - 2), random.randint(0, GRID_SIZE - 2))
            dim = (random.randint(0, GRID_SIZE - 2), random.randint(0, GRID_SIZE - 2))
            # Make suer box is valid
            if start[0] + dim[0] < len(grid) and start[1] + dim[1] < len(grid):
                box = {"start": start, "dim": dim}

        # Use the box to Cut or Extrude
        operation = random.choice([1, -1])
        start = box["start"]
        dim = box["dim"]
        for i in range(start[0], dim[0]):
            for j in range(start[1], dim[1]):
                tile = grid[i][j]
                if complex:
                    tile.pos[2] += operation * CUBE_LENGTH
                else:
                    tile.pos[2] = operation * CUBE_LENGTH


def h_field(
    spec: mj.MjSpec | None = None, grid_loc: list[float] | None = None, name: str = "h_field"
) -> None:
    SQUARE_LENGTH = 2
    HEIGHT = 0.1
    BROWN_RGBA = [0.460, 0.362, 0.216, 1.0]

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    size = 128
    noise = perlin((size, size), (8, 8))

    # Remap noise to 0 to 1
    noise = (noise + 1) / 2
    noise -= np.min(noise)
    noise /= np.max(noise)

    # Makes the edges slope down to avoid sharp boundary
    noise *= edge_slope(size)

    # Create height field
    hfield = spec.add_hfield(
        name=name,
        size=[SQUARE_LENGTH, SQUARE_LENGTH, HEIGHT, HEIGHT / 10],
        nrow=noise.shape[0],
        ncol=noise.shape[1],
        userdata=noise.flatten(),
    )

    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)
    body.add_geom(type=mj.mjtGeom.mjGEOM_HFIELD, hfieldname=name, rgba=BROWN_RGBA)


def floating_platform(
    spec: mj.MjSpec | None = None, grid_loc: list[float] | None = None, name: str = "platform"
) -> None:
    PLATFORM_LENGTH = 0.5
    WIDTH = 0.12
    INWARD_OFFSET = 0.008
    THICKNESS = 0.005
    SIZE = [PLATFORM_LENGTH, WIDTH, THICKNESS]
    TENDON_LENGTH = 0.5
    Z_OFFSET = 0.1

    GOLD = [0.850, 0.838, 0.119, 1]

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    # Platform with sites
    grid_loc[2] += Z_OFFSET
    platform = spec.worldbody.add_body(pos=grid_loc, name=name)
    platform.add_geom(size=SIZE, rgba=GOLD)
    platform.add_freejoint()

    for x_dir in [-1, 1]:
        for y_dir in [-1, 1]:
            # Add site to world
            vector = np.array([x_dir * PLATFORM_LENGTH, y_dir * (WIDTH - INWARD_OFFSET)])
            x_w = grid_loc[0] + vector[0]
            y_w = grid_loc[1] + vector[1]
            z_w = grid_loc[2] + TENDON_LENGTH
            # Rotate sites by theta
            spec.worldbody.add_site(
                name=f"{name}_hook_{x_dir}_{y_dir}", pos=[x_w, y_w, z_w], size=[0.01, 0, 0]
            )
            # Add site to platform
            x_p = x_dir * PLATFORM_LENGTH
            y_p = y_dir * (WIDTH - INWARD_OFFSET)
            platform.add_site(
                name=f"{name}_anchor_{x_dir}_{y_dir}",
                pos=[x_p, y_p, THICKNESS * 2],
                size=[0.01, 0, 0],
            )

            # Connect tendon to sites
            thread = spec.add_tendon(
                name=f"{name}_thread_{x_dir}_{y_dir}",
                limited=True,
                range=[0, TENDON_LENGTH],
                width=0.01,
            )
            thread.wrap_site(f"{name}_hook_{x_dir}_{y_dir}")
            thread.wrap_site(f"{name}_anchor_{x_dir}_{y_dir}")


def simple_suspended_stair(
    spec: mj.MjSpec | None = None,
    grid_loc: list[float] | None = None,
    num_stair: int = 20,
    name: str = "simple_suspended_stair",
) -> None:
    BROWN = [0.460, 0.362, 0.216, 1.0]
    SQUARE_LENGTH = 2
    THICKNESS = 0.05
    OFFSET_Y = -4 / 5 * SQUARE_LENGTH

    V_STEP = 0.076
    H_STEP = 0.12

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    # Create tile
    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)
    body.add_geom(size=[SQUARE_LENGTH, SQUARE_LENGTH, THICKNESS], rgba=BROWN)

    # Create Stairs
    for i in range(num_stair):
        floating_platform(
            spec,
            [grid_loc[0], OFFSET_Y + grid_loc[1] + i * 2 * H_STEP, i * V_STEP],
            name=f"{name}_p_{i}",
        )


def sin_suspended_stair(
    spec: mj.MjSpec | None = None,
    grid_loc: list[float] | None = None,
    num_stair: int = 40,
    name: str = "sin_suspended_stair",
) -> None:
    BROWN = [0.460, 0.362, 0.216, 1.0]
    SQUARE_LENGTH = 2
    THICKNESS = 0.05
    OFFSET_Y = -4 / 5 * SQUARE_LENGTH

    V_STEP = 0.076
    H_STEP = 0.12
    AMPLITUDE = 0.2
    FREQUENCY = 0.5

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    # Plane
    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)
    body.add_geom(size=[SQUARE_LENGTH, SQUARE_LENGTH, THICKNESS], rgba=BROWN)

    for i in range(num_stair):
        x_step = AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * (i * H_STEP))
        floating_platform(
            spec,
            [grid_loc[0] + x_step, OFFSET_Y + grid_loc[1] + i * 2 * H_STEP, i * V_STEP],
            name=f"{name}_p_{i}",
        )


def floating_platform_for_circular_stair(
    spec: mj.MjSpec | None = None,
    grid_loc: list[float] | None = None,
    theta: float = 0,
    name: str = "platform",
) -> None:
    PLATFORM_LENGTH = 0.5
    TENDON_LENGTH = 0.5
    WIDTH = 0.12 / 4  # Platform (body) is made of 4 separate geoms
    THICKNESS = 0.005
    SIZE = [PLATFORM_LENGTH, WIDTH, THICKNESS]
    Z_OFFSET = 0.1

    GOLD = [0.850, 0.838, 0.119, 1]

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX
    spec.compiler.degree = False

    # Platform with sites
    grid_loc[2] += Z_OFFSET
    platform = spec.worldbody.add_body(pos=grid_loc, name=name, euler=[0, 0, theta])
    platform.add_geom(pos=[0, 0, 0], size=SIZE, euler=[0, 0, 0], rgba=GOLD)
    platform.add_geom(pos=[0, 0.02, 0], size=SIZE, euler=[0, 0, 0.05], rgba=GOLD)
    platform.add_geom(pos=[0, 0.05, 0], size=SIZE, euler=[0, 0, 0.1], rgba=GOLD)
    platform.add_geom(pos=[0, 0.08, 0], size=SIZE, euler=[0, 0, 0.15], rgba=GOLD)
    platform.add_freejoint()

    for i, x_dir in enumerate([-1, 1]):
        for j, y_dir in enumerate([-1, 1]):
            # Rotate sites by theta
            rotation_matrix = np.array(
                [[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]]
            )
            vector = np.array([x_dir * PLATFORM_LENGTH, y_dir * WIDTH])
            if i + j == 2:
                vector = np.array([x_dir * PLATFORM_LENGTH, y_dir * 6 * WIDTH])
            vector = np.dot(vector, rotation_matrix)
            x_w = grid_loc[0] + vector[0]
            y_w = grid_loc[1] + vector[1]
            z_w = grid_loc[2] + TENDON_LENGTH

            # Add site to world
            spec.worldbody.add_site(
                name=f"{name}_hook_{x_dir}_{y_dir}", pos=[x_w, y_w, z_w], size=[0.01, 0, 0]
            )
            # Add site to platform
            x_p = x_dir * PLATFORM_LENGTH
            y_p = y_dir * WIDTH
            if i + j == 2:
                y_p = y_dir * 6 * WIDTH
            platform.add_site(
                name=f"{name}_anchor_{x_dir}_{y_dir}",
                pos=[x_p, y_p, THICKNESS * 2],
                size=[0.01, 0, 0],
            )

            # Connect tendon to sites
            thread = spec.add_tendon(
                name=f"{name}_thread_{x_dir}_{y_dir}",
                limited=True,
                range=[0, TENDON_LENGTH],
                width=0.01,
            )
            thread.wrap_site(f"{name}_hook_{x_dir}_{y_dir}")
            thread.wrap_site(f"{name}_anchor_{x_dir}_{y_dir}")


def circular_stairs(
    spec: mj.MjSpec | None = None,
    grid_loc: list[float] | None = None,
    num_stair: int = 60,
    name: str = "circular_stairs",
) -> None:
    BROWN_RGBA = [0.460, 0.362, 0.216, 1.0]
    SQUARE_LENGTH = 2
    THICKNESS = 0.05

    RADIUS = 1.5
    V_STEP = 0.076

    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX
    spec.compiler.degree = False

    # Plane
    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)
    body.add_geom(size=[SQUARE_LENGTH, SQUARE_LENGTH, THICKNESS], rgba=BROWN_RGBA)

    theta_step = 2 * np.pi / num_stair
    for i in range(num_stair):
        theta = i * theta_step
        x = grid_loc[0] + RADIUS * np.cos(theta)
        y = grid_loc[1] + RADIUS * np.sin(theta)
        z = i * V_STEP

        floating_platform_for_circular_stair(spec, [x, y, z], theta=theta, name=f"{name}_p_{i}")
