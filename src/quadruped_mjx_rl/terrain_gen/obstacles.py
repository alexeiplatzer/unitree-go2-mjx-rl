import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from quadruped_mjx_rl.config_utils import Configuration
from quadruped_mjx_rl.terrain_gen.noise_utils import edge_slope, perlin


@dataclass
class Color:
    r: float
    g: float
    b: float
    a: float = 1.0

    @property
    def rgba(self):
        return [self.r, self.g, self.b, self.a]


def _set_body(
    spec: mj.MjSpec | None = None, grid_loc: list[float] | None = None, name: str = "body"
):
    if spec is None:
        spec = mj.MjSpec()

    if grid_loc is None:
        grid_loc = [0, 0]

    # Defaults
    main = spec.default
    main.geom.type = mj.mjtGeom.mjGEOM_BOX

    # Create tile
    body = spec.worldbody.add_body(pos=grid_loc + [0], name=name)
    return body


@dataclass
class TerrainTileConfig(ABC):
    color: Color = Color(0.460, 0.362, 0.216, 1.0)  # (Brown)
    square_side: float = 2.0
    floor_thickness: float = 0.05

    @abstractmethod
    def create_tile(
        self,
        spec: mj.MjSpec | None = None,
        grid_loc: list[float] | None = None,
        name: str = "terrain_tile",
    ):
        pass
        
        
class FlatTile(TerrainTileConfig):
    def create_tile(
        self,
        spec: mj.MjSpec | None = None,
        grid_loc: list[float] | None = None,
        name: str = "flat_tile",
    ):
        body = _set_body(spec, grid_loc, name)
        body.add_geom(
            size=[self.square_side, self.square_side, self.floor_thickness],
            rgba=self.color.rgba,
        )
        
        
class StripesTile(TerrainTileConfig):
    stripe_width: float = 0.1
    stripe_amplitude: float = 0.05
    base_thickness: float = 0.05
    
    def create_tile(
        self,
        spec: mj.MjSpec | None = None,
        grid_loc: list[float] | None = None,
        name: str = "stripes",
    ):
        """
        Generates a 2 m × 2 m tile whose surface is made of long, thin
        stripes running along the y-axis.  Every second stripe is raised
        by `amplitude`, creating a simple uneven terrain.
        """
        body = _set_body(spec, grid_loc, name)

        # Compute how many stripes we can fit across the 2 m span
        n_stripes = int((2 * self.square_side) / self.stripe_width)
        x_start = -self.square_side + self.stripe_width / 2  # centre of first stripe

        for k in range(n_stripes):
            is_high = k % 2 == 1  # alternate low/high
            height = self.base_thickness + (self.stripe_amplitude if is_high else 0.0)
            z_center = height / 2  # MuJoCo boxes are centred
            x_center = x_start + k * self.stripe_width

            body.add_geom(
                pos=[x_center, 0, z_center],
                size=[self.stripe_width / 2, self.square_side, height / 2],
                rgba=self.color,
            )
        

class StairsTile(TerrainTileConfig):
    step_vertical_size: float = 0.076
    step_horizontal_size: float = 0.12
    num_stairs: int = 4
    direction: int = 1
    
    def create_tile(
        self,
        spec: mj.MjSpec | None = None,
        grid_loc: list[float] | None = None,
        name: str = "stairs",
    ):
        H_STEP = self.step_horizontal_size * 2
        V_STEP = self.step_vertical_size * 2

        body = _set_body(spec, grid_loc, name)
        
        # Offset
        x_beginning, y_end = [-self.square_side + self.step_horizontal_size] * 2
        x_end, y_beginning = [self.square_side - self.step_horizontal_size] * 2
        # Dimension
        size_one = [self.step_horizontal_size, self.square_side, self.step_vertical_size]
        size_two = [self.square_side, self.step_horizontal_size, self.step_vertical_size]
        # Geoms positions
        x_pos_l = [x_beginning, 0, self.direction * self.step_vertical_size]
        x_pos_r = [x_end, 0, self.direction * self.step_vertical_size]
        y_pos_up = [0, y_beginning, self.direction * self.step_vertical_size]
        y_pos_down = [0, y_end, self.direction * self.step_vertical_size]

        for i in range(self.num_stairs):
            size_one[1] = self.square_side - H_STEP * i
            size_two[0] = self.square_side - H_STEP * i

            x_pos_l[2], x_pos_r[2], y_pos_up[2], y_pos_down[2] = [
                self.direction * (self.step_vertical_size + V_STEP * i)
            ] * 4

            # Left side
            x_pos_l[0] = x_beginning + H_STEP * i
            body.add_geom(pos=x_pos_l, size=size_one, rgba=self.color.rgba)
            # Right side
            x_pos_r[0] = x_end - H_STEP * i
            body.add_geom(pos=x_pos_r, size=size_one, rgba=self.color.rgba)
            # Top
            y_pos_up[1] = y_beginning - H_STEP * i
            body.add_geom(pos=y_pos_up, size=size_two, rgba=self.color.rgba)
            # Bottom
            y_pos_down[1] = y_end + H_STEP * i
            body.add_geom(pos=y_pos_down, size=size_two, rgba=self.color.rgba)

        # Closing
        size = [
            self.square_side - H_STEP * self.num_stairs,
            self.square_side - H_STEP * self.num_stairs,
            self.step_vertical_size,
        ]
        pos = [0, 0, self.direction * (self.step_vertical_size + V_STEP * self.num_stairs)]
        body.add_geom(pos=pos, size=size, rgba=self.color.rgba)


class DebrisWithSimpleGeoms(TerrainTileConfig):
    debris_color: Color = Color(0.6, 0.12, 0.15, 1.0)  # (Red)

    def create_tile(
        self,
        spec: mj.MjSpec | None = None,
        grid_loc: list[float] | None = None,
        name: str = "simple_debris",
    ):
        body = _set_body(spec, grid_loc, name)
        body.add_geom(
            size=[self.square_side, self.square_side, self.floor_thickness],
            rgba=self.color.rgba,
        )

        # Simple Geoms
        x_beginning, y_end = [-self.square_side + self.floor_thickness] * 2
        x_end, y_beginning = [self.square_side - self.floor_thickness] * 2

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
            body.add_geom(type=g_type, size=size, rgba=self.debris_color.rgba)
            body.add_freejoint()
        

class DebrisWithMeshGeoms(DebrisWithSimpleGeoms):
    scale: float = 0.1
    
    def create_tile(
        self,
        spec: mj.MjSpec | None = None,
        grid_loc: list[float] | None = None,
        name: str = "mesh_debris",
    ):
        step = self.floor_thickness * 8
        
        body = _set_body(spec, grid_loc, name)
        
        spec.default.mesh.scale = np.array([self.scale] * 3, dtype=np.float64)
        
        x_beginning = -self.square_side + self.floor_thickness
        y_beginning = self.square_side - self.floor_thickness

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
                z_extrusion = np.full((drawing.shape[0], 1), self.floor_thickness * 4)
                top = np.concatenate((drawing, z_extrusion), axis=1)
                # Combine to get a mesh
                mesh = np.vstack((base, top))

                # Create a body and add the mesh to the geom of the body
                spec.add_mesh(name=f"d{i}_{j}_{name}", uservert=mesh.flatten())
                pos = [
                    grid_loc[0] + x_beginning + i * step,
                    grid_loc[1] + y_beginning - j * step,
                    0.2,
                ]

                body = spec.worldbody.add_body(pos=pos, name=f"d{i}_{j}_{name}", mass=1)
                body.add_geom(
                    type=mj.mjtGeom.mjGEOM_MESH,
                    meshname=f"d{i}_{j}_{name}", 
                    rgba=self.debris_color.rgba,
                )
                body.add_freejoint()
                
                
class BoxyTerrain(TerrainTileConfig):
    cube_length: float = 0.05
    
    def create_tile(
        self,
        spec: mj.MjSpec | None = None,
        grid_loc: list[float] | None = None,
        name: str = "boxy_terrain",
    ):
        grid_size = int(self.square_side / self.cube_length)
        step = self.cube_length * 2

        body = _set_body(spec, grid_loc, name)

        x_beginning = -self.square_side + self.cube_length
        y_beginning = self.square_side - self.cube_length

        for i in range(grid_size):
            for j in range(grid_size):
                body.add_geom(
                    pos=[
                        x_beginning + i * step,
                        y_beginning - j * step,
                        random.randint(-1, 1) * self.cube_length,
                    ],
                    size=[self.cube_length] * 3,
                    rgba=self.color.rgba,
                )


class BoxExtrusions(BoxyTerrain):


    def create_tile(
        self,
        spec: mj.MjSpec | None = None,
        grid_loc: list[float] | None = None,
        name: str = "box_extrusions",
    ):
        # Warning! complex sometimes leads to creation of holes
        grid_size = int(self.square_side / self.cube_length)
        step = self.cube_length * 2

        body = _set_body(spec, grid_loc, name)

        x_beginning = -self.square_side + self.cube_length
        y_beginning = self.square_side - self.cube_length

        # Create initial grid and store geoms ref
        grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        for i in range(grid_size):
            for j in range(grid_size):
                ref = body.add_geom(
                    pos=[x_beginning + i * step, y_beginning - j * step, 0],
                    size=[self.cube_length] * 3,
                    rgba=self.color.rgba,
                )
                grid[i][j] = ref

        # Extrude or Cut operation using the boxes
        for _ in range(random.randint(4, 50)):
            box = None
            while box is None:
                # Create a box
                start = (random.randint(0, grid_size - 2), random.randint(0, grid_size - 2))
                dim = (random.randint(0, grid_size - 2), random.randint(0, grid_size - 2))
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
                        tile.pos[2] += operation * self.cube_length
                    else:
                        tile.pos[2] = operation * self.cube_length


class HeightField(TerrainTileConfig):
    field_height: float = 0.1

    def create_tile(
        self,
        spec: mj.MjSpec | None = None,
        grid_loc: list[float] | None = None,
        name: str = "height_field",
    ):
        size = 128
        noise = perlin((size, size), (8, 8))

        # Remap noise to 0 to 1
        noise = (noise + 1) / 2
        noise -= np.min(noise)
        noise /= np.max(noise)

        # Makes the edges slope down to avoid a sharp boundary
        noise *= edge_slope(size)

        body = _set_body(spec, grid_loc, name)

        # Create the height field
        hfield = spec.add_hfield(
            name=name,
            size=[
                self.square_side, self.square_side, self.field_height, self.field_height / 10
            ],
            nrow=noise.shape[0],
            ncol=noise.shape[1],
            userdata=noise.flatten(),
        )

        body.add_geom(type=mj.mjtGeom.mjGEOM_HFIELD, hfieldname=name, rgba=self.color.rgba)
