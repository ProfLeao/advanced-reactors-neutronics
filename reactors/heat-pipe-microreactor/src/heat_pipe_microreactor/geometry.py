"""CSG geometry for the hexagonal microreactor core."""

import math
import openmc
from typing import Tuple, List, Optional


def create_fuel_pin(radius_cm: float = 1.185,
                     gap_outer_cm: float = 1.195,
                     clad_outer_cm: float = 1.219,
                     height_cm: float = 40.0,
                     fuel_material: Optional[openmc.Material] = None,
                     gap_material: Optional[openmc.Material] = None,
                     clad_material: Optional[openmc.Material] = None,
                     monolith_material: Optional[openmc.Material] = None,
                     boundary_type: str = "transmission") -> openmc.Universe:
    """Create a fuel-pin universe with a helium gap and Mo-Re cladding.

    The radial regions are fuel, helium gap, cladding, and surrounding monolith.

    Parameters
    ----------
    radius_cm : float
        Fuel radius.
    gap_outer_cm : float
        Outer radius of the gap (the cladding inner radius).
    clad_outer_cm : float
        Outer radius of the cladding.
    height_cm : float
        Active fuel height.
    fuel_material : openmc.Material
        Fuel material.
    gap_material : openmc.Material
        Gap material.
    clad_material : openmc.Material
        Cladding material.
    monolith_material : openmc.Material
        Monolith material.
    boundary_type : str
        Boundary condition on the outer surface.

    Returns
    -------
    openmc.Universe
        Universe containing the fuel pin.
    """
    half_height = height_cm / 2.0

    # Axial planes
    z_bottom = openmc.ZPlane(z0=-half_height, boundary_type=boundary_type)
    z_top = openmc.ZPlane(z0=+half_height, boundary_type=boundary_type)

    # Radial cylinders
    fuel_surface = openmc.ZCylinder(r=radius_cm)
    gap_surface = openmc.ZCylinder(r=gap_outer_cm)
    clad_surface = openmc.ZCylinder(r=clad_outer_cm)
    outer_surface = openmc.ZCylinder(r=clad_outer_cm + 0.1,
                                      boundary_type=boundary_type)

    # Fuel region
    fuel_region = -fuel_surface & +z_bottom & -z_top
    fuel_cell = openmc.Cell(fill=fuel_material, region=fuel_region, name="fuel")

    # Helium gap
    gap_region = +fuel_surface & -gap_surface & +z_bottom & -z_top
    gap_cell = openmc.Cell(fill=gap_material, region=gap_region, name="gap")

    # Cladding
    clad_region = +gap_surface & -clad_surface & +z_bottom & -z_top
    clad_cell = openmc.Cell(fill=clad_material, region=clad_region, name="clad")

    # Surrounding monolith
    outer_region = +clad_surface & -outer_surface & +z_bottom & -z_top
    outer_cell = openmc.Cell(fill=monolith_material, region=outer_region,
                              name="monolith_around_pin")

    universe = openmc.Universe(cells=[fuel_cell, gap_cell, clad_cell, outer_cell],
                                name="fuel_pin")
    return universe


def create_heat_pipe(inner_radius_cm: float = 0.90,
                      outer_radius_cm: float = 1.02,
                      height_cm: float = 40.0,
                      fluid_material: Optional[openmc.Material] = None,
                      wall_material: Optional[openmc.Material] = None,
                      boundary_type: str = "transmission") -> openmc.Universe:
    """Create a heat-pipe universe with sodium and a Mo-Re wall.

    Parameters
    ----------
    inner_radius_cm : float
        Inner radius of the sodium region.
    outer_radius_cm : float
        Outer radius of the wall.
    height_cm : float
        Active height.
    fluid_material : openmc.Material
        Working-fluid material (sodium).
    wall_material : openmc.Material
        Wall material.
    boundary_type : str
        Boundary condition.

    Returns
    -------
    openmc.Universe
        Universe containing the heat pipe.
    """
    half_height = height_cm / 2.0

    z_bottom = openmc.ZPlane(z0=-half_height, boundary_type=boundary_type)
    z_top = openmc.ZPlane(z0=+half_height, boundary_type=boundary_type)

    inner_cyl = openmc.ZCylinder(r=inner_radius_cm)
    outer_cyl = openmc.ZCylinder(r=outer_radius_cm)
    outer_boundary = openmc.ZCylinder(r=outer_radius_cm + 0.1,
                                       boundary_type=boundary_type)

    # Fluido (Na)
    fluid_region = -inner_cyl & +z_bottom & -z_top
    fluid_cell = openmc.Cell(fill=fluid_material, region=fluid_region,
                              name="hp_fluid")

    # Parede (Mo-Re)
    wall_region = +inner_cyl & -outer_cyl & +z_bottom & -z_top
    wall_cell = openmc.Cell(fill=wall_material, region=wall_region,
                             name="hp_wall")

    # Exterior
    outer_cell = openmc.Cell(fill=None, region=+outer_cyl & -outer_boundary
                              & +z_bottom & -z_top, name="hp_outer")

    universe = openmc.Universe(cells=[fluid_cell, wall_cell, outer_cell],
                                name="heat_pipe")
    return universe


def build_full_core(positions: List[Tuple[float, float, str]],
                     pitch_cm: float,
                     fuel_universe: openmc.Universe,
                     hp_universe: openmc.Universe,
                     monolith_material: openmc.Material,
                     reflector_material: Optional[openmc.Material] = None,
                     fuel_universes: Optional[dict] = None,
                     height_cm: float = 40.0,
                     hex_apothem_cm: Optional[float] = None,
                     reflector_thickness_cm: float = 0.0) \
                     -> Tuple[openmc.Geometry, openmc.Universe]:
    """Build the core as explicit CSG cells.

    Each fuel pin or heat pipe is placed explicitly inside a hexagonal prism.
    The remaining volume is filled with monolith and, optionally, reflector.

    Parameters
    ----------
    fuel_universes : dict, optional
        Mapping from ring number (1..6) to fuel universe for ring-wise
        enrichment. If omitted, ``fuel_universe`` is used for every pin.
    hex_apothem_cm : float, optional
        Apothem (center-to-face distance) of the monolith hexagon. If omitted,
        it is derived from the outermost pin plus a 1 cm margin. The reference
        value from Aziz et al. (2024) is about 24.7 cm.
    reflector_thickness_cm : float, optional
        Radial reflector thickness in centimeters. The vacuum boundary is on
        the outer prism, at ``hex_apothem_cm + reflector_thickness_cm``. A
        zero thickness or missing reflector material places it at the monolith.
    """
    half_height = height_cm / 2.0
    z_bottom = openmc.ZPlane(z0=-half_height, boundary_type="transmission")
    z_top = openmc.ZPlane(z0=half_height, boundary_type="transmission")

    if hex_apothem_cm is None:
        max_radius = 0.0
        for x, y, element_type, *_ in positions:
            radius = 1.219 if element_type == "fuel" else 1.02
            max_radius = max(max_radius, math.hypot(x, y) + radius)
        # Keep the boundary outside every pin; a smaller apothem can leave
        # pins outside the monolith and cause lost particles.
        hex_apothem = max_radius + 1.0
    else:
        hex_apothem = hex_apothem_cm

    edge_length = hex_apothem / (math.sqrt(3.0) / 2.0)
    reflector_thickness = max(float(reflector_thickness_cm or 0.0), 0.0)
    if reflector_material is None or reflector_thickness <= 0.0:
        # With no reflector, use a single prism for the vacuum boundary.
        reflector_thickness = 0.0
        inner_hex = openmc.model.HexagonalPrism(
            edge_length=edge_length,
            orientation="x",
            boundary_type="vacuum"
        )
        outer_hex = None
    else:
        # The inner prism separates monolith from reflector. The outer prism
        # carries the vacuum boundary so the reflector remains in the domain.
        inner_hex = openmc.model.HexagonalPrism(
            edge_length=edge_length,
            orientation="x"
        )
        outer_apothem = hex_apothem + reflector_thickness
        outer_hex = openmc.model.HexagonalPrism(
            edge_length=outer_apothem / (math.sqrt(3.0) / 2.0),
            orientation="x",
            boundary_type="vacuum"
        )

    pin_regions = []
    root_cells = []

    for pos in positions:
        x, y, element_type = pos[0], pos[1], pos[2]
        ring = pos[3] if len(pos) > 3 else 0
        radius = 1.219 if element_type == "fuel" else 1.02
        if element_type == "fuel":
            if fuel_universes and ring in fuel_universes:
                universe = fuel_universes[ring]
            else:
                universe = fuel_universe
        else:
            universe = hp_universe
        cell_name = "fuel_pin" if element_type == "fuel" else "heat_pipe"

        cylinder = openmc.ZCylinder(x0=x, y0=y, r=radius)
        region = -cylinder & +z_bottom & -z_top
        pin_regions.append(region)
        pin_cell = openmc.Cell(fill=universe, region=region, name=cell_name)
        # Pin universes are built at the origin and translated to their core
        # positions so OpenMC can locate their contents.
        pin_cell.translation = (x, y, 0.0)
        root_cells.append(pin_cell)

    if pin_regions:
        monolith_region = -inner_hex & ~pin_regions[0]
        for region in pin_regions[1:]:
            monolith_region = monolith_region & ~region
    else:
        monolith_region = -inner_hex

    root_cells.append(openmc.Cell(fill=monolith_material,
                                 region=monolith_region,
                                 name="monolith"))

    if reflector_material is not None and outer_hex is not None:
        # Region between the monolith and the outer vacuum boundary.
        root_cells.append(openmc.Cell(fill=reflector_material,
                                     region=+inner_hex & -outer_hex,
                                     name="reflector"))

    root_universe = openmc.Universe(cells=root_cells, name="root")
    geometry = openmc.Geometry(root_universe)
    return geometry, root_universe


def generate_hex_positions_aziz(pitch_cm: float = 3.03) -> List[Tuple[float, float, str, int]]:
    """Generate core positions from Table 3 of Aziz et al. (2024).

    The layout has six regular hexagonal rings at a 3.03 cm pitch. Fuel-pin
    counts by ring are 6, 6, 18, 12, 30, and 18. The central heat pipe and
    alternating positions in rings 2, 4, and 6 make 37 heat pipes total.

    Parameters
    ----------
    pitch_cm : float
        Hexagonal lattice pitch in centimeters.

    Returns
    -------
    list of (x, y, type, ring)
        Ring is 0 for the central heat pipe and 1..6 for lattice rings.
    """
    u1 = (pitch_cm, 0.0)
    u2 = (pitch_cm / 2.0, pitch_cm * math.sqrt(3.0) / 2.0)

    fuel_per_ring = {1: 6, 2: 6, 3: 18, 4: 12, 5: 30, 6: 18}  # Tabela 3
    hp_rings = {2, 4, 6}

    positions = [(0.0, 0.0, "heatpipe", 0)]  # Central heat pipe (ring 0).

    # Generate points on the regular hexagonal lattice.
    rings = {r: [] for r in range(1, 7)}
    for i in range(-6, 7):
        for j in range(-6, 7):
            ring = max(abs(i), abs(j), abs(-i - j))
            if 1 <= ring <= 6:
                x = i * u1[0] + j * u2[0]
                y = i * u1[1] + j * u2[1]
                rings[ring].append((x, y))

    for ring in range(1, 7):
        rings[ring].sort(key=lambda p: math.atan2(p[1], p[0]) % (2.0 * math.pi))
        for idx, (x, y) in enumerate(rings[ring]):
            if ring in hp_rings and idx % 2 == 1:
                positions.append((x, y, "heatpipe", ring))
            else:
                positions.append((x, y, "fuel", ring))

    n_fuel = sum(1 for _, _, t, *_ in positions if t == "fuel")
    n_hp = sum(1 for _, _, t, *_ in positions if t == "heatpipe")
    assert n_fuel == sum(fuel_per_ring.values()) and n_hp == 37, \
        f"Invalid layout: {n_fuel} fuel pins, {n_hp} heat pipes"

    return positions
