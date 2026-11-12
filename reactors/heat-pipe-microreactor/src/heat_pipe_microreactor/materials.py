"""Temperature-dependent material definitions for the microreactor core."""

import openmc


SUPPORTED_OPENMC_TEMPERATURES = (250.0, 294.0, 600.0, 900.0, 1200.0, 2500.0)


def nearest_supported_temperature(temp: float) -> float:
    """Return the nearest tabulated temperature supported by the library."""
    return min(SUPPORTED_OPENMC_TEMPERATURES, key=lambda value: abs(value - float(temp)))


def create_fuel_material(name: str = "UN_fuel",
                         enrichment_pct: float = 70.0,
                         temperature_K: float = 1200.0,
                         density_gcm3: float = 13.6) -> openmc.Material:
    """Create UN fuel with a specified U-235 enrichment.

    The mass composition is approximately 94.6% uranium and 5.4% nitrogen.
    """
    fuel = openmc.Material(name=name)
    fuel.set_density("g/cm3", density_gcm3)
    e = enrichment_pct / 100.0
    fuel.add_nuclide("U235", e * 0.946, "wo")
    fuel.add_nuclide("U238", (1.0 - e) * 0.946, "wo")
    fuel.add_nuclide("N14", 0.996 * 0.054, "wo")
    fuel.add_nuclide("N15", 0.004 * 0.054, "wo")
    fuel.temperature = temperature_K
    return fuel


def create_fuel_materials_by_ring(enrichments: list,
                                  temperature_K: float = 1200.0) -> dict:
    """Create one UN fuel material per ring.

    Parameters
    ----------
    enrichments : list of float
        U-235 enrichment percentage for each ring, from ring 1 outward.
    temperature_K : float
        Fuel temperature in kelvin.

    Returns
    -------
    dict
        Mapping of ring index (1..n) to an OpenMC material.
    """
    return {ring: create_fuel_material(f"UN_fuel_r{ring}", e, temperature_K)
            for ring, e in enumerate(enrichments, start=1)}


def create_materials(temperature_K: float = 1200.0,
                     enrichment_pct: float = 70.0) -> dict:
    """Create the materials used in the core.

    Parameters
    ----------
    temperature_K : float
        Initial reference temperature in kelvin.
    enrichment_pct : float
        U-235 enrichment for uniform fuel, in percent.

    Returns
    -------
    dict
        Mapping from material role to ``openmc.Material``.
    """
    materials = {}
    temperature_K = nearest_supported_temperature(temperature_K)

    fuel = create_fuel_material("UN_fuel", enrichment_pct, temperature_K)
    materials["fuel"] = fuel

    helium = openmc.Material(name="He_gap")
    helium.set_density("g/cm3", 0.0001785)
    helium.add_element("He", 1.0, "wo")
    helium.temperature = temperature_K
    materials["helium"] = helium

    clad = openmc.Material(name="MoRe_clad")
    clad.set_density("g/cm3", 10.2)
    clad.add_element("Mo", 0.85, "wo")
    clad.add_element("Re", 0.15, "wo")
    clad.temperature = temperature_K
    materials["clad"] = clad

    sodium = openmc.Material(name="Na_coolant")
    sodium.set_density("g/cm3", 0.856)
    sodium.add_element("Na", 1.0, "wo")
    sodium.temperature = 1000.0
    materials["sodium"] = sodium

    hp_wall = openmc.Material(name="MoRe_hp_wall")
    hp_wall.set_density("g/cm3", 10.2)
    hp_wall.add_element("Mo", 0.85, "wo")
    hp_wall.add_element("Re", 0.15, "wo")
    hp_wall.temperature = 1000.0
    materials["hp_wall"] = hp_wall

    monolith = openmc.Material(name="Nb1Zr_monolith")
    monolith.set_density("g/cm3", 8.57)
    monolith.add_element("Nb", 0.99, "wo")
    monolith.add_element("Zr", 0.01, "wo")
    monolith.temperature = temperature_K
    materials["monolith"] = monolith

    reflector = openmc.Material(name="BeO_reflector")
    reflector.set_density("g/cm3", 2.85)
    reflector.add_element("Be", 0.36, "wo")          # BeO: Be=36%, O=64%
    reflector.add_element("O", 0.64, "wo")
    reflector.temperature = 600.0
    materials["reflector"] = reflector

    return materials


def update_material_temperatures(materials: dict,
                                  temperature_map: dict) -> None:
    """Update material temperatures for Doppler feedback.

    Parameters
    ----------
    materials : dict
        Materials returned by :func:`create_materials`.
    temperature_map : dict
        Mapping from material name to temperature in kelvin.
    """
    for name, temp in temperature_map.items():
        if name in materials:
            materials[name].temperature = temp
