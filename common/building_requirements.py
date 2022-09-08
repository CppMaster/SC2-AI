from dataclasses import dataclass, field
from typing import List, Dict, Set

from pysc2.lib.units import Terran


@dataclass
class BuildingRequirements:
    production: List[int] = field(default_factory=lambda: list())
    exists: List[int] = field(default_factory=lambda: list())


unit_to_requirements: Dict[int, BuildingRequirements] = {
    Terran.Barracks: BuildingRequirements(exists=[Terran.SupplyDepot]),
    Terran.SCV: BuildingRequirements(production=[Terran.CommandCenter, Terran.PlanetaryFortress,
                                                 Terran.OrbitalCommand]),
    Terran.Marine: BuildingRequirements(production=[Terran.Barracks, Terran.BarracksTechLab, Terran.BarracksReactor])
}


def get_building_requirement(unit_type: int) -> BuildingRequirements:
    return unit_to_requirements.get(unit_type, BuildingRequirements())
