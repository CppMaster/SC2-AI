from dataclasses import dataclass, field
from typing import List, Dict, Union

from pysc2.lib.units import Terran
from pysc2.lib.upgrades import Upgrades


@dataclass
class BuildingRequirements:
    production: List[int] = field(default_factory=lambda: list())
    exists: List[int] = field(default_factory=lambda: list())


unit_to_requirements: Dict[Terran, BuildingRequirements] = {
    Terran.Barracks: BuildingRequirements(exists=[Terran.SupplyDepot]),
    Terran.SCV: BuildingRequirements(production=[Terran.CommandCenter, Terran.PlanetaryFortress,
                                                 Terran.OrbitalCommand]),
    Terran.Marine: BuildingRequirements(production=[Terran.Barracks, Terran.BarracksTechLab, Terran.BarracksReactor]),
    Terran.Factory: BuildingRequirements(exists=[Terran.Refinery, Terran.Barracks]),
    Terran.Armory: BuildingRequirements(exists=[Terran.Refinery, Terran.Factory])
}

upgrade_to_requirements: Dict[Upgrades, BuildingRequirements] = {
    Upgrades.TerranInfantryArmorsLevel1:
        BuildingRequirements(exists=[Terran.Refinery], production=[Terran.EngineeringBay]),
    Upgrades.TerranInfantryArmorsLevel2:
        BuildingRequirements(exists=[Terran.Refinery, Terran.Armory], production=[Terran.EngineeringBay]),
    Upgrades.TerranInfantryArmorsLevel3:
        BuildingRequirements(exists=[Terran.Refinery, Terran.Armory], production=[Terran.EngineeringBay]),
    Upgrades.TerranInfantryWeaponsLevel1:
        BuildingRequirements(exists=[Terran.Refinery], production=[Terran.EngineeringBay]),
    Upgrades.TerranInfantryWeaponsLevel2:
        BuildingRequirements(exists=[Terran.Refinery, Terran.Armory], production=[Terran.EngineeringBay]),
    Upgrades.TerranInfantryWeaponsLevel3:
        BuildingRequirements(exists=[Terran.Refinery, Terran.Armory], production=[Terran.EngineeringBay]),
}


def get_building_requirement(object_type: Union[Terran, Upgrades]) -> BuildingRequirements:
    if isinstance(object_type, Terran):
        return unit_to_requirements.get(object_type, BuildingRequirements())
    if isinstance(object_type, Upgrades):
        return upgrade_to_requirements[object_type]
    raise ValueError(f"Unknown type: {type(object_type)}")
