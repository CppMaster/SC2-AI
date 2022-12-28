from dataclasses import dataclass
from typing import Union

from pysc2.lib.units import Terran
from pysc2.lib.upgrades import Upgrades


@dataclass
class Cost:
    minerals: int = 0
    vespene: int = 0
    supply: int = 0

    def __add__(self, other: "Cost"):
        return Cost(self.minerals + other.minerals, self.vespene + other.vespene, self.supply + other.supply)


unit_to_cost = {
    Terran.CommandCenter: Cost(400),
    Terran.SupplyDepot: Cost(100),
    Terran.Barracks: Cost(150),
    Terran.SCV: Cost(50, 0, 1),
    Terran.Marine: Cost(50, 0, 1),
    Terran.Refinery: Cost(75),
    Terran.EngineeringBay: Cost(125),
    Terran.Factory: Cost(150, 100),
    Terran.Armory: Cost(150, 100)
}

upgrade_to_cost = {
    Upgrades.TerranInfantryArmorsLevel1: Cost(100, 100),
    Upgrades.TerranInfantryArmorsLevel2: Cost(175, 175),
    Upgrades.TerranInfantryArmorsLevel3: Cost(250, 250),
    Upgrades.TerranInfantryWeaponsLevel1: Cost(100, 100),
    Upgrades.TerranInfantryWeaponsLevel2: Cost(175, 175),
    Upgrades.TerranInfantryWeaponsLevel3: Cost(250, 250),
}


def get_item_cost(object_type: Union[Terran, Upgrades]) -> Cost:
    if isinstance(object_type, Terran):
        return unit_to_cost[object_type]
    if isinstance(object_type, Upgrades):
        return upgrade_to_cost[object_type]
    raise ValueError(f"Unknown type: {type(object_type)}")
