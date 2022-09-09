from dataclasses import dataclass

from pysc2.lib.units import Terran


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
    Terran.Marine: Cost(50, 0, 1)
}
