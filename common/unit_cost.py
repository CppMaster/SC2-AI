from dataclasses import dataclass
from typing import Union

from pysc2.lib.units import Terran
from pysc2.lib.upgrades import Upgrades

@dataclass
class Cost:
    """
    Represents the cost of units and upgrades in StarCraft II.
    
    This dataclass defines the mineral, vespene gas, and supply costs
    for various units and upgrades in the game.
    
    Attributes
    ----------
    minerals : int
        Mineral cost (default is 0).
    vespene : int
        Vespene gas cost (default is 0).
    supply : int
        Supply cost (default is 0).
    """
    minerals: int = 0
    vespene: int = 0
    supply: int = 0

    def __add__(self, other: "Cost") -> "Cost":
        """
        Add two Cost objects together.
        
        Parameters
        ----------
        other : Cost
            The cost to add to this one.
            
        Returns
        -------
        Cost
            A new Cost object with the summed values.
        """
        return Cost(self.minerals + other.minerals, self.vespene + other.vespene, self.supply + other.supply)

    def clone(self) -> "Cost":
        """
        Create a copy of this Cost object.
        
        Returns
        -------
        Cost
            A new Cost object with the same values.
        """
        return Cost(self.minerals, self.vespene, self.supply)

# Mapping of Terran units to their costs
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

# Mapping of Terran upgrades to their costs
upgrade_to_cost = {
    Upgrades.TerranInfantryArmorsLevel1: Cost(100, 100),
    Upgrades.TerranInfantryArmorsLevel2: Cost(175, 175),
    Upgrades.TerranInfantryArmorsLevel3: Cost(250, 250),
    Upgrades.TerranInfantryWeaponsLevel1: Cost(100, 100),
    Upgrades.TerranInfantryWeaponsLevel2: Cost(175, 175),
    Upgrades.TerranInfantryWeaponsLevel3: Cost(250, 250),
}

def get_item_cost(object_type: Union[Terran, Upgrades]) -> Cost:
    """
    Get the cost for a given unit or upgrade.
    
    Parameters
    ----------
    object_type : Union[Terran, Upgrades]
        The unit or upgrade to get the cost for.
        
    Returns
    -------
    Cost
        The cost of the specified object.
        
    Raises
    ------
    ValueError
        If the object type is not recognized.
    """
    if isinstance(object_type, Terran):
        return unit_to_cost[object_type]
    if isinstance(object_type, Upgrades):
        return upgrade_to_cost[object_type]
    raise ValueError(f"Unknown type: {type(object_type)}")
