from dataclasses import dataclass, field
from typing import List, Dict, Union

from pysc2.lib.units import Terran
from pysc2.lib.upgrades import Upgrades

@dataclass
class BuildingRequirements:
    """
    Represents the building requirements for units and upgrades in StarCraft II.
    
    This dataclass defines what buildings are required to produce units or research upgrades.
    
    Attributes
    ----------
    production : List[int]
        List of building types required for production (e.g., Barracks for Marines).
    exists : List[int]
        List of building types that must exist (e.g., Supply Depot for Barracks).
    """
    production: List[int] = field(default_factory=lambda: list())
    exists: List[int] = field(default_factory=lambda: list())

# Mapping of Terran units to their building requirements
unit_to_requirements: Dict[Terran, BuildingRequirements] = {
    Terran.Barracks: BuildingRequirements(exists=[Terran.SupplyDepot]),
    Terran.SCV: BuildingRequirements(production=[Terran.CommandCenter, Terran.PlanetaryFortress,
                                                 Terran.OrbitalCommand]),
    Terran.Marine: BuildingRequirements(production=[Terran.Barracks, Terran.BarracksTechLab, Terran.BarracksReactor]),
    Terran.Factory: BuildingRequirements(exists=[Terran.Refinery, Terran.Barracks]),
    Terran.Armory: BuildingRequirements(exists=[Terran.Refinery, Terran.Factory])
}

# Mapping of Terran upgrades to their building requirements
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
    """
    Get the building requirements for a given unit or upgrade.
    
    Parameters
    ----------
    object_type : Union[Terran, Upgrades]
        The unit or upgrade to get requirements for.
        
    Returns
    -------
    BuildingRequirements
        The building requirements for the specified object.
        
    Raises
    ------
    ValueError
        If the object type is not recognized.
    """
    if isinstance(object_type, Terran):
        return unit_to_requirements.get(object_type, BuildingRequirements())
    if isinstance(object_type, Upgrades):
        return upgrade_to_requirements[object_type]
    raise ValueError(f"Unknown type: {type(object_type)}")
