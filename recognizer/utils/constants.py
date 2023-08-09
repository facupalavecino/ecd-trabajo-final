import os

from typing import Dict
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parents[1].resolve()
"""Project's root directory"""

DATASET_DIR = ROOT_DIR / "data" / "lsa64_preprocessed_hand_videos"

ALL_DATASET_DIR = ROOT_DIR / "data" / "all"
"""Preprocessed hand videos directory"""


TARGET_TO_ENCODING: Dict[str, int] = {
    str(i).zfill(3): i-1 for i in range(1, 65)
}
"""Maps target number to an integer encoding"""


TARGET_TO_WORD: Dict[str, str] = {
    "001": "Opaque",
    "002": "Red",
    "003": "Green",
    "004": "Yellow",
    "005": "Bright",
    "006": "Light-blue",
    "007": "Colors",
    "008": "Pink",
    "009": "Women",
    "010": "Enemy",
    "011": "Son",
    "012": "Man",
    "013": "Away",
    "014": "Drawer",
    "015": "Born",
    "016": "Learn",
    "017": "Call",
    "018": "Skimmer",
    "019": "Bitter",
    "020": "Sweet milk",
    "021": "Milk",
    "022": "Water",
    "023": "Food",
    "024": "Argentina",
    "025": "Uruguay",
    "026": "Country",
    "027": "Last name",
    "028": "Where",
    "029": "Mock",
    "030": "Birthday",
    "031": "Breakfast",
    "032": "Photo",
    "033": "Hungry",
    "034": "Map",
    "035": "Coin",
    "036": "Music",
    "037": "Ship",
    "038": "None",
    "039": "Name",
    "040": "Patience",
    "041": "Perfume",
    "042": "Deaf",
    "043": "Trap",
    "044": "Rice",
    "045": "Barbecue",
    "046": "Candy",
    "047": "Chewing-gum",
    "048": "Spaghetti",
    "049": "Yogurt",
    "050": "Accept",
    "051": "Thanks",
    "052": "Shut down",
    "053": "Appear",
    "054": "To land",
    "055": "Catch",
    "056": "Help",
    "057": "Dance",
    "058": "Bathe",
    "059": "Buy",
    "060": "Copy",
    "061": "Run",
    "062": "Realize",
    "063": "Give",
    "064": "Find",
}
"""Map from label number to word"""
