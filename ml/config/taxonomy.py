"""Shared domain taxonomy: regions, categories, and the attribute types/values
valid for each category. This is the single source of truth used by the data
generator, feature pipeline, and API — never redefine these lists elsewhere.
"""

from __future__ import annotations

REGIONS: list[str] = [
    "Region A",
    "Region B",
    "Region C",
    "Region D",
    "Region E",
    "Region F",
]

# Each region gets a rough climate profile used by the data generator to
# drive seasonality strength (winter-heavy regions see sharper outerwear
# seasonality, etc.) and a set of attribute-value affinities (preference skew).
REGION_CLIMATE: dict[str, str] = {
    "Region A": "cold",
    "Region B": "temperate",
    "Region C": "warm",
    "Region D": "cold",
    "Region E": "temperate",
    "Region F": "warm",
}

CATEGORIES: list[str] = [
    "Outerwear",
    "Footwear",
    "Tops",
    "Bottoms",
    "Accessories",
]

# attribute_type -> list of possible attribute_value per category
CATEGORY_ATTRIBUTES: dict[str, dict[str, list[str]]] = {
    "Outerwear": {
        "material": ["wool", "down", "cotton-blend", "synthetic-shell", "fleece"],
        "closure_type": ["zip", "button", "snap", "toggle"],
        "weight_class": ["light", "mid-weight", "heavy"],
        "fit": ["slim", "regular", "oversized"],
    },
    "Footwear": {
        "material": ["leather", "canvas", "synthetic-mesh", "suede"],
        "closure_type": ["lace-up", "slip-on", "velcro", "buckle"],
        "sole_type": ["rubber", "eva-foam", "leather-sole"],
        "fit": ["narrow", "regular", "wide"],
    },
    "Tops": {
        "material": ["cotton", "linen", "polyester-blend", "silk", "wool"],
        "sleeve_type": ["short-sleeve", "long-sleeve", "sleeveless", "three-quarter"],
        "fit": ["slim", "regular", "relaxed"],
        "pattern": ["solid", "striped", "printed", "checked"],
    },
    "Bottoms": {
        "material": ["denim", "cotton-twill", "linen", "synthetic-blend"],
        "fit": ["skinny", "straight", "relaxed", "wide-leg"],
        "closure_type": ["zip", "elastic-waist", "drawstring", "button"],
        "pattern": ["solid", "striped", "printed"],
    },
    "Accessories": {
        "material": ["leather", "canvas", "wool", "metal", "synthetic"],
        "closure_type": ["buckle", "zip", "clasp", "none"],
        "pattern": ["solid", "printed", "textured"],
    },
}


def all_attribute_types_for_category(category: str) -> list[str]:
    return list(CATEGORY_ATTRIBUTES[category].keys())


def all_attribute_values(category: str, attribute_type: str) -> list[str]:
    return CATEGORY_ATTRIBUTES[category][attribute_type]
