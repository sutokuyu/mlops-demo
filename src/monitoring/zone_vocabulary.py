"""The words the owner uses for a zone, mapped onto the identifiers in the data.

Why this module exists
----------------------
Measured on 2026-10-02. Asked "猫有没有进过水池", the bot answered "没进过水池，数据里没
这个记录" - while the database held kurumi in the ``sink`` zone four times that day,
including the 02:00:16 stay the owner had just watched. Asked again with the identifier
("kurumi 有没有在 sink 这个区域待过"), the same model on the same summary answered
correctly, times and all. So it was not retrieval:

* the question's word and the stored identifier never met - the model translates zone
  identifiers on its own, freely and inconsistently between runs - and
* ``GROUNDING_RULES`` then turned "I cannot match this word" into a confident "there is
  no such record" (one run even invented support: "sink only appears under bagel",
  which is the exact opposite of the data).

One shared vocabulary fixes that half. Each zone lists the words that may refer to it;
:func:`zone_aliases` reads them from ``report.zone_aliases`` (defaults here), the query
tools use them to resolve a question into identifiers, and the prompt gets the same list
as the allowed vocabulary.

Zone names repeat across cameras - living_room and sofa both have a ``sink``, a ``floor``
and an ``on_dining_table`` - so a resolved zone is a *name*, not a place. Anything that
reports a stay must also say which camera.
"""

import re
from collections.abc import Iterable, Mapping, Sequence

from src.monitoring.location_config import REPORT_CONFIG
from src.monitoring.location_zones import load_calibrations

# What each zone is called in ordinary speech. Kept here as the fallback so a config
# without a `report.zone_aliases` block still has a usable vocabulary; the config wins
# per zone when it has an entry.
DEFAULT_ZONE_ALIASES: dict[str, list[str]] = {
    "sink": ["水池", "水槽", "洗碗池", "洗碗槽", "洗手池"],
    "toilet_1": ["猫砂盆", "厕所", "猫厕所"],
    "feeder": ["喂食器", "食盆"],
    "wet_food_bowl_1": ["湿粮碗1"],
    "wet_food_bowl_2": ["湿粮碗2"],
    "water_server": ["饮水机", "水碗", "水盆"],
    "bay_window": ["飘窗", "窗台"],
    "kitchen_counter": ["厨房台面", "料理台", "灶台"],
    "stove": ["炉子", "灶", "炉灶"],
    "on_dining_table": ["餐桌上", "餐桌", "桌上", "餐桌顶部", "餐桌顶上", "餐桌上面"],
    "under_dining_table": ["餐桌底下", "桌下", "餐桌底部", "桌底", "餐桌下方", "餐桌下面"],
    "cat_wall": ["猫墙", "猫爬架", "猫爬架上", "猫爬架顶部", "猫爬架顶上", "猫爬架上面"],
    "on_tv": ["电视上", "电视"],
    "on_white_chair": ["白色椅子", "白椅子", "白色椅子上", "白椅子上"],
    "near_living_room_curtain": ["窗帘", "窗帘旁", "窗帘附近", "窗帘周围", "窗帘旁边"],
    "carpet": ["地毯", "地毯上"],
    "carpet_(in_front_of_sofa)": ["沙发前地毯"],
    "on_sofa": ["沙发上", "沙发", "沙发顶部", "沙发顶上", "沙发上面"],
    "under_sofa": ["沙发底下", "沙发底部", "沙发底", "沙发下方"],
    "on_side_cabinet": ["边柜上", "边柜顶部", "边柜顶上", "边柜上面"],
    "near_side_cabinet": ["边柜旁", "边柜附近", "边柜周围", "边柜旁边"],
    "on_kangaroo_chair": ["袋鼠椅上"],
    "under_kangaroo_chair": ["袋鼠椅底下"],
    "floor": ["地板", "地上"],
}

# Words that mean more than one place. "湿粮碗" is the cat's wet food, and there are two
# of them: the owner means either one (and usually both in the same answer), so a
# question about 湿粮碗 has to reach both zones. Kept separate from the aliases above on
# purpose - there a word claimed by two zones is a typo and a test fails on it, whereas
# here naming several zones is the whole point.
DEFAULT_ZONE_GROUPS: dict[str, list[str]] = {
    "湿粮碗": ["wet_food_bowl_1", "wet_food_bowl_2"],
    "饭碗": ["wet_food_bowl_1", "wet_food_bowl_2"],
}


def _as_words(raw) -> list[str]:
    """Accept a list, a single word, or a comma-separated string (config ergonomics)."""
    if raw in (None, "", []):
        return []
    if isinstance(raw, str):
        raw = [raw]
    words: list[str] = []
    for item in raw:
        for part in str(item).replace("，", ",").split(","):
            part = part.strip()
            if part:
                words.append(part)
    return words


def zone_aliases(config: Mapping | None = None) -> dict[str, list[str]]:
    """Zone identifier -> the words that may refer to it.

    The configured block wins per zone (a configured entry replaces that zone's
    defaults rather than appending to them, so removing a word is possible).
    """
    raw = (config if config is not None else REPORT_CONFIG.get("zone_aliases")) or {}
    merged = {zone: list(words) for zone, words in DEFAULT_ZONE_ALIASES.items()}
    for zone, words in raw.items():
        merged[str(zone)] = _as_words(words)
    return merged


def zone_groups(config: Mapping | None = None) -> dict[str, list[str]]:
    """Owner word -> every zone it can mean.

    The configured block replaces a word's default list; other words keep theirs. See
    ``DEFAULT_ZONE_GROUPS`` for why this is not folded into ``zone_aliases``.
    """
    raw = (config if config is not None else REPORT_CONFIG.get("zone_groups")) or {}
    merged = {word: list(zones) for word, zones in DEFAULT_ZONE_GROUPS.items()}
    for word, zones in raw.items():
        merged[str(word)] = [str(zone) for zone in _as_words(zones)]
    return merged


def alias_index(aliases: Mapping[str, Iterable[str]] | None = None) -> dict[str, str]:
    """Owner word -> zone identifier, lower-cased for matching.

    One word to one zone by design: a word claimed by two zones here is a typo, and
    :func:`ambiguous_words` exists so a test fails on it. A word that genuinely means
    several places belongs in :func:`zone_groups` instead.
    """
    index: dict[str, str] = {}
    for zone, words in (aliases if aliases is not None else zone_aliases()).items():
        for word in words:
            index[str(word).lower()] = zone
    return index


def ambiguous_words(aliases: Mapping[str, Iterable[str]] | None = None) -> dict[str, list[str]]:
    """Words claimed by more than one zone, e.g. ``{"水盆": ["sink", "water_server"]}``."""
    owners: dict[str, list[str]] = {}
    for zone, words in (aliases if aliases is not None else zone_aliases()).items():
        for word in words:
            owners.setdefault(str(word).lower(), []).append(zone)
    return {word: zones for word, zones in owners.items() if len(zones) > 1}


def known_zones() -> list[str]:
    """Every zone identifier that exists in zones.yaml, sorted and de-duplicated."""
    names = {
        zone.name for calibration in load_calibrations().values() for zone in calibration.zones
    }
    return sorted(names)


def _identifier_pattern(zone: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(zone)}(?![A-Za-z0-9_])", re.IGNORECASE)


def word_map(
    aliases: Mapping[str, Iterable[str]] | None = None,
    groups: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, list[str]]:
    """Every owner word -> every zone it can mean, lower-cased.

    Aliases and groups are merged here rather than at the call site, so a resolver never
    has to know which of the two a word came from. A word in both gets the union.
    """
    mapping: dict[str, list[str]] = {}

    def add(word: str, zones: Iterable[str]) -> None:
        bucket = mapping.setdefault(str(word).lower(), [])
        for zone in zones:
            if zone not in bucket:
                bucket.append(zone)

    for word, zone in alias_index(aliases).items():
        add(word, [zone])
    for word, zones in (groups if groups is not None else zone_groups()).items():
        add(word, zones)
    return mapping


def resolve_zones(
    text: str,
    *,
    aliases: Mapping[str, Iterable[str]] | None = None,
    groups: Mapping[str, Iterable[str]] | None = None,
    zones: Sequence[str] | None = None,
) -> list[str]:
    """Every zone the text refers to, by identifier or by one of its words.

    Matched longest first and **consumed**, so "餐桌底下" resolves to
    ``under_dining_table`` instead of also to ``on_dining_table`` through the "餐桌"
    inside it, while "餐桌底下和沙发上" still finds both. A word that means several
    zones contributes all of them, so "去过湿粮碗吗" reaches both bowls. Returns
    identifiers in match order, de-duplicated.
    """
    remaining = text
    found: list[str] = []
    for word, targets in sorted(
        word_map(aliases, groups).items(), key=lambda item: (-len(item[0]), item[0])
    ):
        if not word:
            continue
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        if not pattern.search(remaining):
            continue
        for zone in targets:
            if zone not in found:
                found.append(zone)
        remaining = pattern.sub(" ", remaining)
    for zone in zones if zones is not None else known_zones():
        pattern = _identifier_pattern(zone)
        if not pattern.search(remaining):
            continue
        if zone not in found:
            found.append(zone)
        remaining = pattern.sub(" ", remaining)
    return found


def vocabulary_text(
    *,
    aliases: Mapping[str, Iterable[str]] | None = None,
    groups: Mapping[str, Iterable[str]] | None = None,
    zones: Sequence[str] | None = None,
) -> str:
    """The mapping as prompt text: ``sink = 水池/水槽`` one zone per line.

    Only zones that exist in the data are listed, so the model is never handed a name it
    cannot find - and a zone with no words at all still appears, because knowing the
    identifier is enough to answer a question that uses it. A word that means several
    zones is listed as ``wet_food_bowl_1 + wet_food_bowl_2 = 湿粮碗 (either one)``.
    """
    existing = list(zones if zones is not None else known_zones())
    mapping = zone_aliases() if aliases is None else {z: list(w) for z, w in aliases.items()}
    lines = []
    for zone in existing:
        words = mapping.get(zone) or []
        lines.append(f"- {zone} = {'/'.join(words)}" if words else f"- {zone}")
    for zone, words in mapping.items():
        if zone not in existing and words:
            lines.append(f"- {zone} (not present in any camera's zones) = {'/'.join(words)}")
    for word, zones_ in (groups if groups is not None else zone_groups()).items():
        targets = list(zones_)
        if len(targets) > 1:
            lines.append(f"- {' + '.join(targets)} = {word} (either one)")
    return "\n".join(lines)
