from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolTemplate:
    text: str
    city_surface: str
    transport_surface: str


@dataclass(frozen=True)
class NoToolTemplate:
    text: str
    answer: str


_CITY_SURFACES = {
    "en": {
        "moscow": "Moscow",
        "saint_petersburg": "Saint Petersburg",
        "berlin": "Berlin",
        "london": "London",
    },
    "ru": {
        "moscow": "Москве",
        "saint_petersburg": "Санкт-Петербурге",
        "berlin": "Берлине",
        "london": "Лондоне",
    },
}

_TRANSPORT_SURFACES = {
    "en": {
        "tram": "tram",
        "trolleybus": "trolleybus",
        "bus": "bus",
    },
    "ru": {
        "tram": "трамвай",
        "trolleybus": "троллейбус",
        "bus": "автобус",
    },
}

_TOOL_PATTERNS = {
    "en": [
        "Where is {transport_surface} {route_number} in {city_surface} right now?",
        "Find {transport_surface} number {route_number} in {city_surface}.",
        "Tell me the current location of {transport_surface} {route_number} in {city_surface}.",
        "Is {transport_surface} {route_number} moving in {city_surface}?",
    ],
    "ru": [
        "Где сейчас едет {transport_surface} номер {route_number} в {city_surface}?",
        "Найди {transport_surface} {route_number} в {city_surface}.",
        "Покажи текущее положение: {transport_surface} {route_number}, {city_surface}.",
        "Где находится {transport_surface} {route_number} в {city_surface}?",
    ],
}

_NO_TOOL_TEMPLATES = {
    "en": [
        NoToolTemplate("What is a trolleybus?", "A trolleybus is an electric bus powered by overhead wires."),
        NoToolTemplate("How do I validate a transit ticket?", "Check the local operator rules and validate the ticket before travel."),
        NoToolTemplate("What is the difference between a tram and a bus?", "A tram runs on rails, while a bus runs on roads."),
        NoToolTemplate("Can I bring a bicycle on public transport?", "Bicycle rules depend on the city and operator."),
    ],
    "ru": [
        NoToolTemplate("Что такое троллейбус?", "Троллейбус — это электрический автобус с питанием от контактной сети."),
        NoToolTemplate("Как оплатить проезд в автобусе?", "Способ оплаты зависит от города и перевозчика."),
        NoToolTemplate("Чем трамвай отличается от автобуса?", "Трамвай движется по рельсам, а автобус едет по дороге."),
        NoToolTemplate("Можно ли провозить велосипед в транспорте?", "Правила провоза велосипеда зависят от города и оператора."),
    ],
}


def render_tool_query(language: str, city: str, transport_type: str, route_number: str, template_index: int) -> ToolTemplate:
    city_surface = _CITY_SURFACES[language].get(city, city)
    transport_surface = _TRANSPORT_SURFACES[language][transport_type]
    patterns = _TOOL_PATTERNS[language]
    text = patterns[template_index % len(patterns)].format(
        city_surface=city_surface,
        transport_surface=transport_surface,
        route_number=route_number,
    )
    return ToolTemplate(text=text, city_surface=city_surface, transport_surface=transport_surface)


def render_no_tool_query(language: str, template_index: int) -> NoToolTemplate:
    templates = _NO_TOOL_TEMPLATES[language]
    return templates[template_index % len(templates)]


__all__ = ["NoToolTemplate", "ToolTemplate", "render_no_tool_query", "render_tool_query"]
