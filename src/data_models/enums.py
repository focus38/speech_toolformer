from enum import Enum


class Language(str, Enum):
    RU = "ru"
    EN = "en"


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TransportType(str, Enum):
    TRAM = "tram"
    TROLLEYBUS = "trolleybus"
    BUS = "bus"


class QueryType(str, Enum):
    TOOL = "tool"
    NO_TOOL = "no_tool"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"


class Source(str, Enum):
    SYNTHETIC = "synthetic"
    MANUAL = "manual"
    EXTERNAL = "external"


class Pipeline(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class ParseStatus(str, Enum):
    OK = "ok"
    NO_TOOL = "no_tool"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"
    ERROR = "error"


class ToolResultStatus(str, Enum):
    OK = "ok"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
