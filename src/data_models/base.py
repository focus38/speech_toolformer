from typing import Annotated

from pydantic import ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr


StrictNonEmptyStr = Annotated[StrictStr, Field(min_length=1)]
RouteNumber = Annotated[StrictStr, Field(min_length=1, pattern=r"^[0-9]+[A-Za-zА-Яа-яЁё]?$")]
NonNegativeStrictInt = Annotated[StrictInt, Field(ge=0)]
SampleRate = Annotated[StrictInt, Field(ge=8000)]
NonNegativeStrictFloat = Annotated[StrictFloat, Field(ge=0)]
Rate = Annotated[StrictFloat, Field(ge=0, le=1)]

STRICT_MODEL_CONFIG = ConfigDict(extra="forbid")

__all__ = [
    "NonNegativeStrictFloat",
    "NonNegativeStrictInt",
    "Rate",
    "RouteNumber",
    "SampleRate",
    "STRICT_MODEL_CONFIG",
    "StrictBool",
    "StrictNonEmptyStr",
    "StrictStr",
]
