import dataclasses


@dataclasses.dataclass
class Simple:
    foo: str
    bar: str


@dataclasses.dataclass
class OptionalArgument:
    mandatory: str
    optional: str = None


@dataclasses.dataclass
class WithLink:
    baz: str
    simple: Simple


@dataclasses.dataclass
class WithLength:
    num_data: int


@dataclasses.dataclass
class Complex:
    opt: OptionalArgument
    link: WithLink
    length: WithLength = None
