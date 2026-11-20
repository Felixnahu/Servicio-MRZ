"""Digito de control ICAO 9303: pesos 7-3-1, '<' vale 0, A-Z valen 10-35."""

_WEIGHTS = (7, 3, 1)


def char_value(ch: str) -> int | None:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    return None


def compute(data: str) -> int | None:
    """Digito de control de `data`; None si trae caracteres fuera de [0-9A-Z<]."""
    total = 0
    for i, ch in enumerate(data):
        value = char_value(ch)
        if value is None:
            return None
        total += value * _WEIGHTS[i % 3]
    return total % 10


def verify(data: str, check: str) -> bool:
    """`check` debe ser un digito; '<' nunca valida (marca un numero de documento extendido)."""
    expected = compute(data)
    return expected is not None and check.isdigit() and int(check) == expected
