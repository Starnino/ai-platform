# ----------------- Helpers -----------------

import re
from collections.abc import Mapping
from typing import Any, Dict, Tuple

_ALLOWED_TYPES = {"int", "float", "bool", "str", "list", "dict"}
_ALLOWED_KEYS = {"type", "required", "default", "choices", "min", "max", "pattern", "items"}

def _cast_value(value: Any, to_type: str) -> Any:
    if to_type == "int":
        return int(value)
    if to_type == "float":
        return float(value)
    if to_type == "bool":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot cast '{value}' to bool")
    if to_type == "str":
        return str(value)
    if to_type == "dict":
        if isinstance(value, Mapping):
            return dict(value)
        raise ValueError(f"Expected dict, got {type(value)}")
    if to_type == "list":
        if isinstance(value, (list, tuple)):
            return list(value)
        raise ValueError(f"Expected list/tuple, got {type(value)}")
    raise ValueError(f"Unsupported type '{to_type}'")

def _check_value(value: Any, spec: Mapping[str, Any], path: str = "") -> Any:
    t = spec["type"]

    if t not in _ALLOWED_TYPES:
        raise ValueError(f"{path}: unsupported type '{t}'")

    if t == "list":
        lst = _cast_value(value, "list")
        items = spec.get("items")
        if items:
            itype = items["type"]
            return [
                _check_value(el, {"type": itype}, path=f"{path}[{i}]")
                for i, el in enumerate(lst)
            ]
        return lst

    cv = _cast_value(value, t)

    if t in {"int", "float"}:
        if "min" in spec and cv < spec["min"]:
            raise ValueError(f"{path}: value {cv} < min {spec['min']}")
        if "max" in spec and cv > spec["max"]:
            raise ValueError(f"{path}: value {cv} > max {spec['max']}")
        if "choices" in spec and cv not in spec["choices"]:
            raise ValueError(f"{path}: value {cv} not in choices {spec['choices']}")

    if t == "str":
        if "pattern" in spec and not re.fullmatch(str(spec["pattern"]), cv):
            raise ValueError(f"{path}: value '{cv}' does not match pattern '{spec['pattern']}'")
        if "choices" in spec and cv not in spec["choices"]:
            raise ValueError(f"{path}: value '{cv}' not in choices {spec['choices']}")

    return cv

def _validate_dynamic_spec(spec: Mapping[str, Any], path: str = "") -> Dict[str, Any]:
    """
    Validate and normalize a dynamic parameter specification.
    """
    # checks
    unknown = set(spec.keys()) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys in dynamic spec: {sorted(unknown)}")
    t = spec.get("type")
    if t not in {"int", "float", "bool", "str", "dict", "list"}:
        raise ValueError(f"{path}: unsupported type '{t}'")

    # numeric
    if t in {"int", "float"}:
        for k in ("min", "max"):
            if k in spec and not isinstance(spec[k], (int, float)):
                raise ValueError(f"{path}: '{k}' must be numeric")
        if "min" in spec and "max" in spec and spec["min"] > spec["max"]:
            raise ValueError(f"{path}: 'min' cannot be > 'max'")
    else:
        if "min" in spec or "max" in spec:
            raise ValueError(f"{path}: 'min'/'max' only allowed for int/float")

    # choices
    if "choices" in spec:
        if t not in {"int", "float", "str"}:
            raise ValueError(f"{path}: 'choices' allowed only for int/float/str")
        if not isinstance(spec["choices"], (list, tuple)) or not spec["choices"]:
            raise ValueError(f"{path}: 'choices' must be a non-empty list/tuple")

    # pattern
    if t == "str" and "pattern" in spec:
        try:
            re.compile(str(spec["pattern"]))
        except re.error as e:
            raise ValueError(f"{path}: invalid regex pattern: {e}")
    elif "pattern" in spec:
        raise ValueError(f"{path}: 'pattern' only allowed for str type")

    # items
    if t == "list":
        if "items" in spec:
            items = spec["items"]
            if not isinstance(items, Mapping) or "type" not in items:
                raise ValueError(f"{path}: 'items' must be a mapping with 'type'")
            if items["type"] not in {"int", "float", "bool", "str", "dict"}:
                raise ValueError(f"{path}: 'items.type' must be one of int, float, bool, str, dict")
    else:
        if "items" in spec:
            raise ValueError(f"{path}: 'items' only allowed for list type")

    # normalization
    norm: Dict[str, Any] = {"type": t}
    if spec.get("required"):
        norm["required"] = True
    if "choices" in spec:
        norm["choices"] = list(spec["choices"])
    if "min" in spec:
        norm["min"] = spec["min"]
    if "max" in spec:
        norm["max"] = spec["max"]
    if "pattern" in spec:
        norm["pattern"] = spec["pattern"]
    if "items" in spec:
        norm["items"] = {"type": spec["items"]["type"]}

    if "default" in spec:
        _ = _check_value(spec["default"], norm, path=f"{path}.default")
        norm["default"] = spec["default"]

    return norm

def _split_params(params: Any, path: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(params, Mapping):
        raise ValueError(f"{path}: 'parameters' must be a mapping/dict")

    static_kwargs: Dict[str, Any] = {}
    dynamic_kwargs: Dict[str, Any] = {}

    for k, v in params.items():
        cur_path = f"{path}.{k}" if path else k

        if isinstance(v, Mapping) and "type" in v:
            dynamic_kwargs[k] = _validate_dynamic_spec(v, path=cur_path)
            continue

        if isinstance(v, Mapping):
            child_static, child_dyn = _split_params(v, cur_path)
            if child_static:
                static_kwargs[k] = child_static
            if child_dyn:
                dynamic_kwargs[k] = child_dyn
        else:
            static_kwargs[k] = v

    return static_kwargs, dynamic_kwargs