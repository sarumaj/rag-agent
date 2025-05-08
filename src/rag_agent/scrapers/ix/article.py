import dataclasses as dc
from pathlib import Path
import json
from typing import Any, get_origin, get_args, Iterable
from pydantic import BaseModel

from .config import ExportFormat


@dc.dataclass
class Article:
    id: str
    issue_year: str
    issue_number: str
    url: str
    issue_name: str = ""
    page: str = ""
    title: str = ""
    files: list[Path] = dc.field(default_factory=list)
    export_formats: list[ExportFormat] = dc.field(default_factory=list)


class ArticleEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if dc.is_dataclass(o):
            return dc.asdict(o)
        if isinstance(o, Path) or issubclass(type(o), Path):
            return str(o)
        if issubclass(type(o), BaseModel):
            return o.model_dump(mode="python")
        return super(Article, self).default(o)


class ArticleDecoder(json.JSONDecoder):
    class _Converter:
        def __init__(self, type_args: list[Any], type_origin: Any):
            self.type_args = type_args
            self.type_origin = type_origin

        def __call__(self, x: Iterable[Any]) -> Iterable[Any]:
            if issubclass(self.type_args[0], BaseModel):
                return self.type_origin(
                    self.type_args[0].model_validate(item)
                    for item in x
                )
            else:
                return self.type_origin(
                    self.type_args[0](item)
                    for item in x
                )

        def __repr__(self):
            return f"ArticleDecoder._Converter(type_args={self.type_args}, type_origin={self.type_origin})"

    def __init__(self, *args, **kwargs):
        self._fields = Article.__dataclass_fields__

        self._required_keys = set()
        self._optional_keys = set()
        for key, field in self._fields.items():
            if isinstance(field.default, dc._MISSING_TYPE) and isinstance(field.default_factory, dc._MISSING_TYPE):
                self._required_keys.add(key)
            else:
                self._optional_keys.add(key)

        self._field_type_conversions = {}
        for key, field in self._fields.items():
            if (
                (origin := get_origin(field.type)) in (list, tuple, set) and
                len(type_args := get_args(field.type)) > 0
            ):
                self._field_type_conversions[key] = self._Converter(type_args, origin)

        super(ArticleDecoder, self).__init__(*args, **{**kwargs, "object_hook": self.object_hook})

    def object_hook(self, obj: dict[str, Any]) -> Any:
        if all([
            # check if all required keys are present
            len(self._required_keys-set(obj.keys())) == 0,
            # check if all keys are either required or optional
            len(set(obj.keys())-self._required_keys-self._optional_keys) == 0,
            # check if all values are of the correct type
            *[
                type(value) is self._fields[key].type or
                type(value) is get_origin(self._fields[key].type)
                for key, value in obj.items()
                if key in self._fields
            ],
        ]):
            return Article(**{
                key: (
                    value if key not in self._field_type_conversions else
                    self._field_type_conversions[key](value)
                ) for key, value in obj.items()
            })
        return obj
