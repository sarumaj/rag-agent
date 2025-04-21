import dataclasses as dc
from pathlib import Path
import json
from typing import Any, get_origin, get_args


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


class ArticleEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if dc.is_dataclass(o):
            return dc.asdict(o)
        if isinstance(o, Path) or issubclass(type(o), Path):
            return str(o)
        return super(Article, self).default(o)


class ArticleDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        self._fields = Article.__dataclass_fields__
        self._required_keys = {
            key for key, field in self._fields.items()
            if isinstance(field.default, dc._MISSING_TYPE)
            and isinstance(field.default_factory, dc._MISSING_TYPE)
        }
        self._optional_keys = {
            key for key, field in self._fields.items()
            if not isinstance(field.default, dc._MISSING_TYPE)
            or not isinstance(field.default_factory, dc._MISSING_TYPE)
        }
        self._field_type_conversions = {
            key: lambda x: [get_args(field.type)[0](item) for item in x]
            for key, field in self._fields.items()
            if get_origin(field.type) in (list, tuple, set)
            and len(get_args(field.type)) > 0
        }
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
