import json
from collections import OrderedDict


def obj2confattr(obj):
    if isinstance(obj, list):
        obj = ListAttribute(obj)
    elif isinstance(obj, tuple):
        obj = TupleAttribute(obj)
    elif isinstance(obj, dict):
        obj = DictAttribute(obj)
    return obj


class AttributeJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Attribute):
            return obj._value
        return json.JSONEncoder.default(self, obj)


class Attribute:

    __slots__ = "_value"

    def __init__(self, value):
        self._value = value

    def __repr__(self):
        return self.__str__()

    @property
    def ref(self):
        raise PermissionError

    @ref.setter
    def ref(self, py_value):
        raise NotImplementedError

    @staticmethod
    def check_type(obj, t1, *ts):
        ts = (t1,) + tuple(ts)
        if not isinstance(obj, ts):
            raise TypeError("Expect type {}, but got {}.".format(ts, type(obj)))


class TupleAttribute(Attribute):

    __slots__ = "_value"

    def __init__(self, value):
        Attribute.check_type(value, tuple)
        super().__init__(tuple(map(obj2confattr, value)))

    def __str__(self):
        return str(self._value)

    def __len__(self):
        return len(self._value)

    def __getitem__(self, index):
        return self._value[index]

    @property
    def ref(self):
        raise PermissionError

    @ref.setter
    def ref(self, py_value):
        Attribute.check_type(py_value, tuple)
        if len(self._value) != len(py_value):
            raise ValueError("Expect len(py_value) == {}, but got {}.".format(
                len(self._value),
                py_value
            ))
        for i in range(len(self._value)):
            if isinstance(self._value[i], Attribute):
                self._value[i].ref = py_value[i]
            else:
                self._value[i] = py_value[i]


class ListAttribute(Attribute):

    __slots__ = "_value"

    def __init__(self, value):
        Attribute.check_type(value, list)
        super().__init__(value)

    def __str__(self):
        return str(self._value)

    def __len__(self):
        return len(self._value)

    def __getitem__(self, index):
        return self._value[index]

    @property
    def ref(self):
        raise PermissionError

    @ref.setter
    def ref(self, py_value):
        Attribute.check_type(py_value, list)
        for item in py_value:
            if isinstance(item, Attribute):
                raise ValueError("ListAttribute must not contain Attribute value.")
        self._value = py_value


class DictAttribute(Attribute):

    __slots__ = "_value"

    def __init__(self, value):
        Attribute.check_type(value, dict)
        super().__init__(OrderedDict())
        for k, v in value.items():
            self._value[k] = obj2confattr(v)

    def __str__(self):
        return str(self._value)

    def __len__(self):
        return len(self._value)

    def __getattr__(self, key):
        if key in self._value:
            return self[key]
        else:
            return super().__getattr__(key)

    def __contains__(self, key):
        return key in self._value

    def __getitem__(self, key):
        return self._value[key]

    def keys(self):
        yield from self._value.keys()

    def values(self):
        yield from map(lambda k: self[k], self.keys)

    def items(self):
        yield from zip(self.keys(), self.values())

    def __iter__(self):
        yield from self.keys()

    @property
    def ref(self):
        raise PermissionError

    @ref.setter
    def ref(self, py_value):
        Attribute.check_type(py_value, dict)
        for k in py_value:
            if k not in self._value:
                raise KeyError(k)
            v = self._value[k]
            if isinstance(v, Attribute):
                v.ref = py_value[k]
            else:
                self._value[k] = py_value[k]


class JsonConfigurable(DictAttribute):

    __slots__ = "_value"

    def loads(self, *args, **kwargs):
        self.ref = json.loads(*args, **kwargs)

    def dumps(
        self,
        *,
        skipkeys=False,
        ensure_ascii=True,
        check_circular=True,
        allow_nan=True,
        cls=AttributeJsonEncoder,
        indent=None,
        separators=None,
        default=None,
        sort_keys=False,
        **kw
    ):
        return json.dumps(
            self,
            skipkeys=skipkeys,
            ensure_ascii=ensure_ascii,
            check_circular=check_circular,
            allow_nan=allow_nan,
            cls=cls,
            indent=indent,
            separators=separators,
            default=default,
            sort_keys=sort_keys,
            **kw
        )
