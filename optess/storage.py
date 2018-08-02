#!/usr/bin/env python3

from collections import namedtuple
from overload import overload


_Power = namedtuple('_Power', 'min max')
_Efficiency = namedtuple('_Efficiency', 'charge discharge')


class Storage:
    @overload
    def __init__(self, power, efficiency, selfdischarge):
        self._power = None
        self._efficiency = None
        self._selfdischarge = None
        self.power = power
        self.efficiency = efficiency
        self.selfdischarge = selfdischarge

    @__init__.add
    def __init__(self, storage):
        self.__init__(storage.power, storage.efficiency, storage.selfdischarge)

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, value):
        try:
            self._power = _Power(-float(value), float(value))
        except TypeError:
            self._power = _Power(float(value[0]), float(value[1]))
        if self._power.min > self._power.max:
            raise ValueError('Min power lower max power')

    @property
    def efficiency(self):
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value):
        try:
            self._efficiency = _Efficiency(float(value), float(value))
        except TypeError:
            self._efficiency = _Efficiency(float(value[0]), float(value[1]))
        if any(val <= 0 for val in self._efficiency):
            raise ValueError('Efficiency must be greater than zero')

    @property
    def selfdischarge(self):
        return self._selfdischarge

    @selfdischarge.setter
    def selfdischarge(self, value):
        self._selfdischarge = float(value)

    def pprint(self):
        # TODO implement
        pass

    def pplot(self):
        # TODO implement
        pass

    def __repr__(self):
        strfmt = '<{cls}(Power({pwr.min}, {pwr.max}), ' \
                 'Efficiency({eff.charge}, {eff.discharge}), {selfdis})>'
        fields = dict(cls=self.__class__.__name__,
                      pwr=self.power,
                      eff=self.efficiency,
                      selfdis=self.selfdischarge)
        return strfmt.format(**fields)

    def __mul__(self, other):
        """Lets a storage get multiplied by a scalar to scale the power"""
        factor = float(other)
        return Storage([self.power.min*factor, self.power.max*factor],
                       self.efficiency, self.selfdischarge)

    def __rmul__(self, other):
        return self.__mul__(other)
