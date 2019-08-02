#!/usr/bin/env python3

from collections import namedtuple
from overload import overload


Power = namedtuple('Power', 'min max')
Efficiency = namedtuple('Efficiency', 'charge discharge')


class Storage:
    @overload
    def __init__(self, power, energy, efficiency=1, selfdischarge=1e99):
        """power can be a 2-tuple, defining discharge and charge power,
        if only one value is provided, they are treated equal (except sign).
        Standard values of efficiency self discharge rate equal an ideal
        storage without losses."""
        self._power = None
        self._energy = None
        self._efficiency = None
        self._selfdischarge = None
        self.power = power
        self.energy = energy
        self.efficiency = efficiency
        self.selfdischarge = selfdischarge

    @__init__.add
    def __init__(self, storage):
        self.__init__(storage.power, storage.energy, storage.efficiency,
                      storage.selfdischarge)

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, value):
        try:
            self._power = Power(-float(value), float(value))
        except TypeError:
            self._power = Power(float(value[0]), float(value[1]))
        if self._power.min >= self._power.max:
            raise ValueError('Min power lower or equal max power')

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, value):
        self._energy = float(value)

    @property
    def efficiency(self):
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value):
        try:
            self._efficiency = Efficiency(float(value), float(value))
        except TypeError:
            self._efficiency = Efficiency(float(value[0]), float(value[1]))
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
        strfmt = '<{cls}(power=({pwr.min}, {pwr.max}), ' \
                 'energy={enrgy}, ' \
                 'efficiency=({eff.charge}, {eff.discharge}), ' \
                 'selfdischarge={selfdis})>'
        fields = dict(cls=self.__class__.__name__,
                      pwr=self.power,
                      enrgy=self.energy,
                      eff=self.efficiency,
                      selfdis=self.selfdischarge)
        return strfmt.format(**fields)

    # noinspection PyTypeChecker
    def __mul__(self, other):
        """Lets a storage get multiplied by a scalar to scale the power and
        Energy"""
        factor = float(other)
        return Storage([self.power.min*factor, self.power.max*factor],
                       self.energy*factor, self.efficiency, self.selfdischarge)

    def __rmul__(self, other):
        return self.__mul__(other)
