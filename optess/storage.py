#!/usr/bin/env python3

from collections import namedtuple
from overload import overload
from warnings import warn


Power = namedtuple('Power', 'min max')
Efficiency = namedtuple('Efficiency', 'charge discharge')


class Storage:
    @overload
    def __init__(self, power, energy, efficiency, selfdischarge):
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
        if self._power.min > self._power.max:
            raise ValueError('Min power lower max power')

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

    def __mul__(self, other):
        """Lets a storage get multiplied by a scalar to scale the power and
        Energy"""
        factor = float(other)
        return Storage([self.power.min*factor, self.power.max*factor],
                       [self.energy.min*factor, self.energy.max*factor],
                       self.efficiency, self.selfdischarge)

    def __rmul__(self, other):
        return self.__mul__(other)


class IdealStorage(Storage):
    def __init__(self, power, energy, efficiency=None, selfdischarge=None):
        if efficiency is not None or selfdischarge is not None:
            warn('An ideal storage does not have any losses, ignoring '
                 'efficiency and self discharge values.')
        efficiency = 1
        selfdischarge = 1e99
        super().__init__(power, energy, efficiency, selfdischarge)

    def __repr__(self):
        strfmt = '<{cls}(power=({pwr.min}, {pwr.max}), ' \
                 'energy={enrgy})>'
        fields = dict(cls=self.__class__.__name__,
                      pwr=self.power,
                      enrgy=self.energy)
        return strfmt.format(**fields)

    def __mul__(self, other):
        """Lets a storage get multiplied by a scalar to scale the power and
        Energy"""
        factor = float(other)
        # noinspection PyTypeChecker
        return IdealStorage([self.power.min*factor, self.power.max*factor],
                            [self.energy.min*factor, self.energy.max*factor])
