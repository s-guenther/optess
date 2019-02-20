#!/usr/bin/env python3

import os

import optess as oe


# sig = oe.DataFactory.freq(128, time=5, seed=1234)
sig = oe.DataFactory.freq(256, mu=0.5, time=20, seed=314159)
# sdia = oe.SingleDia(sig, rel_obj_fronts=[0.8], resolution=4)
# sdia.compute_parallel()

objval = 11.578
obj = oe.Objective('energy', objval)
storval = 2.25
stor = oe.Storage(storval, 0.95, 50)

if os.path.isfile('medenergy.hyb'):
    hyb = oe.HybridDia(sig, stor, obj, name='medenergy')
    hyb.compute_parallel(cuts=[0.2, 0.5, 0.8], curves=1)
    hyb.save('medenergy.hyb')
else:
    hyb = oe.HybridDia.load('medenergy.hyb')

overmixed = oe.OverdimDia(sig, stor, hyb.inter[0.5], obj, name='medenergy')
overmixed.build_mesh_arrays()
overmixed.compute_parallel()

overdim = oe.overdimdia.OverdimDiaDim(sig, stor, hyb.inter[0.5], obj,
                                      name='medenergy')
overdim.build_mesh_arrays()
overdim.compute_parallel()

overbase = oe.overdimdia.OverdimDiaBase(sig, stor, hyb.inter[0.5], obj,
                                        name='medenergy')
overbase.build_mesh_arrays()
overbase.compute_parallel()

overpeak = oe.overdimdia.OverdimDiaPeak(sig, stor, hyb.inter[0.5], obj,
                                        name='medenergy')
overpeak.build_mesh_arrays()
overpeak.compute_parallel()

overenergy = oe.overdimdia.OverdimDiaEnergy(sig, stor, hyb.inter[0.5], obj,
                                            name='medenergy')
overenergy.build_mesh_arrays()
overenergy.compute_parallel()

overpower = oe.overdimdia.OverdimDiaPower(sig, stor, hyb.inter[0.5], obj,
                                          name='medenergy')
overpower.build_mesh_arrays()
overpower.compute_parallel()

dummybreakpoint = True
