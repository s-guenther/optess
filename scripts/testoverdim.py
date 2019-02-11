#!/usr/bin/env python3

import os

import optess as oe


# sig = oe.DataFactory.freq(128, time=5, seed=1234)
sig = oe.DataFactory.freq(256, time=20, seed=314159)

objval = 10.2
obj = oe.Objective('power', objval)
storval = max(sig.vals) - objval
stor = oe.Storage(storval, 0.95, 50)

if not os.path.isfile('med.hyb'):
    hyb = oe.HybridDia(sig, stor, obj, name='med')
    hyb.compute_parallel(cuts=[0.2, 0.5, 0.8], curves=1)
    hyb.save('med.hyb')
else:
    hyb = oe.HybridDia.load('med.hyb')

overmixed = oe.OverdimDia(sig, stor, hyb.inter[0.8], obj, name='med')
overmixed.build_mesh_arrays()
overmixed.compute_parallel()

overdim = oe.overdimdia.OverdimDiaDim(sig, stor, hyb.inter[0.8], obj,
                                      name='med')
overdim.build_mesh_arrays()
overdim.compute_parallel()

overbase = oe.overdimdia.OverdimDiaBase(sig, stor, hyb.inter[0.8], obj,
                                        name='med')
overbase.build_mesh_arrays()
overbase.compute_parallel()

overpeak = oe.overdimdia.OverdimDiaPeak(sig, stor, hyb.inter[0.8], obj,
                                        name='med')
overpeak.build_mesh_arrays()
overpeak.compute_parallel()

overenergy = oe.overdimdia.OverdimDiaEnergy(sig, stor, hyb.inter[0.8], obj,
                                            name='med')
overenergy.build_mesh_arrays()
overenergy.compute_parallel()

overpower = oe.overdimdia.OverdimDiaPower(sig, stor, hyb.inter[0.8], obj,
                                          name='med')
overpower.build_mesh_arrays()
overpower.compute_parallel()

dummybreakpoint = True
