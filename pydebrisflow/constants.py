from __future__ import annotations

"""Indices and integer identifiers shared by the solver modules."""

# Conservative-state indices.
HF = 0       # mobile fluid volume depth [m]
HSU = 1      # total solid volume depth in upper layer [m]
HCU = 2      # coarse solid volume depth in upper layer [m]
HSL = 3      # total solid volume depth in lower layer [m]
HCL = 4      # coarse solid volume depth in lower layer [m]
MX = 5       # mixture mass momentum x [kg m-1 s-1]
MY = 6       # mixture mass momentum y [kg m-1 s-1]
NV = 7

BC_OUTFLOW = 0
BC_REFLECTIVE = 1
BC_PERIODIC = 2

FLUX_HLL = 0
FLUX_HLLC = 1

LIMITER_MINMOD = 0
LIMITER_MC = 1
LIMITER_VANLEER = 2
LIMITER_SUPERBEE = 3