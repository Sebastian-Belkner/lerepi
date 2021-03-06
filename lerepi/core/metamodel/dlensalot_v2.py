#!/usr/bin/env python

"""dlensalot.py: Contains the metamodel of the Dlensalot formalism.
"""
__author__ = "S. Belkner, J. Carron, L. Legrand"

import abc
import attr


class DLENSALOT_Concept:
    """An abstract element base type for the Dlensalot formalism."""
    __metaclass__ = abc.ABCMeta


@attr.s
class DLENSALOT_Model(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        data: 
    """
    job = attr.ib(default=-1)
    analysis = attr.ib(default=-1)
    data  = attr.ib(default=[])
    noisemodel = attr.ib(default=[])
    qerec = attr.ib(default=[])
    itrec = attr.ib(default=-1)
    madel = attr.ib(default=-1)


# TODO These could become slurm jobs via script using appropriate srun -c XX
@attr.s
class DLENSALOT_Job(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        QE_delensing:
    """
    QE_lensrec = attr.ib(default=-1)
    MAP_lensrec = attr.ib(default=-1)
    inspect_result = attr.ib(default=-1)
    map_delensing = attr.ib(default=-1)
    build_OBD = attr.ib(default=-1)
    OMP_NUM_THREADS = attr.ib(default=-1)


@attr.s
class DLENSALOT_Analysis(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
    TEMP_suffix = attr.ib(default=-1)
    K = attr.ib(default=-1)
    V = attr.ib(default=-1)
    ITMAX = attr.ib(default=-1)

    nsims_mf = attr.ib(default=-1)
    LENSRES = attr.ib(default=-1)
    Lmin = attr.ib(default=-1)
    lmax_filt = attr.ib(default=-1)
    lmax_unl = attr.ib(default=-1)
    mmax_unl = attr.ib(default=-1)
    lmax_ivf = attr.ib(default=-1)
    mmax_ivf = attr.ib(default=-1)
    lmin_ivf = attr.ib(default=-1)
    mmin_ivf = attr.ib(default=-1)
    lmax_unl = attr.ib(default=-1)
    zbounds =  attr.ib(default=-1)
    zbounds_len = attr.ib(default=-1)
    pbounds = attr.ib(default=-1)
    STANDARD_TRANSFERFUNCTION = attr.ib(default=-1)


@attr.s
class DLENSALOT_Data(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        DATA_LIBDIR: path to the data
    """
    IMIN = attr.ib(default=-1)
    IMAX = attr.ib(default=-1)
    class_parameters = attr.ib(default=-1)
    package_ = attr.ib(default=-1)
    module_ = attr.ib(default=-1)
    class_ = attr.ib(default=-1)
    beam = attr.ib(default=-1)
    lmax_transf = attr.ib(default=-1)
    nside = attr.ib(default=-1)


@attr.s
class DLENSALOT_Noisemodel(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ = attr.ib(default=-1)
    BMARG_LIBDIR = attr.ib(default=-1)
    BMARG_LCUT = attr.ib(default=-1)
    BMARG_RESCALE = attr.ib(default=-1)
    ninvjob_geometry = attr.ib(default=-1)
    lmin_tlm = attr.ib(default=-1)
    lmin_elm = attr.ib(default=-1)
    lmin_blm = attr.ib(default=-1)
    CENTRALNLEV_UKAMIN = attr.ib(default=-1)
    nlev_t = attr.ib(default=-1)
    nlev_p = attr.ib(default=-1)
    nlev_dep = attr.ib(default=-1)
    inf = attr.ib(default=-1)
    mask = attr.ib(default=-1)
    rhits_normalised = attr.ib(default=-1)
    tpl = attr.ib(default=-1)


@attr.s
class DLENSALOT_Qerec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    FILTER_QE = attr.ib(default=-1)
    CG_TOL = attr.ib(default=-1)
    ninvjob_qe_geometry = attr.ib(default=-1)
    lmax_qlm = attr.ib(default=-1)
    mmax_qlm = attr.ib(default=-1)
    chain = attr.ib(default=-1)
    QE_LENSING_CL_ANALYSIS = attr.ib(default=-1)
    overwrite_libdir = attr.ib(default=-1)


@attr.s
class DLENSALOT_Itrec(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    FILTER = attr.ib(default=-1)
    TOL = attr.ib(default=-1)
    tasks = attr.ib(default=-1)
    lenjob_geometry = attr.ib(default=-1)
    lenjob_pbgeometry = attr.ib(default=-1)
    iterator_typ = attr.ib(default=-1)
    mfvar = attr.ib(default=-1)
    soltn_cond = attr.ib(default=-1)
    stepper = attr.ib(default=-1)
    overwrite_itdir = attr.ib(default=-1)
    tasks = attr.ib(default=-1)
    dlm_mod = attr.ib(default=-1)


@attr.s
class DLENSALOT_Mapdelensing(DLENSALOT_Concept):
    """_summary_

    Args:
        DLENSALOT_Concept (_type_): _description_
    """
    edges = attr.ib(default=-1)
    IMIN = attr.ib(default=-1)
    IMAX = attr.ib(default=-1)
    dlm_mod = attr.ib(default=-1)
    iterations = attr.ib(default=-1)
    droplist = attr.ib(default=-1)
    base_mask = attr.ib(default=-1)
    nlevels = attr.ib(default=-1)
    lmax_cl = attr.ib(default=-1)
    Cl_fid = attr.ib(default=-1)
    libdir_it = attr.ib(default=-1)


@attr.s
class DLENSALOT_Chaindescriptor(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        p0: 
    """
    p0 = attr.ib(default=-1)
    p1 = attr.ib(default=-1)
    p2 = attr.ib(default=-1)
    p3 = attr.ib(default=-1)
    p4 = attr.ib(default=-1)
    p5 = attr.ib(default=-1)
    p6 = attr.ib(default=-1)
    p7 = attr.ib(default=-1)


@attr.s
class DLENSALOT_Stepper(DLENSALOT_Concept):
    """A root model element type of the Dlensalot formalism.

    Attributes:
        typ:
    """
    typ = attr.ib(default=-1)
    lmax_qlm = attr.ib(default=-1)
    mmax_qlm = attr.ib(default=-1)
    xa = attr.ib(default=-1)
    xb = attr.ib(default=-1)
