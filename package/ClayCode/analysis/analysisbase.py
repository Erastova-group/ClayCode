# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
"""Adapted analysis building blocks --- :mod:`MDAnalysis.analysis.base`
============================================================

MDAnalysis provides building blocks for creating analysis classes. One can
think of each analysis class as a "tool" that performs a specific analysis over
the trajectory frames and stores the results in the tool.

Analysis classes are derived from :class:`ClayAnalysisBase` by subclassing. This
inheritance provides a common workflow and API for users and makes many
additional features automatically available (such as frame selections and a
verbose progressbar). The important points for analysis classes are:

#. Analysis tools are Python classes derived from :class:`ClayAnalysisBase`.
#. When instantiating an analysis, the :class:`Universe` or :class:`AtomGroup`
   that the analysis operates on is provided together with any other parameters
   that are kept fixed for the specific analysis.
#. The analysis is performed with :meth:`~ClayAnalysisBase.run` method. It has a
   common set of arguments such as being able to select the frames the analysis
   is performed on. The `verbose` keyword argument enables additional output. A
   progressbar is shown by default that also shows an estimate for the
   remaining time until the end of the analysis.
#. Results are always stored in the attribute :attr:`ClayAnalysisBase.results`,
   which is an instance of :class:`Results`, a kind of dictionary that allows
   allows item access via attributes. Each analysis class decides what and how
   to store in :class:`Results` and needs to document it. For time series, the
   :attr:`ClayAnalysisBase.times` contains the time stamps of the analyzed frames.


Example of using a standard analysis tool
-----------------------------------------

For example, the :class:`MDAnalysis.analysis.rms.RMSD` performs a
root-mean-square distance analysis in the following way:

.. code-block:: python

   import MDAnalysis as mda
   from MDAnalysisTests.datafiles import TPR, XTC

   from MDAnalysis.analysis import rms

   crdin = mda.Universe(TPR, XTC)

   # (2) instantiate analysis
   rmsd = rms.RMSD(crdin, select='name CA')

   # (3) the run() method can select frames in different ways
   # run on all frames (with progressbar)
   rmsd.run(verbose=True)

   # or start, stop, and step can be used
   rmsd.run(start=2, stop=8, step=2)

   # a list of frames to run the analysis on can be passed
   rmsd.run(frames=[0,2,3,6,9])

   # a list of booleans the same length of the trajectory can be used
   rmsd.run(frames=[True, False, True, True, False, False, True, False,
                    False, True])

   # (4) analyze the results, e.g., plot
   t = rmsd.times
   x2 = rmsd.results.rmsd[:, 2]   # RMSD at column index 2, see docs

   import matplotlib.pyplot as plt
   plt.plot(t, x2)
   plt.xlabel("time (ps)")
   plt.ylabel("RMSD (Å)")


Writing new analysis tools
--------------------------

In order to write new analysis tools, derive a class from :class:`ClayAnalysisBase`
and define at least the :meth:`_single_frame` method, as described in
:class:`ClayAnalysisBase`.

.. SeeAlso::

   The chapter `Writing your own trajectory analysis`_ in the *User Guide*
   contains a step-by-step example for writing analysis tools with
   :class:`ClayAnalysisBase`.


.. _`Writing your own trajectory analysis`:
   https://userguide.mdanalysis.org/stable/examples/analysis/custom_trajectory_analysis.html


Classes
-------

The :class:`Results` and :class:`ClayAnalysisBase` classes are the essential
building blocks for almost all MDAnalysis tools in the
:mod:`MDAnalysis.analysis` module. They aim to be easily useable and
extendable.

:class:`AnalysisFromFunction` and the :func:`analysis_class` functions are
simple wrappers that make it even easier to create fully-featured analysis
tools if only the single-frame analysis function needs to be written.

"""
import os
import re
from collections import UserDict
from pathlib import Path
from typing import TypeVar, Union, Literal, Sequence, Optional
import pickle as pkl
import inspect
import logging
import itertools

import numpy as np
import pandas as pd
from MDAnalysis import coordinates
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.log import ProgressBar
from numpy.typing import NDArray

logger = logging.getLogger(Path(__file__).stem)

analysis_class = TypeVar("analysis_class")
analysis_data = TypeVar("analysis_data")


class Results(UserDict):
    r"""Container object for storing results.

    :class:`Results` are dictionaries that provide two ways by which values
    can be accessed: by dictionary key ``results["value_key"]`` or by object
    attribute, ``results.value_key``. :class:`Results` stores all results
    obtained from an analysis after calling :meth:`~ClayAnalysisBase.run()`.

    The implementation is similar to the :class:`sklearn.utils.Bunch`
    class in `scikit-learn`_.

    .. _`scikit-learn`: https://scikit-learn.org/

    Raises
    ------
    AttributeError
        If an assigned attribute has the same name as a default attribute.

    ValueError
        If a key is not of type ``str`` and therefore is not able to be
        accessed by attribute.

    Examples
    --------
    >>> from MDAnalysis.analysis.base import Results
    >>> results = Results(a=1, b=2)
    >>> results['b']
    2
    >>> results.b
    2
    >>> results.a = 3
    >>> results['a']
    3
    >>> results.c = [1, 2, 3, 4]
    >>> results['c']
    [1, 2, 3, 4]


    .. versionadded:: 2.0.0
    """

    def _validate_key(self, key):
        if key in dir(self):
            raise AttributeError(f"'{key}' is a protected dictionary " "attribute")
        elif isinstance(key, str) and not key.isidentifier():
            raise ValueError(f"'{key}' is not a valid attribute")

    def __init__(self, *args, **kwargs):
        kwargs = dict(*args, **kwargs)
        if "data" in kwargs.keys():
            raise AttributeError(f"'data' is a protected dictionary attribute")
        self.__dict__["data"] = {}
        self.update(kwargs)

    def __setitem__(self, key, item):
        self._validate_key(key)
        super().__setitem__(key, item)

    def __setattr__(self, attr, val):
        if attr == "data":
            super().__setattr__(attr, val)
        else:
            self.__setitem__(attr, val)

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError as err:
            raise AttributeError(
                "'Results' object has no " f"attribute '{attr}'"
            ) from err

    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError as err:
            raise AttributeError(
                "'Results' object has no " f"attribute '{attr}'"
            ) from err

    def __getstate__(self):
        return self.data

    def __setstate__(self, state):
        self.data = state


class AnalysisData(UserDict):
    r""":class:`UserDict` for storing analysis data.

    The class is a named dictionary to store and process analysis data.
    Default values for bin :attr:`AnalysisData.bin_step` and
    :attr:`AnalysisData.cutoff` are 0.1 and 20.

    Parameters
    ----------
    name : str
        data name
    cutoff : float, optional
        maximum value for included raw data e.g. distance from a surface
    bin_step : float, optional
        bin size in data histogram
    n_bins : int, optional
        number of bins in data histogram
    min : float, optional
        minimum value for included raw data

    Attributes
    ----------
    name : str
        data name
    cutoff : float, defaults to 20 (\AA)
        maximum value for included raw data e.g. distance from a surface
    min : float, defaults to 0.0 (\AA)
        minimum value for included raw data
    n_bins : int
        number of bins in data histogram
    bin_step : float, defaults to
        bin size in data histogram
    hist :

    edges :

    bins :

    timeseries :

    bins :

    hist2d :


    """
    _default_bin_step = 0.1
    _default_cutoff = 20

    # __slots__ = ['name', 'bins', 'timeseries', 'hist', 'edges', '_n_bins', 'n_bins', 'cutoff', 'bin_step']

    def __init__(self, name: str, cutoff=None, bin_step=None, n_bins=None, min=0.0):
        self.name = name
        assert (
            len([a for a in [cutoff, bin_step, n_bins] if a is not None]) > 1
        ), f"Must provide at least 2 arguments for cutoff {cutoff}, n_bins {n_bins}, bin_step {bin_step}"
        self._cutoff = None
        self._bin_step = None
        self._n_bins = None
        self.cutoff = cutoff
        self.bin_step = bin_step
        self.n_bins = n_bins
        self._min = float(min)
        if self.cutoff is None:
            self.cutoff = self.__class__._default_cutoff

        if self.n_bins is None and self.bin_step is None:
            # print('bin_step n_bins None')
            self.bin_step = self.__class__._default_bin_step
        elif self.bin_step is None:
            # print('bin_step None')
            self.bin_step = self.cutoff / (self.n_bins)
        elif self.n_bins is None:
            # print('n_bins None')
            self.n_bins = np.rint(self.cutoff / self.bin_step)
        logger.info(f"{name!r}:")
        logger.info(f"cutoff: {self.cutoff}")
        logger.info(f"n_bins: {self.n_bins}")
        logger.info(f"bin_step: {self.bin_step}")
        hist, edges = np.histogram(
            [-1], bins=self.n_bins, range=(self._min, self.cutoff)
        )
        hist = hist.astype(np.float64)
        hist *= 0.0
        self.hist = hist
        self.edges = edges
        self.bins = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.timeseries = []
        self.df = pd.DataFrame(index=self.bins)
        self.df.index.name = "bins"
        self.hist2d = {}

    @property
    def n_bins(self):
        return self._n_bins

    @n_bins.setter
    def n_bins(self, n_bins):
        if n_bins is not None:
            self._n_bins = int(n_bins)
            if self.cutoff / self.n_bins != self.bin_step:
                self.bin_step = self.cutoff / self.n_bins

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, cutoff):
        if cutoff is not None:
            self._cutoff = float(cutoff)
        # else:
        #     self._cutoff = self._default_cutoff

    @property
    def bin_step(self):
        return self._bin_step

    @bin_step.setter
    def bin_step(self, bin_step):
        if bin_step is not None:
            self._bin_step = float(bin_step)

    def get_hist_data(self, use_abs=True, guess_min=True):
        data = np.ravel(self.timeseries)
        if use_abs == True:
            data = np.abs(data)
        # if guess_min == True:
        #     ll = np.min(data)
        # else:
        #     ll = self._min
        hist, _ = np.histogram(data, self.edges, range=(self._min, self.cutoff))
        hist = hist / len(self.timeseries)
        self.hist[:] = hist

    def get_rel_data(self, other: analysis_data, use_abs=True, **kwargs):
        r"""Create new instance with modified data values.

        :param other:
        :type other:
        :param use_abs:
        :type use_abs:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        data = np.concatenate(
            [[np.ravel(self.timeseries)], [np.ravel(other.timeseries)]], axis=0
        )
        if use_abs == False:
            pass
        elif use_abs == True:
            data = np.abs(data)
        elif len(use_abs) == 2:
            use_abs = np.array(use_abs)
            mask = np.broadcast_to(use_abs[:, np.newaxis], data.shape)
            data[mask] = np.abs(data[mask])
        data = np.divide(data[0], data[1], where=data != 0)[0]
        if "cutoff" in kwargs.keys():
            cutoff = kwargs["cutoff"]
        else:
            cutoff = self.cutoff
        if "bin_step" in kwargs.keys():
            bin_step = kwargs["bin_step"]
        else:
            bin_step = self.bin_step
        new_data = self.__class__(
            name=f"{self.name}_{other.name}", cutoff=cutoff, bin_step=bin_step
        )
        new_data.timeseries = data
        return new_data

    def get_norm(self, n_atoms: Union[dict[int], int]):
        # if type(n_atoms) == dict:
        #     if len(n_atoms.keys()) == 1:
        #         n_atoms = n_atoms.values()[0]
        #     else:
        #         self.norm_hist = self.norm_hist.copy()
        #         for key in self.n_atoms:
        #             self.norm_hist[key] = self.norm_hist[key] / n_atoms[key]
        if type(n_atoms) == int:
            self.norm_hist = self.hist.copy()
            self.norm_hist = self.norm_hist / n_atoms



    @property
    def has_hist(self):
        if np.all(self.hist != 0):
            return True
        else:
            return False

    # def add_ts(self, data: NDArray):
    #     self.timeseries.append(data)

    def get_hist_2d(
        self,
        other: analysis_data,
        use_abs: Union[Literal[bool, Sequence[bool]]] = False,
        save: Union[Literal[False], str] = True,
    ):
        data = np.concatenate(
            [[np.ravel(self.timeseries)], [np.ravel(other.timeseries)]], axis=0
        )
        if use_abs == False:
            pass
        elif use_abs == True:
            data = np.abs(data)
        elif len(use_abs) == 2:
            use_abs = np.array(use_abs)
            mask = np.broadcast_to(use_abs[:, np.newaxis], data.shape)
            data[mask] = np.abs(data[mask])

        self.hist2d[other.name] = np.histogram2d(
            data[0],
            data[1],
            bins=(self.edges, other.edges),
            range=((self._min, self.cutoff), (other._min, other.cutoff)),
        )
        if type(save) == str:
            outname = f"{save}_{self.name}_{other.name}"
            with open(f"{outname}.p", "wb") as outfile:
                pkl.dump(self.hist2d, outfile)
        return self.hist2d[other.name][0]

    def get_df(self):
        if not hasattr(self, "hist"):
            self.get_hist_data()
        self.df[self.name] = self.hist
        if hasattr(self, "norm_hist"):
            self.df[f"{self.name}_rel"] = self.norm_hist

    def __repr__(self):
        return (
            f"AnalysisData({self.name!r}, "
            f"edges = ({self._min}, {self.cutoff}), "
            f"bin_step = {self.bin_step}, "
            f"n_bins = {self.n_bins}, "
            f"has_data = {self.has_hist})"
        )

    def save(self, savename, rdf=False, **kwargs):
        if rdf is True:
            self.df[self.name] = self.df[f"{self.name}_rel"]
            df_str = self.df[self.name].reset_index().to_string(index=False)
        else:
            df_str = (
                self.df.loc[:, [self.name, f"{self.name}_rel"]]
                .reset_index()
                .to_string(index=False)
            )
        outname = f"{savename}_{self.name}"
        with open(f"{outname}.dat", "w") as outfile:
            outfile.write(
                f"# {savename}: {self.name}\n"
                "# --------------------------\n"
                f"# bin_step: {self.bin_step} A\n"
                f"# n_bins: {self.n_bins}\n"
                f"# cutoff: {self.cutoff} A\n"
            )
            for k, v in kwargs.items():
                v = str(v)
                v = re.sub('\n', '\n# ', v, flags=re.MULTILINE)
                outfile.write(f"# {k}: {v}\n")
            outfile.write("# --------------------------\n" f"# {df_str}")
        with open(f"{outname}.p", "wb") as outfile:
            pkl.dump(self, outfile)
        logger.info(f"Wrote output for {outname!r}")


class ClayAnalysisBase(object):
    r"""Base class for defining multi-frame analysis

    The class is designed as a template for creating multi-frame analyses.
    This class will automatically take care of setting up the trajectory
    reader for iterating, and it offers to show a progress meter.
    Computed results are stored inside the :attr:`results` attribute.

    To define a new Analysis, :class:`ClayAnalysisBase` needs to be subclassed
    and :meth:`_single_frame` must be defined. It is also possible to define
    :meth:`_prepare` and :meth:`_conclude` for pre- and post-processing.
    All results should be stored as attributes of the :class:`Results`
    container.

    Parameters
    ----------
    trajectory : MDAnalysis.coordinates.base.ReaderBase
        A trajectory Reader
    verbose : bool, optional
        Turn on more logging and debugging

    Attributes
    ----------
    times: numpy.ndarray
        array of Timestep times. Only exists after calling
        :meth:`ClayAnalysisBase.run`
    frames: numpy.ndarray
        array of Timestep frame indices. Only exists after calling
        :meth:`ClayAnalysisBase.run`
    results: :class:`Results`
        results of calculation are stored after call
        to :meth:`ClayAnalysisBase.run`


    Example
    -------
    .. code-block:: python

       from MDAnalysis.analysis.base import ClayAnalysisBase

       class NewAnalysis(ClayAnalysisBase):
           def __init__(self, atomgroup, parameter, **kwargs):
               super(NewAnalysis, self).__init__(atomgroup.universe.trajectory,
                                                 **kwargs)
               self._parameter = parameter
               self._ag = atomgroup

           def _prepare(self):
               # OPTIONAL
               # Called before iteration on the trajectory has begun.
               # AnalysisData structures can be set up at this time
               self.results.example_result = []

           def _single_frame(self):
               # REQUIRED
               # Called after the trajectory is moved onto each new frame.
               # store an example_result of `some_function` for a single frame
               self.results.example_result.append(some_function(self._ag,
                                                                self._parameter))

           def _conclude(self):
               # OPTIONAL
               # Called once iteration on the trajectory is finished.
               # Apply normalisation and averaging to results here.
               self.results.example_result = np.asarray(self.example_result)
               self.results.example_result /=  np.sum(self.result)

    Afterwards the new analysis can be run like this

    .. code-block:: python

       import MDAnalysis as mda
       from MDAnalysisTests.datafiles import PSF, DCD

       crdin = mda.Universe(PSF, DCD)

       na = NewAnalysis(crdin.select_atoms('name CA'), 35)
       na.run(start=10, stop=20)
       print(na.results.example_result)
       # results can also be accessed by key
       print(na.results["example_result"])


    .. versionchanged:: 1.0.0
        Support for setting `start`, `stop`, and `step` has been removed. These
        should now be directly passed to :meth:`ClayAnalysisBase.run`.

    .. versionchanged:: 2.0.0
        Added :attr:`results`

    .. note::

        Changes made to :class:`MDAnalysis.analysis.base.AnalysisBase`:

        #. added :meth:`ClayAnalysisBase._init_data` for named result
           initialisation as :class:`AnalysisData`
        #. :class:`ClayAnalysisBase` has modified :meth:`AnalysisBase._conclude` method:
           gets absolute values of data if selected and uses :meth:`AnalysisData.get_hist_data`,
           :meth:`AnalysisData.get_norm` and :meth:`AnalysisData.get_df`
        #. added :meth:`ClayAnalysisBase._get_results` to include instance attributes
           in `results` dictionary and save as pickle if specified.
        #. :class:`ClayAnalysisBase` has modified :meth:`AnalysisBase.run` method:
           added :meth:`ClayAnalysisBase._get_results` and :meth:`ClayAnalysisBase._save` to workflow.
        #. added abstract :meth:`ClayAnalysisBase.save` method

    """
    # histogram attributes format:
    # --------------------------
    # name: [name, bins, timeseries, hist, hist2d, edges, n_bins, cutoff, bin_step]

    _attrs = []

    def __init__(self, trajectory, verbose=False, **kwargs):
        self._trajectory = trajectory
        self._verbose = verbose
        self.results = Results()
        self.sel_n_atoms = None
        self._abs = True

    def _init_data(self, **kwargs):
        data = self.__class__._attrs
        if len(data) == 0:
            data = [self.__class__.__name__.lower()]
        self.data = {}
        for item in data:
            if item in kwargs.keys():
                args = kwargs[item]
            else:
                args = kwargs
            self.data[item] = AnalysisData(item, **args)

    def _setup_frames(self, trajectory, start=None, stop=None, step=None, frames=None):
        """Pass a Reader object and define the desired iteration pattern
        through the trajectory

        Parameters
        ----------
        trajectory : mda.Reader
            A trajectory Reader
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        frames : array_like, optional
            array of integers or booleans to slice trajectory; cannot be
            combined with `start`, `stop`, `step`

            .. versionadded:: 2.2.0

        Raises
        ------
        ValueError
            if *both* `frames` and at least one of `start`, `stop`, or `frames`
            is provided (i.e., set to another value than ``None``)


        .. versionchanged:: 1.0.0
            Added .frames and .times arrays as attributes

        .. versionchanged:: 2.2.0
            Added ability to iterate through trajectory by passing a list of
            frame indices in the `frames` keyword argument

        """
        self._trajectory = trajectory
        if frames is not None:
            if not all(opt is None for opt in [start, stop, step]):
                raise ValueError("start/stop/step cannot be combined with frames")
            slicer = frames
        else:
            start, stop, step = trajectory.check_slice_indices(start, stop, step)
            slicer = slice(start, stop, step)
        self._sliced_trajectory = trajectory[slicer]
        self.start = start
        self.stop = stop
        self.step = step
        self.n_frames = len(self._sliced_trajectory)
        self.frames = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)

    def _single_frame(self):
        """Calculate data from a single frame of trajectory

        Don't worry about normalising, just deal with a single frame.
        """
        raise NotImplementedError("Only implemented in child classes")

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        pass  # pylint: disable=unnecessary-pass

    def _get_2d_hist(self, second: str):
        """Second data series for 2d histogram."""
        ...

    def _conclude(self, n_atoms: Optional[int] = None, **kwargs):
        """Finalize the results you've gathered.

        Called at the end of the :meth:`run` method to finish everything up.
        """
        # pass  # pylint: disable=unnecessary-pass
        # print(self._abs, n_atoms)
        # if n_atoms is None:
        #     n_atoms = self.sel_n_atoms
        # else:
        #     n_atoms = int(n_atoms)
        if type(self._abs) == bool:
            self._abs = [self._abs for a in range(len(self.data))]
            # print(self._abs, len(self.data))
        for vi, v in enumerate(self.data.values()):
            v.get_hist_data(use_abs=self._abs[vi])
            v.get_norm(self.sel_n_atoms)
            v.get_df()

    def _get_results(self):
        """Finalize the results you've gathered.

        Called at the end of the :meth:`run` method to finish everything up.
        """
        self.results = {}
        for key, val in self.__dict__.items():
            if not key.startswith("_") and key != "results":
                self.results[key] = val
        logger.info(f"{self.save}")
        if self.save is False:
            pass
        else:
            outdir = Path(self.save).parent
            logger.info(f"Saving results in {str(outdir.absolute())!r}")
            if not outdir.is_dir():
                os.mkdir(outdir)
                logger.info(f"Created {outdir}")
            with open(f"{self.save}.p", "wb") as outfile:
                pkl.dump(self.results, outfile)

    def run(self, start=None, stop=None, step=None, frames=None, verbose=True):
        """Perform the calculation

        Parameters
        ----------
        start : int, optional
            start frame of analysis
        stop : int, optional
            stop frame of analysis
        step : int, optional
            number of frames to skip between each analysed frame
        frames : array_like, optional
            array of integers or booleans to slice trajectory; `frames` can
            only be used *instead* of `start`, `stop`, and `step`. Setting
            *both* `frames` and at least one of `start`, `stop`, `step` to a
            non-default value will raise a :exc:`ValueError`.

            .. versionadded:: 2.2.0

        verbose : bool, optional
            Turn on verbosity


        .. versionchanged:: 2.2.0
            Added ability to analyze arbitrary frames by passing a list of
            frame indices in the `frames` keyword argument.

        """
        logger.info("Choosing frames to analyze")
        # if verbose unchanged, use class default
        verbose = getattr(self, "_verbose", False) if verbose is None else verbose

        self._setup_frames(
            self._trajectory, start=start, stop=stop, step=step, frames=frames
        )
        logger.info("Starting preparation")
        self._prepare()
        logger.info("Starting analysis loop over %d trajectory frames", self.n_frames)
        for i, ts in enumerate(ProgressBar(self._sliced_trajectory, verbose=verbose)):
            self._frame_index = i
            self._ts = ts
            self.frames[i] = ts.frame
            self.times[i] = ts.time
            self._single_frame()
        logger.info("Finishing up")
        self._conclude(self.sel_n_atoms)
        logger.info(f"Getting results")
        self._get_results()
        self._save()
        return self

    def _save(self):
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data: {self._attrs}, "
            f"loaded frames: {self.n_frames})"
        )


class AnalysisFromFunction(ClayAnalysisBase):
    r"""Create an :class:`ClayAnalysisBase` from a function working on AtomGroups

    Parameters
    ----------
    function : callable
        function to evaluate at each frame
    trajectory : MDAnalysis.coordinates.Reader, optional
        trajectory to iterate over. If ``None`` the first AtomGroup found in
        args and kwargs is used as a source for the trajectory.
    *args : list
        arguments for `function`
    **kwargs : dict
        arguments for `function` and :class:`ClayAnalysisBase`

    Attributes
    ----------
    results.frames : numpy.ndarray
            simulation frames used in analysis
    results.times : numpy.ndarray
            simulation times used in analysis
    results.timeseries : numpy.ndarray
            Results for each frame of the wrapped function,
            stored after call to :meth:`AnalysisFromFunction.run`.

    Raises
    ------
    ValueError
        if `function` has the same `kwargs` as :class:`ClayAnalysisBase`

    Example
    -------
    .. code-block:: python

        def rotation_matrix(mobile, ref):
            return mda.analysis.align.rotation_matrix(mobile, ref)[0]

        rot = AnalysisFromFunction(rotation_matrix, trajectory,
                                    mobile, ref).run()
        print(rot.results.timeseries)


    .. versionchanged:: 1.0.0
        Support for directly passing the `start`, `stop`, and `step` arguments
        has been removed. These should instead be passed to
        :meth:`AnalysisFromFunction.run`.

    .. versionchanged:: 2.0.0
        Former :attr:`results` are now stored as :attr:`results.timeseries`
    """

    def __init__(self, function, trajectory=None, *args, **kwargs):
        if (trajectory is not None) and (
            not isinstance(trajectory, coordinates.base.ProtoReader)
        ):
            args = (trajectory,) + args
            trajectory = None

        if trajectory is None:
            # all possible places to find trajectory
            for arg in itertools.chain(args, kwargs.values()):
                if isinstance(arg, AtomGroup):
                    trajectory = arg.universe.trajectory
                    break

        if trajectory is None:
            raise ValueError("Couldn't find a trajectory")

        self.function = function
        self.args = args

        self.kwargs = kwargs

        super(AnalysisFromFunction, self).__init__(trajectory)

    def _prepare(self):
        self.results.timeseries = []

    def _single_frame(self):
        self.results.timeseries.append(self.function(*self.args, **self.kwargs))

    def _conclude(self):
        self.results.frames = self.frames
        self.results.times = self.times
        self.results.timeseries = np.asarray(self.results.timeseries)


def analysis_class(function):
    r"""Transform a function operating on a single frame to an
    :class:`ClayAnalysisBase` class.

    Parameters
    ----------
    function : callable
        function to evaluate at each frame

    Attributes
    ----------
    results.frames : numpy.ndarray
            simulation frames used in analysis
    results.times : numpy.ndarray
            simulation times used in analysis
    results.timeseries : numpy.ndarray
            Results for each frame of the wrapped function,
            stored after call to :meth:`AnalysisFromFunction.run`.

    Raises
    ------
    ValueError
        if `function` has the same `kwargs` as :class:`ClayAnalysisBase`

    Examples
    --------

    For use in a library, we recommend the following style

    .. code-block:: python

        def rotation_matrix(mobile, ref):
            return mda.analysis.align.rotation_matrix(mobile, ref)[0]
        RotationMatrix = analysis_class(rotation_matrix)

    It can also be used as a decorator

    .. code-block:: python

        @analysis_class
        def RotationMatrix(mobile, ref):
            return mda.analysis.align.rotation_matrix(mobile, ref)[0]

        rot = RotationMatrix(crdin.trajectory, mobile, ref).run(step=2)
        print(rot.results.timeseries)


    .. versionchanged:: 2.0.0
        Former :attr:`results` are now stored as :attr:`results.timeseries`
    """

    class WrapperClass(AnalysisFromFunction):
        def __init__(self, trajectory=None, *args, **kwargs):
            super(WrapperClass, self).__init__(function, trajectory, *args, **kwargs)

    return WrapperClass


def _filter_baseanalysis_kwargs(function, kwargs):
    """
    Create two dictionaries with `kwargs` separated for `function` and
    :class:`ClayAnalysisBase`

    Parameters
    ----------
    function : callable
        function to be called
    kwargs : dict
        keyword argument dictionary

    Returns
    -------
    base_args : dict
        dictionary of ClayAnalysisBase kwargs
    kwargs : dict
        kwargs without ClayAnalysisBase kwargs

    Raises
    ------
    ValueError
        if `function` has the same `kwargs` as :class:`ClayAnalysisBase`

    """
    try:
        # pylint: disable=deprecated-method
        base_argspec = inspect.getfullargspec(ClayAnalysisBase.__init__)
    except AttributeError:
        # pylint: disable=deprecated-method
        base_argspec = inspect.getargspec(ClayAnalysisBase.__init__)

    n_base_defaults = len(base_argspec.defaults)
    base_kwargs = {
        name: val
        for name, val in zip(
            base_argspec.args[-n_base_defaults:], base_argspec.defaults
        )
    }

    try:
        # pylint: disable=deprecated-method
        argspec = inspect.getfullargspec(function)
    except AttributeError:
        # pylint: disable=deprecated-method
        argspec = inspect.getargspec(function)

    for base_kw in base_kwargs.keys():
        if base_kw in argspec.args:
            raise ValueError(
                "argument name '{}' clashes with ClayAnalysisBase argument."
                "Now allowed are: {}".format(base_kw, base_kwargs.keys())
            )

    base_args = {}
    for argname, default in base_kwargs.items():
        base_args[argname] = kwargs.pop(argname, default)

    return base_args, kwargs
