# MIT License
#
# Copyright (c) 2018-2021 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Module responsible for computing various statistics on tree sequences.
"""
import sys
import threading

import numpy as np

import _tskit


class LdCalculator:
    """
    Class for calculating `linkage disequilibrium
    <https://en.wikipedia.org/wiki/Linkage_disequilibrium>`_ coefficients
    between pairs of sites in a :class:`TreeSequence`.

    .. note:: This interface is deprecated and a replacement is planned.
        Please see https://github.com/tskit-dev/tskit/issues/1900 for
        more information. Note also that the current implementation is
        quite limited (see warning below).

    .. warning:: This class does not currently support sites that have more than one
        mutation. Using it on such a tree sequence will raise a LibraryError with
        an "Only infinite sites mutations supported" message.

        Silent mutations are also not supported and will result in a LibraryError.

    :param TreeSequence tree_sequence: The tree sequence of interest.
    """

    def __init__(self, tree_sequence):
        self._tree_sequence = tree_sequence
        self._ll_ld_calculator = _tskit.LdCalculator(
            tree_sequence.get_ll_tree_sequence()
        )
        # To protect low-level C code, only one method may execute on the
        # low-level objects at one time.
        self._instance_lock = threading.Lock()

    def get_r2(self, a, b):
        # Deprecated alias for r2(a, b)
        return self.r2(a, b)

    def r2(self, a, b):
        """
        Returns the value of the :math:`r^2` statistic between the pair of
        sites at the specified indexes. This method is *not* an efficient
        method for computing large numbers of pairwise LD values; please use either
        :meth:`.r2_array` or :meth:`.r2_matrix` for this purpose.

        :param int a: The index of the first site.
        :param int b: The index of the second site.
        :return: The value of :math:`r^2` between the sites at indexes
            ``a`` and ``b``.
        :rtype: float
        """
        with self._instance_lock:
            return self._ll_ld_calculator.get_r2(a, b)

    def get_r2_array(self, a, direction=1, max_mutations=None, max_distance=None):
        # Deprecated alias for r2_array
        return self.r2_array(
            a,
            direction=direction,
            max_mutations=max_mutations,
            max_distance=max_distance,
        )

    def r2_array(
        self, a, direction=1, max_mutations=None, max_distance=None, max_sites=None
    ):
        """
        Returns the value of the :math:`r^2` statistic between the focal
        site at index :math:`a` and a set of other sites. The method
        operates by starting at the focal site and iterating over adjacent
        sites (in either the forward or backwards direction) until either a
        maximum number of other sites have been considered (using the
        ``max_sites`` parameter), a maximum distance in sequence
        coordinates has been reached (using the ``max_distance`` parameter) or
        the start/end of the sequence has been reached. For every site
        :math:`b` considered, we then insert the value of :math:`r^2` between
        :math:`a` and :math:`b` at the corresponding index in an array, and
        return the entire array. If the returned array is :math:`x` and
        ``direction`` is :data:`tskit.FORWARD` then :math:`x[0]` is the
        value of the statistic for :math:`a` and :math:`a + 1`, :math:`x[1]`
        the value for :math:`a` and :math:`a + 2`, etc. Similarly, if
        ``direction`` is :data:`tskit.REVERSE` then :math:`x[0]` is the
        value of the statistic for :math:`a` and :math:`a - 1`, :math:`x[1]`
        the value for :math:`a` and :math:`a - 2`, etc.

        :param int a: The index of the focal sites.
        :param int direction: The direction in which to travel when
            examining other sites. Must be either
            :data:`tskit.FORWARD` or :data:`tskit.REVERSE`. Defaults
            to :data:`tskit.FORWARD`.
        :param int max_sites: The maximum number of sites to return
            :math:`r^2` values for. Defaults to as many sites as
            possible.
        :param int max_mutations: Deprecated synonym for max_sites.
        :param float max_distance: The maximum absolute distance between
            the focal sites and those for which :math:`r^2` values
            are returned.
        :return: An array of double precision floating point values
            representing the :math:`r^2` values for sites in the
            specified direction.
        :rtype: numpy.ndarray
        """
        if max_mutations is not None and max_sites is not None:
            raise ValueError("max_mutations is a deprecated synonym for max_sites")
        if max_mutations is not None:
            max_sites = max_mutations
        max_sites = -1 if max_sites is None else max_sites
        if max_distance is None:
            max_distance = sys.float_info.max
        with self._instance_lock:
            return self._ll_ld_calculator.get_r2_array(
                a,
                direction=direction,
                max_sites=max_sites,
                max_distance=max_distance,
            )

    def get_r2_matrix(self):
        # Deprecated alias for r2_matrix
        return self.r2_matrix()

    def r2_matrix(self):
        """
        Returns the complete :math:`m \\times m` matrix of pairwise
        :math:`r^2` values in a tree sequence with :math:`m` sites.

        :return: An 2 dimensional square array of double precision
            floating point values representing the :math:`r^2` values for
            all pairs of sites.
        :rtype: numpy.ndarray
        """
        m = self._tree_sequence.num_sites
        A = np.ones((m, m), dtype=float)
        for j in range(m - 1):
            a = self.get_r2_array(j)
            A[j, j + 1 :] = a
            A[j + 1 :, j] = a
        return A


class BlockBootstrapDesign:
    """
    TODO: docstring

import tskit
import msprime
import numpy as np

model = msprime.Demography.island_model([1e4]*2, migration_rate=1e-4)
haps_per_pop = 200
ts = msprime.sim_ancestry(
    samples={"pop_0":haps_per_pop/2,"pop_1":haps_per_pop/2}, 
    recombination_rate=1e-8, 
    demography=model, 
    sequence_length=10e6,
)
sample_sets = [np.arange(haps_per_pop), np.arange(haps_per_pop, 2*haps_per_pop)]

# --- example: Fst ---
windows = np.linspace(0, ts.sequence_length, 5)
bootstrap_design = tskit.BlockBootstrapDesign(
    ts, blocks_per_window=50, windows=windows, block_by='trees'
)
bootstrap_design.num_trees_per_block
divergence = ts.divergence(
    sample_sets=sample_sets, 
    windows=bootstrap_design.breakpoints,
    mode='branch',
    span_normalise=False,
)
diversity = ts.diversity(
    sample_sets=sample_sets,
    windows=bootstrap_design.breakpoints,
    mode='branch',
    span_normalise=False,
)
bootstrapper = bootstrap_design.block_bootstrap(
    np.column_stack([divergence, diversity]), 
    reduction=lambda x: np.array([1 - 2*(x[1] + x[2])/(2*x[0] + x[1] + x[2])]),
    span_normalise=True, #inputs should *not* be span-normalised
)
    
# check correctness
bootstrapper.observed_value
ts.Fst(sample_sets=sample_sets, windows=windows, mode='branch')
    
# member functions give raw bootstrap replicates or summaries
bootstrapper.resample(num_replicates=100, random_seed=1)
    
# streaming calculations (w/o ever storing more than one replicate)
# vs generating all samples at once (faster)
bootstrapper.mean(num_replicates=100, random_seed=1)
bootstrapper.resample(100, random_seed=1).mean(axis=0)

bootstrapper.stddev(num_replicates=100, random_seed=1)
bootstrapper.resample(100, random_seed=1).std(axis=0)

# faster, but more memory intensive
    
# --- example: covariance between diversity and divergence ---
bootstrapper = bootstrap_design.block_bootstrap(
    np.column_stack([divergence, diversity]), 
    span_normalise=True,
)
# without `reduction` the per-window output is [diversity, divergence]
bootstrapper.observed_value
covar = bootstrapper.covariance(num_replicates=100, random_seed=1)
    
bootstrap_replicates = bootstrapper.resample(num_replicates=100, random_seed=1)
np.cov(bootstrap_replicates[:,0,:].T)
covar[0,:,:]

# --- example: consistency for linear statistic --- 
bootstrap_design = tskit.BlockBootstrapDesign(
    ts, blocks_per_window=200, block_by='sites'
)
bootstrap_design.num_trees_per_block
bootstrapper = bootstrap_design.block_bootstrap(
    ts.divergence(sample_sets=sample_sets, windows=bootstrap_design.breakpoints, mode='branch', span_normalise=False),
    span_normalise=True,
)
bootstrapper.observed_value
bootstrapper.



    """

    def __init__(self, ts, blocks_per_window, windows=None, block_by='trees'):

        if windows is None:
            windows = np.array([0, ts.sequence_length])
        else:
            assert isinstance(windows, np.ndarray)
            assert len(windows.shape) == 1
            assert len(windows > 1)
            assert np.min(windows) >= 0
            assert np.max(windows) <= ts.sequence_length
        self._window_breakpoints = windows
        self.num_windows = len(self._window_breakpoints) - 1

        assert isinstance(blocks_per_window, int)
        assert blocks_per_window > 0
        self._block_breakpoints = []
        tree_breakpoints = np.array([x for x in ts.breakpoints()])
        for i in range(self.num_windows):
            window = [self._window_breakpoints[i], self._window_breakpoints[i+1]]
            if block_by == 'trees':
                in_window = np.logical_and(tree_breakpoints >= window[0], tree_breakpoints < window[1])
                breakpoints = np.unique(np.append(tree_breakpoints[in_window], window))
                breakpoints = np.quantile(breakpoints, np.linspace(0, 1, blocks_per_window + 1))
            elif block_by == 'sites':
                breakpoints = np.linspace(window[0], window[1], blocks_per_window + 1)
            else:
                raise ValueError('`block_by` must be one of ["trees", "sites"]')
            self._block_breakpoints.extend(breakpoints)
        self._block_breakpoints = np.unique(self._block_breakpoints)
        self._block_span = np.diff(self._block_breakpoints)
        self.num_blocks = len(self._block_breakpoints) - 1
        self.blocks_per_window = blocks_per_window
        #print(self.blocks_per_window)
        #print(self.num_windows)
        #print(self.num_blocks)
        assert self.num_blocks == self.blocks_per_window * self.num_windows
        assert self.num_blocks == len(self._block_span)
        #TODO: this isn't correct if any tree breakpoints coincide with window breakpoints
        self.num_trees_per_block, _ = np.histogram(tree_breakpoints, self._block_breakpoints)
        self.num_trees_per_block += 1

    @property
    def breakpoints(self):
        """
        Return breakpoints of blocks in physical coordinates.
        """
        return self._block_breakpoints

    @property
    def span(self):
        """
        Return span of blocks in physical coordinates.
        """
        return self._block_span


    def block_bootstrap(self, blocked_statistics, span_normalise=True, reduction=None, random_seed=None):
        """
        Return ``BlockBootstrap`` instance using the design.
        """
        return BlockBootstrap(self, blocked_statistics, span_normalise, reduction, random_seed)


class BlockBootstrap:
    """
    TODO: docstring
    """

    def __init__(self, design, blocked_statistics, span_normalise=True, reduction=None, random_seed=None):
        assert isinstance(design, BlockBootstrapDesign)
        self.design = design
        self.span_normalise = span_normalise

        assert isinstance(blocked_statistics, np.ndarray)
        assert len(blocked_statistics.shape) == 2
        assert blocked_statistics.shape[0] == design.num_blocks
        self.blocked_statistics = blocked_statistics
        self.num_statistics = blocked_statistics.shape[1]

        if reduction is None:
            reduction = lambda x: x
        else:
            assert callable(reduction)
        test_eval = reduction(blocked_statistics[0])
        assert isinstance(test_eval, np.ndarray)
        assert test_eval.ndim == 1
        assert len(test_eval) > 0
        self.reduction = reduction
        self.num_outputs = len(test_eval)

        self.rng = np.random.default_rng(random_seed)

    def _recalculate_statistic(self, weights):
        """
        Return reduction of statistics that have been weighted by block and
        summed within windows.

        Let `X` be a vector of length `blocks_per_window * num_windows`.
        Blocks are arranged contiguously in terms of physical coordinates, so
        if the statistic of interest is a sum over blocks within each window:

        `np.sum(X.reshape((num_windows, blocks_per_window)), axis=1)` 

        A bootstrap replicate is equivalent to reweighting blocks within each
        window with integers that are draws from a uniform multinomial. To do
        this efficiently for multiple bootstrap replicates/statistics/time
        windows, rewrite as a tensor product.
        """
        assert len(weights.shape) == 3
        assert weights.shape[1] == self.design.num_windows
        assert weights.shape[2] == self.design.blocks_per_window
        assert np.all(np.sum(weights, axis=2) == self.design.blocks_per_window)

        num_replicates = weights.shape[0]

        stats = self.blocked_statistics.reshape(
            (self.design.num_windows, self.design.blocks_per_window, self.num_statistics)
        )
        block_span = self.design.span.reshape(
            (self.design.num_windows, self.design.blocks_per_window)
        )

        # want views, not copies; check that memory pointer is the same
        assert stats.__array_interface__['data'] == self.blocked_statistics.__array_interface__['data']
        assert block_span.__array_interface__['data'] == self.design.span.__array_interface__['data']

        window_span = np.einsum('ijk,jk->ij', weights, block_span)
        if not self.span_normalise:
            window_span.fill(1)

        # straightforward to extend to other dimensions, e.g. time windows
        windowed_stats = np.einsum(
            'ijk,...ij,...i->...ik', stats, weights, 1./window_span, optimize='greedy'
            #'ijk,lij,li->lik', stats, weights, 1./window_span, optimize='greedy' #would be better
        )
        assert windowed_stats.shape == (num_replicates, self.design.num_windows, self.num_statistics)
        return np.apply_along_axis(self.reduction, 2, windowed_stats).squeeze() #don't squeeze if num_replicates is not None

    def resample(self, num_replicates=1, random_seed=None):
        """
        Return `num_replicates` bootstrap replicates.
        """

        assert num_replicates > 0
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        # could "chunk" this as output size might be more reasonable than
        # size of 'bootstrap weights'. choose 'chunk size' to keep total
        # dimension bounded
        bootstrap_weights = self.rng.multinomial(
            self.design.blocks_per_window, 
            [1/self.design.blocks_per_window]*self.design.blocks_per_window,
            (num_replicates, self.design.num_windows),
        )
        return self._recalculate_statistic(bootstrap_weights)

    @property
    def observed_value(self):
        """
        The statistic calcuated from the actual data.
        """

        weights = np.ones(
            (1, self.design.num_windows, self.design.blocks_per_window)
        )
        return self._recalculate_statistic(weights)

    def mean(self, num_replicates, random_seed=None):
        """
        The mean of the bootstrap distribution of the statistic(s) within windows.
        Uses a numerically stable streaming calculation so as to work with many
        bootstrap replicates and large arrays.
        """

        assert num_replicates > 0
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        mean = np.zeros(self.observed_value.shape)
        for n in range(num_replicates):
            replicate = self.resample()
            mean += (replicate - mean)/(n + 1)
        return mean

    def variance(self, num_replicates, random_seed=None):
        """
        The variance of the bootstrap distribution of the statistic(s) within
        windows. Uses a numerically stable streaming calculation so as to work
        with many bootstrap replicates and large arrays.
        """

        assert num_replicates > 1
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        mean = np.zeros(self.observed_value.shape)
        var = np.zeros(self.observed_value.shape)
        for n in range(num_replicates):
            replicate = self.resample()
            delta = replicate - mean
            mean += delta/(n + 1)
            var += delta*(replicate - mean)
        return var / (num_replicates - 1)

    def stddev(self, num_replicates, random_seed=None):
        """
        The standard deviation of the bootstrap distribution of the statistic(s)
        within windows. Uses a numerically stable streaming calculation so as
        to work with many bootstrap replicates and large arrays.
        """

        return np.sqrt(self.variance(num_replicates, random_seed))

    def covariance(self, num_replicates, random_seed=None):
        """
        The covariance of the bootstrap distribution of a vector of statistics
        within windows. Uses a numerically stable streaming calculation so as
        to work with many bootstrap replicates and large arrays.

        .. note: The maximum possible rank of the covariance matrix is
            ``min(blocks_per_window, num_replicates, num_statistics)``. Thus,
            for high-dimensional statistics, it may not be possible to get a
            positive-definite covariance matrix.
        """

        # TODO: this could be prone to overflow with large trees and additive
        # statistics. It'd be more stable to do rank-1 updates to a LDL'
        # factorization.

        assert num_replicates > 1
        if random_seed is not None:
            self.rng = np.random.default_rng(random_seed)
        mean = np.zeros(self.observed_value.shape)
        covar = np.zeros(self.observed_value.shape + (self.num_outputs,))
        for n in range(num_replicates):
            replicate = self.resample()
            delta = replicate - mean
            mean += delta/(n + 1)
            delta_w = (replicate - mean)/(n + 1)
            covar = covar * n/(n + 1) + np.einsum('...i,...j->...ij', delta, delta_w)
        return covar * num_replicates/(num_replicates - 1)

    #def quantiles(self, quantiles, num_replicates):
    #    """
    #    TODO: will have to be approximate for streaming calculation
    #    """














