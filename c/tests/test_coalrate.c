/*
 * MIT License
 *
 * Copyright (c) 2019-2024 Tskit Developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "testlib.h"

#include <unistd.h>
#include <stdlib.h>
#include <float.h>

/* Check diversity/divergence from coalescence counts against stats implm */
static void
verify_pair_coalescence_counts(tsk_treeseq_t *ts, tsk_size_t num_time_windows,
    double *time_windows, tsk_flags_t options)
{
    int ret;
    const tsk_size_t n = tsk_treeseq_get_num_samples(ts);
    const tsk_size_t N = tsk_treeseq_get_num_nodes(ts);
    const tsk_size_t T = tsk_treeseq_get_num_trees(ts);
    const tsk_id_t *samples = tsk_treeseq_get_samples(ts);
    const double *breakpoints = tsk_treeseq_get_breakpoints(ts);
    const tsk_size_t P = 2;
    const tsk_size_t I = P * (P + 1) / 2;
    tsk_size_t sample_set_sizes[P];
    tsk_id_t index_tuples[2 * I];
    tsk_size_t dim = T * N * I;
    double C1[dim]; //, C2[dim];
    tsk_size_t i, j, k;

    for (i = 0; i < P; i++) {
        sample_set_sizes[i] = 0;
    }
    for (j = 0; j < n; j++) {
        i = j / (n / P);
        sample_set_sizes[i]++;
    }

    for (j = 0, i = 0; j < P; j++) {
        for (k = j; k < P; k++) {
            index_tuples[i++] = (tsk_id_t) j;
            index_tuples[i++] = (tsk_id_t) k;
        }
    }

    ret = tsk_treeseq_pair_coalescence_stat(ts, P, sample_set_sizes, samples, I,
        index_tuples, T, breakpoints, num_time_windows, time_windows, options, C1);
    CU_ASSERT_EQUAL_FATAL(ret, 0);

    /* TODO: check against stats implm here, can only do with no time windows */
}

static void
test_pair_coalescence_counts(void)
{
    tsk_treeseq_t ts;
    tsk_treeseq_from_text(&ts, 100, nonbinary_ex_nodes, nonbinary_ex_edges, NULL,
        nonbinary_ex_sites, nonbinary_ex_mutations, NULL, NULL, 0);
    double max_time = tsk_treeseq_get_max_time(&ts);
    double time_windows[3] = { 0.0, max_time / 2, INFINITY };
    verify_pair_coalescence_counts(&ts, 0, NULL, 0);
    verify_pair_coalescence_counts(&ts, 0, NULL, TSK_STAT_SPAN_NORMALISE);
    verify_pair_coalescence_counts(&ts, 2, time_windows, 0);
    verify_pair_coalescence_counts(&ts, 2, time_windows, TSK_STAT_SPAN_NORMALISE);
    tsk_treeseq_free(&ts);
}

int
main(int argc, char **argv)
{
    CU_TestInfo tests[] = {
        { "test_pair_coalescence_counts", test_pair_coalescence_counts },
        { NULL, NULL },
    };
    return test_main(tests, argc, argv);
}
