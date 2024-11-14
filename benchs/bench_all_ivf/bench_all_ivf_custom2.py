# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import time
import json

import logging
import threading
import subprocess
import gc
import signal

import faiss
import numpy as np

try:
    import datasets_fb as datasets
except ModuleNotFoundError:
    import datasets_oss as datasets

sanitize = datasets.sanitize


def setup_logging():
    log_filename = get_unique_log_filename('benchmark.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_unique_log_filename(base_filename):
    if not os.path.exists(base_filename):
        return base_filename
    else:
        base, ext = os.path.splitext(base_filename)
        counter = 1
        while True:
            new_filename = f"{base}{counter}{ext}"
            if not os.path.exists(new_filename):
                return new_filename
            counter += 1


def record_log(stop_event):
    while not stop_event.is_set():
        with open("/proc/vmstat", "r") as vmstat_file:
            vmstat = vmstat_file.read()

        with open("/sys/devices/system/node/node0/meminfo", "r") as meminfo_node0_file:
            meminfo_node0 = meminfo_node0_file.read()

        with open("/sys/devices/system/node/node1/meminfo", "r") as meminfo_node1_file:
            meminfo_node1 = meminfo_node1_file.read()

        logging.info(f"VMSTAT:\n{vmstat}")
        logging.info(f"NODE0 MEMINFO:\n{meminfo_node0}")
        logging.info(f"NODE1 MEMINFO:\n{meminfo_node1}")
        time.sleep(10)


# Node0의 메모리를 꽉 채우기 위한 배열 선언.
mem_consume_arrays = []

def allocate_memory(gb_size):
    """
    지정된 GB 크기의 메모리를 할당하는 함수.
    :param gb_size: 할당할 메모리 크기(GB 단위)
    """
    global mem_consume_arrays
    num_elements_per_gb = (1024 ** 3) // 8  # 1GB에 해당하는 float 요소 수

    for i in range(gb_size):
        # 1GB 메모리 할당
        large_array = [0.0] * num_elements_per_gb
        mem_consume_arrays.append(large_array)

        # 리스트에 접근하여 첫 번째와 마지막 요소를 설정하고 출력
        large_array[0] = i + 1.0
        large_array[-1] = (i + 1) * 10.0

    print(f"{gb_size}GB 메모리 할당 및 접근 완료.")


def release_memory():
    """
    할당된 메모리를 해제하는 함수.
    """
    global mem_consume_arrays
    mem_consume_arrays.clear()
    gc.collect()  # 가비지 컬렉터 강제 실행
    print("모든 메모리 해제 완료.")


def enable_cores_on_node(node):
    """
    특정 NUMA 노드에 속한 모든 코어를 활성화.

    Parameters:
    - node (int): 활성화할 NUMA 노드 번호 (0 또는 1)
    """
    time.sleep(5)
    if node == 0:
        node_cpus = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    elif node == 1:
        node_cpus = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    else:
        print("Invalid NUMA node. Please specify 0 or 1.")
        return

    for cpu in node_cpus:
        try:
            # echo 명령어를 사용하여 코어 활성화
            subprocess.run(f"echo 1 | sudo tee /sys/devices/system/cpu/cpu{cpu}/online", shell=True, check=True)
            print(f"CPU {cpu} has been enabled.")
        except subprocess.CalledProcessError:
            print(f"Failed to enable CPU {cpu}.")
        except Exception as e:
            print(f"An error occurred: {e}")
    time.sleep(30)


def disable_cores_on_node(node):
    """
    특정 NUMA 노드에 속한 모든 코어를 비활성화.
    Node0의 cpu0은 비활성화 불가.

    Parameters:
    - node (int): 비활성화할 NUMA 노드 번호 (0 또는 1)
    """
    time.sleep(5)

    if node == 0:
        node_cpus = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    elif node == 1:
        node_cpus = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    else:
        print("Invalid NUMA node. Please specify 0 or 1.")
        return

    for cpu in node_cpus:
        try:
            # echo 명령어를 사용하여 코어 비활성화
            subprocess.run(f"echo 0 | sudo tee /sys/devices/system/cpu/cpu{cpu}/online", shell=True, check=True)
            print(f"CPU {cpu} has been disabled.")
        except subprocess.CalledProcessError:
            print(f"Failed to disable CPU {cpu}.")
        except Exception as e:
            print(f"An error occurred: {e}")
    time.sleep(30)


def set_numa_balancing(config_num):
    try:
        # Use the value argument in the echo command
        command = f"echo {config_num} | sudo tee /proc/sys/kernel/numa_balancing"
        subprocess.run(command, shell=True, check=True)
        print("Successfully set NUMA balancing.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set NUMA balancing: {e}")
    time.sleep(10)


def unwind_index_ivf(index):
    if isinstance(index, faiss.IndexPreTransform):
        assert index.chain.size() == 1
        vt = index.chain.at(0)
        index_ivf, vt2 = unwind_index_ivf(faiss.downcast_index(index.index))
        assert vt2 is None
        if vt is None:
            vt = lambda x: x
        else:
            vt = faiss.downcast_VectorTransform(vt)
        return index_ivf, vt
    if hasattr(faiss, "IndexRefine") and isinstance(index, faiss.IndexRefine):
        return unwind_index_ivf(faiss.downcast_index(index.base_index))
    if isinstance(index, faiss.IndexIVF):
        return index, None
    else:
        return None, None


def apply_AQ_options(index, args):
    # if not(
    #    isinstance(index, faiss.IndexAdditiveQuantize) or
    #    isinstance(index, faiss.IndexIVFAdditiveQuantizer)):
    #    return
    if args.RQ_train_default:
        print("set default training for RQ")
        index.rq.train_type
        index.rq.train_type = faiss.ResidualQuantizer.Train_default
    if args.RQ_beam_size != -1:
        print("set RQ beam size to", args.RQ_beam_size)
        index.rq.max_beam_size
        index.rq.max_beam_size = args.RQ_beam_size
    if args.LSQ_encode_ils_iters != -1:
        print("set LSQ ils iterations to", args.LSQ_encode_ils_iters)
        index.lsq.encode_ils_iters
        index.lsq.encode_ils_iters = args.LSQ_encode_ils_iters
    if args.RQ_use_beam_LUT != -1:
        print("set RQ beam LUT to", args.RQ_use_beam_LUT)
        index.rq.use_beam_LUT
        index.rq.use_beam_LUT = args.RQ_use_beam_LUT


def eval_setting(index, xq, gt, k, inter, min_time):
    """ evaluate searching in terms of precision vs. speed """
    nq = xq.shape[0]
    ivf_stats = faiss.cvar.indexIVF_stats
    ivf_stats.reset()
    nrun = 0
    t0 = time.time()
    while True:
        D, I = index.search(xq, k)
        nrun += 1
        t1 = time.time()
        if t1 - t0 > min_time:
            break
    ms_per_query = ((t1 - t0) * 1000.0 / nq / nrun)
    res = {
        "ms_per_query": ms_per_query,
        "nrun": nrun
    }
    res["n"] = ms_per_query
    if inter:
        rank = k
        inter_measure = faiss.eval_intersection(gt[:, :rank], I[:, :rank]) / (nq * rank)
        print("%.4f" % inter_measure, end=' ')
        res["inter_measure"] = inter_measure
    else:
        res["recalls"] = {}
        for rank in 1, 10, 100:
            recall = (I[:, :rank] == gt[:, :1]).sum() / float(nq)
            print("%.4f" % recall, end=' ')
            res["recalls"][rank] = recall
    print("   %9.5f  " % ms_per_query, end=' ')
    print("%12d   " % (ivf_stats.ndis / nrun), end=' ')
    print(nrun)
    res["ndis"] = ivf_stats.ndis / nrun
    return res


######################################################
# Training
######################################################

def run_train(args, ds, res):
    nq, d = ds.nq, ds.d
    nb, d = ds.nq, ds.d

    print("build index, key=", args.indexkey)

    index = faiss.index_factory(
        d, args.indexkey, faiss.METRIC_L2 if ds.metric == "L2" else
        faiss.METRIC_INNER_PRODUCT
    )

    index_ivf, vec_transform = unwind_index_ivf(index)

    if args.by_residual != -1:
        by_residual = args.by_residual == 1
        print("setting by_residual = ", by_residual)
        index_ivf.by_residual  # check if field exists
        index_ivf.by_residual = by_residual

    if index_ivf:
        print("Update add-time parameters")
        # adjust default parameters used at add time for quantizers
        # because otherwise the assignment is inaccurate
        quantizer = faiss.downcast_index(index_ivf.quantizer)
        if isinstance(quantizer, faiss.IndexRefine):
            print("   update quantizer k_factor=", quantizer.k_factor, end=" -> ")
            quantizer.k_factor = 32 if index_ivf.nlist < 1e6 else 64
            print(quantizer.k_factor)
            base_index = faiss.downcast_index(quantizer.base_index)
            if isinstance(base_index, faiss.IndexIVF):
                print("   update quantizer nprobe=", base_index.nprobe, end=" -> ")
                base_index.nprobe = (
                    16 if base_index.nlist < 1e5 else
                    32 if base_index.nlist < 4e6 else
                    64)
                print(base_index.nprobe)
        elif isinstance(quantizer, faiss.IndexHNSW):
            hnsw = quantizer.hnsw
            print(
                f"   update HNSW quantizer options, before: "
                f"{hnsw.efSearch=:} {hnsw.efConstruction=:}"
            )
            hnsw.efSearch = 40 if index_ivf.nlist < 4e6 else 64
            hnsw.efConstruction = 200
            print(f"       after: {hnsw.efSearch=:} {hnsw.efConstruction=:}")

    apply_AQ_options(index_ivf or index, args)

    if index_ivf:
        index_ivf.verbose = True
        index_ivf.quantizer.verbose = True
        index_ivf.cp.verbose = True
    else:
        index.verbose = True

    maxtrain = args.maxtrain
    if maxtrain == 0:
        if 'IMI' in args.indexkey:
            maxtrain = int(256 * 2 ** (np.log2(index_ivf.nlist) / 2))
        elif index_ivf:
            maxtrain = 50 * index_ivf.nlist
        else:
            # just guess...
            maxtrain = 256 * 100
        maxtrain = max(maxtrain, 256 * 100)
        print("setting maxtrain to %d" % maxtrain)

    try:
        xt2 = ds.get_train(maxtrain=maxtrain)
    except NotImplementedError:
        print("No training set: training on database")
        xt2 = ds.get_database()[:maxtrain]

    print("train, size", xt2.shape)
    assert np.all(np.isfinite(xt2))

    if (isinstance(vec_transform, faiss.OPQMatrix) and
            isinstance(index_ivf, faiss.IndexIVFPQFastScan)):
        print("  Forcing OPQ training PQ to PQ4")
        ref_pq = index_ivf.pq
        training_pq = faiss.ProductQuantizer(
            ref_pq.d, ref_pq.M, ref_pq.nbits
        )
        vec_transform.pq
        vec_transform.pq = training_pq

    if args.get_centroids_from == '':

        if args.clustering_niter >= 0:
            print(("setting nb of clustering iterations to %d" %
                   args.clustering_niter))
            index_ivf.cp.niter = args.clustering_niter

        if args.train_on_gpu:
            print("add a training index on GPU")
            train_index = faiss.index_cpu_to_all_gpus(
                faiss.IndexFlatL2(index_ivf.d))
            index_ivf.clustering_index = train_index

    else:
        print("Getting centroids from", args.get_centroids_from)
        src_index = faiss.read_index(args.get_centroids_from)
        src_quant = faiss.downcast_index(src_index.quantizer)
        centroids = src_quant.reconstruct_n()
        print("  centroid table shape", centroids.shape)

        if isinstance(vec_transform, faiss.VectorTransform):
            print("  training vector transform")
            vec_transform.train(xt2)
            print("  transform centroids")
            centroids = vec_transform.apply_py(centroids)

        if not index_ivf.quantizer.is_trained:
            print("  training quantizer")
            index_ivf.quantizer.train(centroids)

        print("  add centroids to quantizer")
        index_ivf.quantizer.add(centroids)
        del src_index

    t0 = time.time()
    index.train(xt2)
    res.train_time = time.time() - t0
    print("  train in %.3f s" % res.train_time)
    return index


######################################################
# Populating index
######################################################

def run_add(args, ds, index, res):
    print("adding")
    t0 = time.time()
    if args.add_bs == -1:
        assert args.split == [1, 0], "split not supported with full batch add"
        index.add(sanitize(ds.get_database()))
    else:
        totn = ds.nb // args.split[0]  # approximate
        i0 = 0
        print(f"Adding in block sizes {args.add_bs} with split {args.split}")
        for xblock in ds.database_iterator(bs=args.add_bs, split=args.split):
            i1 = i0 + len(xblock)
            print("  adding %d:%d / %d [%.3f s, RSS %d kiB] " % (
                i0, i1, totn, time.time() - t0,
                faiss.get_mem_usage_kb()))
            index.add(xblock)
            i0 = i1

    res.t_add = time.time() - t0
    print(f"  add in {res.t_add:.3f} s index size {index.ntotal}")


######################################################
# Search
######################################################

def run_search(args, ds, index, res):
    index_ivf, vec_transform = unwind_index_ivf(index)

    if args.no_precomputed_tables:
        if isinstance(index_ivf, faiss.IndexIVFPQ):
            print("disabling precomputed table")
            index_ivf.use_precomputed_table = -1
            index_ivf.precomputed_table.clear()

    if args.indexfile:
        print("index size on disk: ", os.stat(args.indexfile).st_size)

    if hasattr(index, "code_size"):
        print("vector code_size", index.code_size)

    if hasattr(index_ivf, "code_size"):
        print("vector code_size (IVF)", index_ivf.code_size)

    print("current RSS:", faiss.get_mem_usage_kb() * 1024)

    precomputed_table_size = 0
    if hasattr(index_ivf, 'precomputed_table'):
        precomputed_table_size = index_ivf.precomputed_table.size() * 4

    print("precomputed tables size:", precomputed_table_size)

    # Index is ready

    xq = sanitize(ds.get_queries())
    nq, d = xq.shape
    gt = ds.get_groundtruth(k=args.k)

    if not args.accept_short_gt:  # Deep1B has only a single NN per query
        assert gt.shape[1] == args.k

    if args.searchthreads != -1:
        print("Setting nb of threads to", args.searchthreads)
        faiss.omp_set_num_threads(args.searchthreads)
    else:
        print("nb search threads: ", faiss.omp_get_max_threads())

    ps = faiss.ParameterSpace()
    ps.initialize(index)

    parametersets = args.searchparams

    if args.inter:
        header = (
                '%-40s     inter@%3d time(ms/q)   nb distances #runs' %
                ("parameters", args.k)
        )
    else:

        header = (
                '%-40s     R@1   R@10  R@100  time(ms/q)   nb distances #runs' %
                "parameters"
        )

    res.search_results = {}
    if parametersets == ['autotune']:

        ps.n_experiments = args.n_autotune
        ps.min_test_duration = args.min_test_duration

        for kv in args.autotune_max:
            k, vmax = kv.split(':')
            vmax = float(vmax)
            print("limiting %s to %g" % (k, vmax))
            pr = ps.add_range(k)
            values = faiss.vector_to_array(pr.values)
            values = np.array([v for v in values if v < vmax])
            faiss.copy_array_to_vector(values, pr.values)

        for kv in args.autotune_range:
            k, vals = kv.split(':')
            vals = np.fromstring(vals, sep=',')
            print("setting %s to %s" % (k, vals))
            pr = ps.add_range(k)
            faiss.copy_array_to_vector(vals, pr.values)

        # setup the Criterion object
        if args.inter:
            print("Optimize for intersection @ ", args.k)
            crit = faiss.IntersectionCriterion(nq, args.k)
        else:
            print("Optimize for 1-recall @ 1")
            crit = faiss.OneRecallAtRCriterion(nq, 1)

        # by default, the criterion will request only 1 NN
        crit.nnn = args.k
        crit.set_groundtruth(None, gt.astype('int64'))

        # then we let Faiss find the optimal parameters by itself
        print("exploring operating points, %d threads" % faiss.omp_get_max_threads());
        ps.display()

        t0 = time.time()
        op = ps.explore(index, xq, crit)
        res.t_explore = time.time() - t0
        print("Done in %.3f s, available OPs:" % res.t_explore)

        op.display()

        print("Re-running evaluation on selected OPs")
        print(header)
        opv = op.optimal_pts
        maxw = max(max(len(opv.at(i).key) for i in range(opv.size())), 40)
        for i in range(opv.size()):
            opt = opv.at(i)

            ps.set_index_parameters(index, opt.key)

            print(opt.key.ljust(maxw), end=' ')
            sys.stdout.flush()

            res_i = eval_setting(index, xq, gt, args.k, args.inter, args.min_test_duration)
            res.search_results[opt.key] = res_i

    else:
        print(header)
        for param in parametersets:
            print("%-40s " % param, end=' ')
            sys.stdout.flush()
            ps.set_index_parameters(index, param)

            res_i = eval_setting(index, xq, gt, args.k, args.inter, args.min_test_duration)
            res.search_results[param] = res_i


######################################################
# Driver function
######################################################


def main():
    # logging 세팅.
    setup_logging()

    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('general options')
    aa('--nthreads', default=-1, type=int,
       help='nb of threads to use at train and add time')
    aa('--json', default=False, action="store_true",
       help="output stats in JSON format at the end")
    aa('--todo', default=["check_files"],
       choices=["train", "add", "search", "check_files"],
       nargs="+", help='what to do (check_files means decide depending on which index files exist)')

    group = parser.add_argument_group('dataset options')
    aa('--db', default='deep1M', help='dataset')
    aa('--compute_gt', default=False, action='store_true',
       help='compute and store the groundtruth')
    aa('--force_IP', default=False, action="store_true",
       help='force IP search instead of L2')
    aa('--accept_short_gt', default=False, action='store_true',
       help='work around a problem with Deep1B GT')

    group = parser.add_argument_group('index construction')
    aa('--indexkey', default='HNSW32', help='index_factory type')
    aa('--trained_indexfile', default='',
       help='file to read or write a trained index from')
    aa('--maxtrain', default=256 * 256, type=int,
       help='maximum number of training points (0 to set automatically)')
    aa('--indexfile', default='', help='file to read or write index from')
    aa('--split', default=[1, 0], type=int, nargs=2, help="database split")
    aa('--add_bs', default=-1, type=int,
       help='add elements index by batches of this size')

    group = parser.add_argument_group('IVF options')
    aa('--by_residual', default=-1, type=int,
       help="set if index should use residuals (default=unchanged)")
    aa('--no_precomputed_tables', action='store_true', default=False,
       help='disable precomputed tables (uses less memory)')
    aa('--get_centroids_from', default='',
       help='get the centroids from this index (to speed up training)')
    aa('--clustering_niter', default=-1, type=int,
       help='number of clustering iterations (-1 = leave default)')
    aa('--train_on_gpu', default=False, action='store_true',
       help='do training on GPU')

    group = parser.add_argument_group('index-specific options')
    aa('--M0', default=-1, type=int, help='size of base level for HNSW')
    aa('--RQ_train_default', default=False, action="store_true",
       help='disable progressive dim training for RQ')
    aa('--RQ_beam_size', default=-1, type=int,
       help='set beam size at add time')
    aa('--LSQ_encode_ils_iters', default=-1, type=int,
       help='ILS iterations for LSQ')
    aa('--RQ_use_beam_LUT', default=-1, type=int,
       help='use beam LUT at add time')

    group = parser.add_argument_group('searching')
    aa('--k', default=100, type=int, help='nb of nearest neighbors')
    aa('--inter', default=False, action='store_true',
       help='use intersection measure instead of 1-recall as metric')
    aa('--searchthreads', default=-1, type=int,
       help='nb of threads to use at search time')
    aa('--searchparams', nargs='+', default=['autotune'],
       help="search parameters to use (can be autotune or a list of params)")
    aa('--n_autotune', default=500, type=int,
       help="max nb of autotune experiments")
    aa('--autotune_max', default=[], nargs='*',
       help='set max value for autotune variables format "var:val" (exclusive)')
    aa('--autotune_range', default=[], nargs='*',
       help='set complete autotune range, format "var:val1,val2,..."')
    aa('--min_test_duration', default=3.0, type=float,
       help='run test at least for so long to avoid jitter')
    aa('--indexes_to_merge', default=[], nargs="*",
       help="load these indexes to search and merge them before searching")

    args = parser.parse_args()

    # numa_balancing 옵션 설정
    # config_num = 1
    # set_numa_balancing(config_num)
    # logging.info(f"Successfully set NUMA balancing to {config_num}.")

    if args.todo == ["check_files"]:
        if os.path.exists(args.indexfile):
            args.todo = ["search"]
        elif os.path.exists(args.trained_indexfile):
            args.todo = ["add", "search"]
        else:
            args.todo = ["train", "add", "search"]
        logging.info(f"Setting todo to {args.todo}")
        print("setting todo to", args.todo)

    logging.info(f"Arguments: {args}")
    print("args:", args)

    os.system('echo -n "nb processors "; '
              'cat /proc/cpuinfo | grep ^processor | wc -l; '
              'cat /proc/cpuinfo | grep ^"model name" | tail -1')

    # object to collect results
    res = argparse.Namespace()
    res.args = args.__dict__

    res.cpu_model = [
        l for l in open("/proc/cpuinfo", "r")
        if "model name" in l][0]

    logging.info("Load dataset")
    print("Load dataset")

    try:
        ds = datasets.load_dataset(
            dataset=args.db, compute_gt=args.compute_gt)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        print(f"Error loading dataset: {e}")
        return

    if args.force_IP:
        ds.metric = "IP"

    print(ds)
    logging.info(ds)

    if args.nthreads != -1:
        logging.info(f"Set nb of threads to {args.nthreads}")
        print("Set nb of threads to", args.nthreads)
        faiss.omp_set_num_threads(args.nthreads)
    else:
        print("nb threads: ", faiss.omp_get_max_threads())
        logging.info(f"nb threads: {faiss.omp_get_max_threads()}")

    # Start recording vmstat in a separate thread
    stop_event = threading.Event()
    logger_thread = threading.Thread(target=record_log, args=(stop_event,))
    logger_thread.start()

    index = None
    if "train" in args.todo:
        logging.info("Starting training index")
        print("================== Training index")
        start_time = time.time()
        try:
            index = run_train(args, ds, res)
            if args.trained_indexfile:
                logging.info(f"Storing trained index {args.trained_indexfile}")
                print("storing trained index", args.trained_indexfile)
                faiss.write_index(index, args.trained_indexfile)
        except Exception as e:
            logging.error(f"Error during training: {e}")
            print(f"Error during training: {e}")
            stop_event.set()
            logger_thread.join()
            return
        logging.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    if "add" in args.todo:
        logging.info("Starting adding vectors to index")
        print("================== Adding vectors to index")
        if not index:
            assert args.trained_indexfile
            logging.info(f"Reading trained index {args.trained_indexfile}")
            print("reading trained index", args.trained_indexfile)
            index = faiss.read_index(args.trained_indexfile)

        start_time = time.time()
        try:
            run_add(args, ds, index, res)
            if args.indexfile:
                logging.info(f"Storing index {args.indexfile}")
                print("storing", args.indexfile)
                faiss.write_index(index, args.indexfile)
        except Exception as e:
            logging.error(f"Error during adding: {e}")
            print(f"Error during adding: {e}")
            stop_event.set()
            logger_thread.join()
            return
        logging.info(f"Adding vectors completed in {time.time() - start_time:.2f} seconds")
        print(f"Adding vectors completed in {time.time() - start_time:.2f} seconds")

    if "search" in args.todo:
        print("================== Searching")

        # Node0 메모리를 가득 채우는 프로세스 실행.
        #logging.info("Searching : Consuming memory")
        #print("Searching : Consuming memory")
        #process = subprocess.Popen(['numactl', '--membind=0', 'python', 'consume_memory2.py'])
        #time.sleep(15)


        # Index 를 파일에서 메모리로 불러오기
        logging.info("Searching : Reading Index File")
        if not index:
            if args.indexfile:
                logging.info(f"Reading index {args.indexfile}")
                print("reading index", args.indexfile)
                index = faiss.read_index(args.indexfile)
            elif args.indexes_to_merge:
                logging.info(f"Merging {len(args.indexes_to_merge)} indexes")
                print(f"Merging {len(args.indexes_to_merge)} indexes")
                sz = 0
                for fname in args.indexes_to_merge:
                    logging.info(f"    reading {fname} (current size {sz})")
                    print(f"    reading {fname} (current size {sz})")
                    index_i = faiss.read_index(fname)
                    if index is None:
                        index = index_i
                    else:
                        index.merge_from(index_i, index.ntotal)
                    sz = index.ntotal
            else:
                logging.error("Provide --indexfile")
                assert False, "provide --indexfile"
        logging.info("Searching : Completed reading Index File")


        # Node0 메모리를 가득 채우는 프로세스 종료.
        #os.kill(process.pid, signal.SIGTERM)
        #process.wait()
        #logging.info("Searching : Terminated consuming memory")
        #print("Searching : Terminated consuming memory")


        # Node1 을 CXL Node와 같이 CPU 독립적으로 설정.
        logging.info(f"Searching : disable cores on node 1")
        disable_cores_on_node(1)
        logging.info(f"Searching : disabled cores on node 1")


        # numa_balancing 옵션 설정
        config_num = 2
        logging.info(f"Setting NUMA balancing to {config_num}.")
        set_numa_balancing(config_num)
        logging.info(f"Successfully set NUMA balancing to {config_num}.")


        logging.info(f"Searching : Run Search.")
        start_time = time.time()
        try:
            run_search(args, ds, index, res)
        except Exception as e:
            logging.error(f"Error during searching: {e}")
            print(f"Error during searching: {e}")
            stop_event.set()
            logger_thread.join()
            return
        logging.info(f"Searching : Search completed in {time.time() - start_time:.2f} seconds")
        print(f"Searching completed in {time.time() - start_time:.2f} seconds")

        # Node1 원상 복구 : 실험 종료에 의함.
        # enable_cores_on_node(1)

        # numa_balancing 옵션 원상 복구.
        config_num = 0
        logging.info(f"Setting NUMA balancing to {config_num}.")
        set_numa_balancing(config_num)
        logging.info(f"Successfully set NUMA balancing to {config_num}.")


    if args.json:
        logging.info(f"JSON results: {json.dumps(res.__dict__)}")
        print("JSON results:", json.dumps(res.__dict__))

    # Stop recording vmstat
    stop_event.set()
    logger_thread.join()


if __name__ == "__main__":
    main()

