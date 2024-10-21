import numpy as np
import time
import sys

import util
import searcher

tsp_path = "/data3/lhy/project/MonaLisaTSP/mona-lisa100K.tsp.txt"
dist_matrix_path = "/data3/lhy/project/MonaLisaTSP/mona_lisa100K_dist_matrix.npy"
result_path = "/data3/lhy/project/MonaLisaTSP/result/ga_result.json"

load_matrix_time_start = time.time()
print("Loading the distance matrix...")
sys.stdout.flush()
dist_matrix = np.load(dist_matrix_path)
dist_matrix = util.upper_matrix_to_symmetric_matrix(dist_matrix)
load_matrix_time_end = time.time()
load_matrix_time = load_matrix_time_end - load_matrix_time_start
print(f"Load matrix took: {int(load_matrix_time//60):d}min {(load_matrix_time%60):.3f}s")
sys.stdout.flush()

points_loc = util.read_tsp_txt(tsp_path)[0]

ga_test = searcher.GASearcher(points_loc, dist_matrix)
best_pop, best_fit, best_fit_list = ga_test.ga_optimize()

result_dict = {}
result_dict["node_order"] = best_pop
result_dict["path_length"] = best_fit
result_dict["train_fits_list"] = best_fit_list
util.save_json(result_dict, result_path)
