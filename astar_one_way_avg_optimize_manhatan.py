import argparse
import numpy as np
import time
import sys

import util
import searcher

if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument("--nearest_node_num", type=int)
    # 解析参数
    args = parser.parse_args()
    
    tsp_path = "/data2/lhy/project/MonaLisaTSP/mona-lisa100K.tsp.txt"
    dist_matrix_path = "/data2/lhy/project/MonaLisaTSP/mona_lisa100K_dist_matrix_manhatan.npy"
    result_path = f"/data2/lhy/project/MonaLisaTSP/result/astar_one_way_avg_manhatan_{args.nearest_node_num}_result.json"

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

    astar_searcher = searcher.AStarSearcher(points_loc, dist_matrix)
    node_order, path_length = astar_searcher.astar_search_one_way_avg(nearest_node_num=args.nearest_node_num)

    for i in range(len(node_order)):
        node_order[i] = int(node_order[i])
    path_length = float(path_length)

    result_dict = {}
    result_dict["node_order"] = node_order
    result_dict["path_length"] = path_length

    util.save_json(result_dict, result_path)
