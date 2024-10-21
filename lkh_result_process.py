import re
import util
import numpy as np
import time


if __name__ == "__main__":
    lkh_result_txt = "/data3/lhy/project/MonaLisaTSP/result/lkh_init.txt"
    tsp_path = "/data3/lhy/project/MonaLisaTSP/mona-lisa100K.tsp.txt"
    dist_matrix_path = "/data3/lhy/project/MonaLisaTSP/mona_lisa100K_dist_matrix.npy"
    result_path = "/data3/lhy/project/MonaLisaTSP/result/lkh_result_init.json"
    
    result_dict = {}

    with open(lkh_result_txt, "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        lines = [int(line)-1 for line in lines if re.match(r'^\d', line)]
        # print(len(lines))
    node_order = lines
    result_dict["node_order"] = node_order

    points_loc = util.read_tsp_txt(tsp_path)[0]
    util.draw_path(node_order, points_loc, "/data3/lhy/project/MonaLisaTSP/img/lkh_init.jpg")

    dist_matrix = util.load_dist_matrix(dist_matrix_path)
    total_len = util.cal_total_length(node_order, dist_matrix)
    print("总长度为：", total_len)
    result_dict["path_length"] = total_len

    # 计算hn
    hn_list = [0] * len(node_order) # 初始化hn列表
    rest_len = total_len    # 初始化剩余长度
    for i in range(len(node_order)):
        node_index = node_order[i]
        hn = rest_len
        if i != len(node_order)-1:
            rest_len -= dist_matrix[node_order[i]][node_order[i+1]]
        hn_list[i] = hn
    print(hn_list)
    result_dict["hn_list"] = hn_list    

    util.save_json(result_dict, result_path)

