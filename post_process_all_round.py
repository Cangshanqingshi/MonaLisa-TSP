# 这个代码是针对已经有的A*搜索结果进行后处理

# 思路：
# 首先计算所有线段的长度
# 长度在3σ以外的都视为异常点
# 遍历异常点之间的组合找到最小的点

import numpy as np
import time
import sys

import util
import searcher


# 定义计算路径长度的函数
def calculate_path_length(order, dist_matrix):
    length = 0
    for i in range(len(order)):
        if i < len(order) - 1:
            length += dist_matrix[order[i]][order[i + 1]]
        else:
            length += dist_matrix[order[i]][order[0]]  # 回到起点
    return length


# 根据值在列表里找索引
def find_index(list, value):
    my_list = np.array(list)
    index_array = np.where(my_list == value)[0]  # 返回满足条件的所有索引
    index = index_array[0]

    return index

# 更新异常线段
def update_abnormal_segments(current_order, dist_matrix):
    # 实现一个函数来更新异常线段列表
    abnormal_segments = []
    lengths = []
    for i in range(len(current_order)):
        if i < len(current_order) - 1:
            length = dist_matrix[current_order[i]][current_order[i + 1]]
        else:
            length = dist_matrix[current_order[i]][current_order[0]]

        lengths.append(length)

    mean_length = np.mean(lengths)
    std_dev = np.std(lengths)
    threshold = mean_length + 3 * std_dev

    for i, length in enumerate(lengths):
        if lengths[i] > threshold:
            if i < len(current_order) - 1: # 如果不是最后一个的话
                line_segment_info = [current_order[i], current_order[i+1], dist_matrix[current_order[i]][current_order[i+1]]]
            else:
                line_segment_info = [current_order[i], current_order[0], dist_matrix[current_order[i]][current_order[0]]]
            
            abnormal_segments.append((i, line_segment_info))

    return abnormal_segments


# 模拟退火算法的实现
# 异常线段有一百条，也就是有两百个点，C_200^2=19900个组合，也就是至少需要迭代20000次，基本上初步测试的是一秒20代，所以这里迭代184197代
def simulated_annealing(initial_order, dist_matrix, abnormal_segments, initial_temp=1e5, cooling_rate=0.9999, stopping_temp=1e-3):
    start_time = time.time()
    current_order = initial_order.copy()
    current_length = calculate_path_length(current_order, dist_matrix)

    temperature = initial_temp
    iteration = 0  # 初始化迭代计数器

    while temperature > stopping_temp:
        # 随机选择两个异常线段进行交换
        idx1, seg1 = abnormal_segments[np.random.randint(len(abnormal_segments))]
        # 随机选择第二条异常线段，直到其端点与第一条线段的端点不重复
        idx2, seg2 = abnormal_segments[np.random.randint(len(abnormal_segments))]
        while len(set([seg1[0], seg1[1], seg2[0], seg2[1]])) != 4:
            idx2, seg2 = abnormal_segments[np.random.randint(len(abnormal_segments))]

        # 交换端点
        new_order = current_order.copy()
        if idx1 < len(current_order) - 1 and idx2 < len(current_order) - 1:
            new_order[idx1 + 1], new_order[idx2 + 1] = new_order[idx2 + 1], new_order[idx1 + 1]
        # 之前筛选的时候已经保证了四个端点不重复，所以只可能有一个点在队列最后。
        elif idx1 == len(current_order) - 1:
            new_order[0], new_order[idx2 + 1] = new_order[idx2 + 1], new_order[0]
        elif idx2 == len(current_order) - 1:
            new_order[idx1 + 1], new_order[0] = new_order[0], new_order[idx1 + 1]
        
        new_length = calculate_path_length(new_order, dist_matrix)

        # 计算长度差
        delta_length = new_length - current_length

        if delta_length < 0:
            current_order = new_order,
            current_length = new_length
            # 更新异常线段
            abnormal_segments = update_abnormal_segments(current_order, dist_matrix)

        # 降低温度
        temperature *= cooling_rate

        # 输出当前迭代次数和路径长度
        if iteration % 1000 == 0:  # 每1000次迭代输出一次
            spend_time = time.time() - start_time
            # 转换成小时、分钟和秒
            hours = spend_time // 3600
            minutes = (spend_time % 3600) // 60
            seconds = int(spend_time % 60)
            print(f"Iteration {iteration}: Path Length = {current_length:.5f}, Time = {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            sys.stdout.flush()
        iteration += 1

    return current_order, current_length


# 2-opt优化，把异常点和异常点四分之一范围内的点交换顺序，如果有减小就保留
def two_opt_pruned(initial_order, dist_matrix, abnormal_segments, threshold = 5000):
    start_time = time.time()
    iteration = 0
    current_order = initial_order.copy()
    current_length = calculate_path_length(current_order, dist_matrix)
    
    # 遍历所有异常线段的索引
    for idx, seg_info in abnormal_segments:
        # 选定阈值
        current_threshold = max(threshold, seg_info[2])
        # 获取异常线段的两个端点索引
        p1_idx = idx
        p2_idx = (idx + 1) % len(current_order)
        p1, p2 = current_order[p1_idx], current_order[p2_idx]
        # 查找与 p1 距离小于阈值的节点编号
        nearby_indices_p1 = [i for i in range(len(dist_matrix)) if dist_matrix[p1][i] < current_threshold and i != p1 and i != p2]
        # 查找与 p2 距离小于阈值的节点编号
        nearby_indices_p2 = [i for i in range(len(dist_matrix)) if dist_matrix[p2][i] < current_threshold and i != p1 and i != p2]
        
        # 对找到的索引进行 2-opt 翻转操作
        for j in nearby_indices_p1:
            j_idx = find_index(current_order, j)    # 找到对应编号在节点顺序中的位置
            new_order = current_order.copy()
            new_order[p1_idx], new_order[j_idx] = new_order[j_idx], new_order[p1_idx]
            new_length = calculate_path_length(new_order, dist_matrix)
            # 如果新路径更短，则更新路径
            if new_length < current_length:
                current_order = new_order
                current_length = new_length
                print("Current Length:", current_length)
                sys.stdout.flush()
        
        # 对找到的索引进行 2-opt 翻转操作
        for j in nearby_indices_p2:
            j_idx = find_index(current_order, j)    # 找到对应编号在节点顺序中的位置
            new_order = current_order.copy()
            new_order[p2_idx], new_order[j_idx] = new_order[j_idx], new_order[p2_idx]
            new_length = calculate_path_length(new_order, dist_matrix)
            # 如果新路径更短，则更新路径
            if new_length < current_length:
                current_order = new_order
                current_length = new_length
                print("Current Length:", current_length)
                sys.stdout.flush()
        
        spend_time = time.time() - start_time
        iteration += 1
        # 转换成小时、分钟和秒
        hours = spend_time // 3600
        minutes = (spend_time % 3600) // 60
        seconds = int(spend_time % 60)
        print(f"Iteration {iteration}: Path Length = {current_length:.5f}, Time = {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        sys.stdout.flush()

    return current_order, current_length


if __name__ == "__main__":
    tsp_path = "/data2/lhy/project/MonaLisaTSP/mona-lisa100K.tsp.txt"
    dist_matrix_path = "/data2/lhy/project/MonaLisaTSP/mona_lisa100K_dist_matrix.npy"
    raw_result_path = f"/data2/lhy/project/MonaLisaTSP/result/astar_one_way_avg_20_result.json"
    result_path = f"/data2/lhy/project/MonaLisaTSP/result/result_all_round.json"
    img_path = f"/data2/lhy/project/MonaLisaTSP/img/result_all_round.jpg"

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
    json_data = util.load_json(raw_result_path)
    node_order, path_length = json_data["node_order"], json_data["path_length"]

    path_information = []   # 路径信息的列表，其中存储每一个线段的信息，包含：起始点、结束点、长度
    for i in range(len(node_order)):
        if i < len(node_order) - 1: # 如果不是最后一个的话
            line_segment_info = [node_order[i], node_order[i+1], dist_matrix[node_order[i]][node_order[i+1]]]
        else:
            line_segment_info = [node_order[i], node_order[0], dist_matrix[node_order[i]][node_order[0]]]
        
        path_information.append(line_segment_info)

    # 计算所有线段的长度
    lengths = [segment[2] for segment in path_information]

    # 计算均值和标准差
    mean_length = np.mean(lengths)
    std_dev_length = np.std(lengths)

    # 确定异常线段的阈值
    threshold = mean_length + 3 * std_dev_length

    # 找出异常线段
    abnormal_segments = []
    for idx, length in enumerate(lengths):
        if length > threshold:
            abnormal_segments.append((idx, path_information[idx]))

    # 打印异常线段信息
    print(f"Found {len(abnormal_segments)} abnormal segments:")
    for idx, segment in abnormal_segments:
        print(f"Segment: {segment} with length {segment[2]}")
    sys.stdout.flush()

    # 执行模拟退火优化
    # optimized_order, optimized_length = simulated_annealing(node_order, dist_matrix, abnormal_segments)
    optimized_order, optimized_length = two_opt_pruned(node_order, dist_matrix, abnormal_segments)
    
    print(f"Optimized path length: {optimized_length}")
    print(f"Optimized node order: {optimized_order}")
    sys.stdout.flush()

    # 画图
    util.draw_path(optimized_order, points_loc, img_path)

    result_dict = {}
    result_dict["node_order"] = optimized_order
    result_dict["path_length"] = optimized_length

    util.save_json(result_dict, result_path)
        
