import numpy as np
import re
import matplotlib.pyplot as plt
import time
import sys
import json


# 读取tsp.txt文件，返回一个列表和横纵坐标的范围，列表中的每个元素是一个点的对应坐标
def read_tsp_txt(tsp_path):
    with open(tsp_path, "r") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        lines = [line.split(" ") for line in lines]
        lines = [line for line in lines if re.match(r'^\d', line[0])]
    
    min_x = 1e8
    min_y = 1e8
    max_x = -1
    max_y = -1
    for line in lines:
        line[1] = int(line[1])
        line[2] = int(line[2])
        if line[1] < min_x:
            min_x = line[1]
        if line[1] > max_x:
            max_x = line[1]
        if line[2] < min_y:
            min_y = line[2]
        if line[2] > max_y:
            max_y = line[2]

    lines = [line[1:] for line in lines]

    return lines, [min_x, max_x], [min_y, max_y]


# 输入的是点的横纵坐标和图片保存路径，绘制出对应的散点图并保存
def scatter_plot(points, filepath):
    # 提取x和y坐标
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    
    # 创建散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c='black', marker='o', s=5, alpha=0.25)
    
    # 设置坐标轴的范围
    plt.xlim(0, 20000)
    plt.ylim(0, 20000)
    
    # 添加标题和标签
    plt.title('Scatter Plot of Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # 显示网格线
    plt.grid(True)
    
    # 显示散点图
    plt.show()

    # 保存散点图到文件
    plt.savefig(filepath, format='jpg')


# 计时装饰器
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {int(execution_time//60):d}min {(execution_time%60):.3f}s")
        return result
    return wrapper


# 把点的坐标变成邻接矩阵，邻接矩阵的元素值是点之间的欧氏距离
@timer
def calculate_distance_matrix(points_loc, matrix_path):
        dist_matrix = np.zeros((len(points_loc), len(points_loc)))
        for i in range(len(points_loc)):
            for j in range(len(points_loc)):
                dist_matrix[i, j] = np.sqrt((points_loc[i][0] - points_loc[j][0])**2 + (points_loc[i][1] - points_loc[j][1])**2)
                # 由于是完全图，所以不需要计算完，只求上三角就可以了
                if i > j:
                    pass
            if i % 20000 == 0:
                print("Calculating the distance matrix... " + str(i) + " / " + str(len(points_loc)))
                sys.stdout.flush()
        print("Done!")
        sys.stdout.flush()

        # 将其转换为上三角矩阵
        upper_triangular_matrix = np.triu(dist_matrix)

        # 保存邻接矩阵到文件
        np.save(matrix_path, upper_triangular_matrix)
        print("Distance matrix saved to " + matrix_path)
        sys.stdout.flush()


# 把点的坐标变成邻接矩阵，邻接矩阵的元素值是点之间的曼哈顿距离
@timer
def calculate_distance_matrix_manhatan(points_loc, matrix_path):
        dist_matrix = np.zeros((len(points_loc), len(points_loc)))
        for i in range(len(points_loc)):
            for j in range(len(points_loc)):
                dist_matrix[i, j] = np.abs(points_loc[i][0] - points_loc[j][0]) + np.abs(points_loc[i][1] - points_loc[j][1])
                # 由于是完全图，所以不需要计算完，只求上三角就可以了
                if i > j:
                    pass
            if i % 20000 == 0:
                print("Calculating the distance matrix... " + str(i) + " / " + str(len(points_loc)))
                sys.stdout.flush()
        print("Done!")
        sys.stdout.flush()

        # 将其转换为上三角矩阵
        upper_triangular_matrix = np.triu(dist_matrix)

        # 保存邻接矩阵到文件
        np.save(matrix_path, upper_triangular_matrix)
        print("Distance matrix saved to " + matrix_path)
        sys.stdout.flush()


# 将上三角矩阵转换为对称的矩阵
def upper_matrix_to_symmetric_matrix(upper_triangular_matrix):
    symmetric_matrix = upper_triangular_matrix + upper_triangular_matrix.T - np.diag(upper_triangular_matrix.diagonal())
    return symmetric_matrix


# 保存json文件
def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f)


# 读取json文件
def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


# 画路径图
def draw_path(node_order, point_loc, img_path):
    x, y = [], []
    for i in node_order:
        x.append(point_loc[i][0])
        y.append(point_loc[i][1])
    # 最后需要画回去
    x.append(x[0])
    y.append(y[0])

    plt.plot(x, y, '-', color='#FF3030', alpha=0.8, linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.savefig(img_path, format='jpg')


# 载入距离矩阵
def load_dist_matrix(dist_matrix_path):
    load_matrix_time_start = time.time()
    print("Loading the distance matrix...")
    sys.stdout.flush()
    dist_matrix = np.load(dist_matrix_path)
    dist_matrix = upper_matrix_to_symmetric_matrix(dist_matrix)
    load_matrix_time_end = time.time()
    load_matrix_time = load_matrix_time_end - load_matrix_time_start
    print(f"Load matrix took: {int(load_matrix_time//60):d}min {(load_matrix_time%60):.3f}s")
    sys.stdout.flush()

    return dist_matrix


# 计算总路径长度
def cal_total_length(node_order, dist_matrix):
        dis_sum = 0  # 路线距离
        dis = 0
        for i in range(len(node_order)):
            if i < len(node_order) - 1:
                # 依次计录一个数以及下一个数的距离，存入城市间的距离矩阵
                dis = dist_matrix[node_order[i], node_order[i+1]]
                dis_sum = dis_sum + dis
            else:
                # 最后一个数，无下一个数的情况，要回到开始的点
                dis = dist_matrix[node_order[i], node_order[0]]
                dis_sum = dis_sum + dis
        
        # 返回城市间的路线距离
        return round(dis_sum, 1)    # round是对浮点数四舍五入的意思，第二个参数是舍入位数


if __name__ == "__main__":
    tsp_path = "/data2/lhy/project/MonaLisaTSP/mona-lisa100K.tsp.txt"
    # print(read_tsp_txt(tsp_path))   #  [17, 19986], [13, 19978]可以看出是范围在0到20000之间的整数，是正方形的图
    
    # init_scatter_path = "/data3/lhy/project/MonaLisaTSP/img/init_scatter.jpg"
    points_loc = read_tsp_txt(tsp_path)[0]
    # scatter_plot(points_loc, init_scatter_path)
