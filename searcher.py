# 这个文件主要存储各种搜索器
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import time
import copy

import util


# 用numpy实现softmax函数
def softmax(x):
    # 限制 x 的最大值
    # x>709的时候指数太大以至于计算不出来了，所以会返回inf，从而出现警告，所以需要加以限制
    x = np.clip(x, -500, 500)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class GASearcher():
    '''
    遗传算法搜索器
    输入：点坐标，邻接矩阵，种群大小，变异率，交叉率，最大迭代次数，锦标赛小组大小
    '''
    def __init__(self, point_loc, dist_matrix, population_size=25, mutation_rate=0.1, 
                 crossover_rate=0.7, max_generations=200, tournament_size=5,
                 img_file_path="./img"):
        self.point_loc = point_loc  # 点的坐标
        self.dist_matrix = dist_matrix  # 邻接矩阵
        self.population_size = population_size  # 种群数量
        self.mutation_rate = mutation_rate  # 变异概率
        self.crossover_rate = crossover_rate    # 交叉概率
        self.max_generations = max_generations  # 最大代数
        self.tournament_size = tournament_size  # 锦标赛小组大小
        self.img_file_path = img_file_path  # 图片保存路径
        self.start_time = time.time()   # 记录开始时间
        self.node_num = len(self.dist_matrix[0])    # 总节点数量


    # 适应度的计算,适应度就是路线距离，越短越好
    def calFitness(self, node_order):
        # 贪婪策略得到距离矩阵（解码过程）
        # 计算路径距离（评价函数）
        dis_sum = 0  # 路线距离
        dis = 0
        for i in range(len(node_order)):
            if i < len(node_order) - 1:
                # 依次计录一个数以及下一个数的距离，存入城市间的距离矩阵
                dis = self.dist_matrix[node_order[i], node_order[i+1]]
                dis_sum = dis_sum + dis
            else:
                # 最后一个数，无下一个数的情况
                dis = self.dist_matrix[node_order[i], node_order[0]]
                dis_sum = dis_sum + dis
        # 返回城市间的路线距离
        return round(dis_sum, 1)    # round是对浮点数四舍五入的意思，第二个参数是舍入位数


    # 按照最近的贪心方法生成每一次选择最近点的序列（但是贪心是按照概率进行的，有概率选择次优解）
    def generate_node_order(self):
        unvisited = np.arange(self.node_num)  # 初始化未访问的节点列表
        node_order = [0] * self.node_num    # 初始化节点顺序列表
        start_node = random.randint(0, self.node_num - 1)   # 随机一个出发节点

        sorted_node_num = 0 # 已经编入序列的节点数
        begin_flag = 0  # 标记前面要填入的节点的位置
        end_flag = self.node_num - 1    # 标记后面要填入的节点的位置
        while sorted_node_num < self.node_num:
            if sorted_node_num == 0:    # 刚开始生成序列的情况
                node_order[begin_flag] = start_node
                unvisited = np.delete(unvisited, np.where(unvisited == start_node)[0][0])    # 标记为已访问
                sorted_node_num += 1    # 已排序节点数量加一
                begin_flag += 1         # 前面的标记往后移动一位
            elif (self.node_num - sorted_node_num) == 1:    # 只剩下一个节点未访问的情况
                # 放进去就完了
                node_order[begin_flag] = unvisited[0]
                break
            else:   # 其他就是还剩下大于等于2的未访问节点数了
                # 先找前面要填充进来的节点
                # 找到当前flag的前一个节点（也就是环上按顺序最后一个被访问的节点）里未访问的两个最近节点的索引和距离
                closest_nodes, closest_distances = self.choose_closest_nodes(self.dist_matrix[node_order[begin_flag-1]], 
                                                                             unvisited, 
                                                                             choose_num=2)
                closest_probability = softmax(closest_distances)    # 把对应的距离进行softmax作为选择的概率
                if random.random() > closest_probability[1]:    # 距离越大，选中的概率越小
                    chosen_node = closest_nodes[0]
                else:
                    chosen_node = closest_nodes[1]
                
                node_order[begin_flag] = chosen_node
                unvisited = np.delete(unvisited, np.where(unvisited == chosen_node)[0][0])    # 标记为已访问
                sorted_node_num += 1    # 已排序节点数量加一
                begin_flag += 1         # 前面的标记往后移动一位
                
                # 再找后面要填充进来的节点
                if end_flag == (self.node_num - 1): # 如果现在排序列表末尾没有存入节点
                    # 找到环上开始的节点里未访问的两个最近节点的索引和距离
                    closest_nodes, closest_distances = self.choose_closest_nodes(self.dist_matrix[node_order[0]], 
                                                                                unvisited, 
                                                                                choose_num=2)
                else:
                    # 找到当前flag的后一个节点（也就是环上按逆序上一个被访问的节点）里未访问的两个最近节点的索引和距离
                    closest_nodes, closest_distances = self.choose_closest_nodes(self.dist_matrix[node_order[end_flag + 1]], 
                                                                                unvisited, 
                                                                                choose_num=2)
                closest_probability = softmax(closest_distances)    # 把对应的距离进行softmax作为选择的概率
                if random.random() > closest_probability[1]:    # 距离越大，选中的概率越小
                    chosen_node = closest_nodes[0]
                else:
                    chosen_node = closest_nodes[1]
                
                node_order[end_flag] = chosen_node
                unvisited = np.delete(unvisited, np.where(unvisited == chosen_node)[0][0])    # 标记为已访问
                sorted_node_num += 1    # 已排序节点数量加一
                end_flag -= 1         # 后面的标记往前移动一位
        
        return node_order


    # 获取距离当前节点最近的n个节点
    # 给出一个节点到所有节点的距离，未访问的节点列表
    def choose_closest_nodes(self, distances, unvisited, choose_num=2):
        '''
        示例
        distances = np.array([0, 10, 20, 9, 15, 7])  # 当前节点到所有节点的距离
        unvisited = np.array([1, 3, 4, 5])           # 未访问节点的列表
        choose_num = 2                              # 要选择的节点数
        '''
        # 1. 筛选出未访问节点对应的距离
        unvisited_distances = distances[unvisited]

        # 2. 找到距离最近的n个未访问节点
        sorted_indices = np.argsort(unvisited_distances)  # 按距离排序，返回排序后的索引
        closest_indices = sorted_indices[:choose_num]          # 取出距离最小的n个

        # 3. 获取这些未访问节点的实际节点编号
        closest_nodes = unvisited[closest_indices]

        # 4. 获取节点对应的距离
        closest_distances = unvisited_distances[closest_indices]

        return closest_nodes, closest_distances

    # 联赛选择算子
    def tournament_select(self, pops, fits):
        new_pops, new_fits = [], []
        popsize = self.population_size
        tournament_size = self.tournament_size

        fits_copy = fits.copy()  # 浅拷贝适应度列表，避免修改原本的数据

        while len(new_pops) < popsize:    # 需要一次补充一个个体直至补充到种群数量
            # 步骤1 从群体中随机选择M个个体，计算每个个体的目标函数值
            tournament_list = random.sample(range(0, popsize), tournament_size) # 随机采样锦标赛小组个个体
            tournament_fit = [fits_copy[i] for i in tournament_list]
            # 转化为 numpy 数组
            tournament_list_array = np.array(tournament_list)
            tournament_fit_array = np.array(tournament_fit)
            
            # 步骤2 根据每个个体的目标函数值，计算其适应度
            # 将 tournament_list 和 tournament_fit 合并成一个二维数组，并按适应度排序
            tournament_array = np.vstack((tournament_list_array, tournament_fit_array)).T
            sorted_tournament_array = tournament_array[np.argsort(tournament_array[:, 1])]
            # 步骤3 选择适应度最大的个体
            best_index = int(sorted_tournament_array[0, 0])  # 获取适应度最大的个体的索引
            fit = fits[best_index]     # 获取未经过处理的适应度
            pop = pops[int(sorted_tournament_array[0, 0])]
            new_pops.append(pop)
            new_fits.append(fit)

            # 将适应度乘以1.0003，略微变大以降低再次被选择的可能，这个也是模拟自然界，锦标赛之后也会累，后面再参加锦标赛就没有那么好的性能了
            fits_copy[best_index] *= 1.0003
        

        return new_pops, new_fits

    
    # 交叉算子
    def crossover(self, parent1_pops, parent2_pops):
        popsize = self.population_size
        child_pops = []
        for i in range(popsize):
            # 初始化
            child = [None] * len(parent1_pops[i])   # 初始化孩子的节点顺序
            # 取对应的父母节点
            parent1 = parent1_pops[i]
            parent2 = parent2_pops[i]
            # 如果不满足交叉概率则直接继承
            if random.random() >= self.crossover_rate:
                child = parent1.copy()  # 随机生成一个（或者随机保留父代中的一个）
                # random.shuffle(child)   # 要对孩子节点进行打乱顺序
            else:
                # 从父代 parent1 中随机选择两个位置，start_pos 和 end_pos。
                # 这两个位置定义了一个区间，表示在这个区间内的基因（城市的顺序）将从 parent1 复制到子代 child。
                # 如果 start_pos 大于 end_pos，则交换它们的值，确保 start_pos 小于或等于 end_pos。
                # parent1
                start_pos = random.randint(0, len(parent1) - 1)
                end_pos = random.randint(0, len(parent1) - 1)
                if start_pos > end_pos:
                    tem_pop = start_pos
                    start_pos = end_pos
                    end_pos = tem_pop
                child[start_pos:end_pos + 1] = parent1[start_pos:end_pos + 1].copy()
                # parent2 -> child, 从 parent2 填充剩余基因：
                list1 = list(range(end_pos + 1, len(parent2)))
                list2 = list(range(0, start_pos))
                list_index = list1 + list2
                j = -1
                for i in list_index:
                    for j in range(j + 1, len(parent2)):
                        if parent2[j] not in child:
                            child[i] = parent2[j]
                            break
            child_pops.append(child)

        return child_pops


    # 变异操作
    def mutate(self, pops):
        pops_mutate = []
        for i in range(len(pops)):
            pop = pops[i].copy()
            # 随机多次成对变异
            # 随机选出两个位置进行交换
            t = random.randint(1, len(pop)/4)   # 这个参数的设计是按照经验规律自己设定的
            count = 0
            while count < t:
                # 变异就是把自己的顺序随机交换两个位置
                if random.random() < self.mutation_rate:
                    mut_pos1 = random.randint(0, len(pop) - 1)
                    mut_pos2 = random.randint(0, len(pop) - 1)
                    '''
                    #如果不相等则进行取反的操作，这里使用交换
                    if mut_pos1 != mut_pos2:
                        tem = pop[mut_pos1]
                        pop[mut_pos1] = pop[mut_pos2]
                        pop[mut_pos2] = tem
                    '''
                    if mut_pos1 == mut_pos2:
                        mut_pos2 = mut_pos1 + 1
                    tem = pop[mut_pos1]
                    pop[mut_pos1] = pop[mut_pos2]
                    pop[mut_pos2] = tem
                pops_mutate.append(pop)
                count += 1
        return pops_mutate


    # 生存竞争
    def survive_compete(self, pop_list1, fit_list1, pop_list2, fit_list2):
        # 合并两个种群和它们的适应度
        combined_pops = pop_list1 + pop_list2
        combined_fits = fit_list1 + fit_list2

        # 对适应度进行排序，并获取索引
        sorted_indices = np.argsort(combined_fits)

        # 选择前一半的个体
        half_size = len(combined_pops) // 2
        top_indices = sorted_indices[:half_size]

        # 创建新的种群和适应度列表
        new_pops = []
        new_fits = []

        # 将选择的个体添加到新列表中
        for index in top_indices:
            new_pops.append(combined_pops[index])
            new_fits.append(combined_fits[index])

        return new_pops, new_fits


    # 画路径图
    def draw_path(self, node_order, img_path):
        x, y = [], []
        for i in node_order:
            x.append(self.point_loc[i][0])
            y.append(self.point_loc[i][1])
        # 最后需要画回去
        x.append(x[0])
        y.append(y[0])

        plt.plot(x, y, '-', color='#FF3030', alpha=0.8, linewidth=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        plt.savefig(img_path, format='jpg')


    def ga_optimize(self):
        iteration = 0
        # 初始化,构造种群数量个个体
        print("-"*50)
        print("Ancestors Generating...^(*￣(oo)￣)^")
        sys.stdout.flush()
        pops = []
        for i in range(self.population_size):
            pops.append(self.generate_node_order())
        execution_time = time.time() - self.start_time
        print(f"Time Used: {int(execution_time//60):d}min {(execution_time%60):.3f}s")
        sys.stdout.flush()
        # 画出初始化得到的城市连接图
        self.draw_path(pops[0], self.img_file_path + "/ga_0.jpg")
        # 计算适应度
        fits = [None] * self.population_size
        for i in range(self.population_size):
            fits[i] = self.calFitness(pops[i])
        # 保留当前最优,最小的fits为最优解
        best_fit = min(fits)
        best_pop = pops[fits.index(best_fit)]
        print('初代最优值 %.1f' % (best_fit))
        sys.stdout.flush()
        best_fit_list = []
        best_fit_list.append(best_fit)

        while iteration <= self.max_generations:
            # 锦标赛以将适应度更好的亲代排在前面
            print("Tournament Selecting...⊙w⊙∥")
            pop1, fits1 = self.tournament_select(pops, fits)
            pop2, fits2 = self.tournament_select(pops, fits)
            print("Fitness of parents1:", fits1)
            print("Fitness of parents2:", fits2)
            execution_time = time.time() - self.start_time
            print(f"Time Used: {int(execution_time//60):d}min {(execution_time%60):.3f}s")
            sys.stdout.flush()

            # 交叉
            print("Cross Overing...❤❤❤")
            child_pops = self.crossover(pop1, pop2)
            execution_time = time.time() - self.start_time
            print(f"Time Used: {int(execution_time//60):d}min {(execution_time%60):.3f}s")
            sys.stdout.flush()

            # 变异
            print("Mutateing...(╯▔皿▔)╯")
            child_pops = self.mutate(child_pops)
            execution_time = time.time() - self.start_time
            print(f"Time Used: {int(execution_time//60):d}min {(execution_time%60):.3f}s")
            sys.stdout.flush()

            # 计算子代适应度
            print("Fits Calculating...")
            child_fits = [None] * self.population_size
            for i in range(self.population_size):
                child_fits[i] = self.calFitness(child_pops[i])
            execution_time = time.time() - self.start_time
            print(f"Time Used: {int(execution_time//60):d}min {(execution_time%60):.3f}s")
            sys.stdout.flush()

            # 本地种群生存竞争
            print("The local population is competing for survival...")
            pops, fits = self.survive_compete(pops, fits, child_pops, child_fits)

            '''
            # 每过20代引入一批和本地种群数量一样的外来者
            if iteration % 20 == 0:
                print("Outsiders Generating...^(*￣(oo)￣)^")
                sys.stdout.flush()
                outsider_pops = []
                for i in range(self.population_size):
                    outsider_pops.append(self.generate_node_order())
                # 计算外来者适应度
                print("Fits Calculating...")
                outsider_fits = [None] * self.population_size
                for i in range(self.population_size):
                    outsider_fits[i] = self.calFitness(outsider_pops[i])
                execution_time = time.time() - self.start_time
                print(f"Time Used: {int(execution_time//60):d}min {(execution_time%60):.3f}s")
                sys.stdout.flush()

                # 外来者和本地种群生存竞争
                print("Local populations are competing with populations from other regions for survival...")
                pops, fits = self.survive_compete(pops, fits, outsider_pops, outsider_fits)
            '''

            # 更新最优值
            if best_fit > min(fits):
                best_fit = min(fits)
                best_pop = pops[fits.index(best_fit)]
            best_fit_list.append(best_fit)  # 记录训练数据

            
            execution_time = time.time() - self.start_time
            print('第%d代最优值 %.1f' % (iteration, best_fit))
            print(f"Time Used: {int(execution_time//60):d}min {(execution_time%60):.3f}s")

            if iteration % 20 == 0:
                self.draw_path(best_pop, self.img_file_path + f"/ga_{int(iteration):d}.jpg")
                print("Current Best Point Order:", best_pop)
            
            print("-"*50)
            sys.stdout.flush()
            iteration += 1
    
        # 路径顺序
        print("-"*50)
        print("Best Point Order:", best_pop)
        print("Best Fitness:", best_fit)
        self.draw_path(best_pop, self.img_file_path + "/ga_result.jpg")
        
        '''
        !!!注意,这里输出的point_order是0开头的,后续处理需要处理需要注意!!!
        '''
        
        return best_pop, best_fit, best_fit_list


class AStarSearcher():
    '''
    A*搜索器
    输入：点坐标、邻接矩阵、图片保存路径
    h(n)：当前点到当前点最远未访问点的欧氏距离加上最远未访问点到初始点的欧氏距离
    '''
    def __init__(self, point_loc, dist_matrix, 
                 img_file_path="./img"):
        self.point_loc = point_loc  # 点的坐标
        self.dist_matrix = dist_matrix  # 邻接矩阵
        self.img_file_path = img_file_path  # 图片保存路径

        self.node_num = len(self.dist_matrix[0])    # 总节点数量
        self.unvisited = np.arange(self.node_num)  # 初始化未访问的节点列表
        self.start_node = 0 # 把第一个节点作为初始节点进行查询

        self.search_list = []  # 要搜索节点的列表
        self.search_fn_list = []    # 和搜索节点列表对应的f(n)值列表
        self.current_gn = 0 # 当前已经走过的路径长度
        self.current_node = None   # 当前搜索到的节点，在这里先设置为空
        self.result_node_order = [] # 输出的搜索结果，存储节点顺序

        # 下面的参数是为了双向搜索准备的
        self.back_search_list = []  # 要搜索节点的列表
        self.back_search_fn_list = []    # 和搜索节点列表对应的f(n)值列表
        self.back_current_gn = 0 # 当前已经走过的路径长度
        self.back_current_node = None   # 当前搜索到的节点，在这里先设置为空
        self.back_result_node_order = [] # 输出的搜索结果，存储节点顺序

        self.start_time = time.time()


    # 获取距离当前节点最近的n个节点
    # 给出一个节点到所有节点的距离，未访问的节点列表
    def choose_closest_nodes(self, distances, unvisited, choose_num=20):
        '''
        示例
        distances = np.array([0, 10, 20, 9, 15, 7])  # 当前节点到所有节点的距离
        unvisited = np.array([1, 3, 4, 5])           # 未访问节点的列表
        choose_num = 2                              # 要选择的节点数
        '''
        # 1. 筛选出未访问节点对应的距离
        unvisited_distances = distances[unvisited]

        # 2. 找到距离最近的n个未访问节点
        sorted_indices = np.argsort(unvisited_distances)  # 按距离排序，返回排序后的索引
        closest_indices = sorted_indices[:choose_num]          # 取出距离最小的n个

        # 3. 获取这些未访问节点的实际节点编号
        closest_nodes = unvisited[closest_indices]

        # 4. 获取节点对应的距离
        closest_distances = unvisited_distances[closest_indices]

        return closest_nodes, closest_distances


    # 单向生成的astar
    def astar_search_one_way(self, nearest_node_num=20):
        # 增加限制：如果当前未访问的节点超过二十个，就只取最近的二十个作为孩子节点；不然就全部拿来算
        # 初始化
        print("Initing...")
        sys.stdout.flush()
        self.search_list.append(self.start_node)
        # 处理初始节点
        self.current_node = self.search_list.pop()  # 将初始节点弹出进行搜索
        self.result_node_order.append(self.current_node)  # 将初始节点存入结果列表
        self.unvisited = np.delete(self.unvisited, np.where(self.unvisited == self.current_node)[0][0])    # 标记为已访问
        num_visited_point = 1

        # 开始搜索
        print("Searching...")
        sys.stdout.flush()
        while len(self.unvisited) != 0:
            # 输出实时信息
            if num_visited_point % 1000 == 0:
                print("Processed:", f"{int(num_visited_point)}/{int(self.node_num)}...")
                print("Current Length:", self.current_gn)
                print("Current List:", self.result_node_order)
                execution_time = time.time() - self.start_time
                print(f"Time Used: {int(execution_time//60):d}min {(execution_time%60):.3f}s")
                sys.stdout.flush()
            
            # 筛选靠近的点作为孩子节点
            if len(self.unvisited) >= nearest_node_num:
                distances = self.dist_matrix[self.current_node]
                closest_nodes, _ = self.choose_closest_nodes(distances, self.unvisited, choose_num=nearest_node_num)
            else:
                distances = self.dist_matrix[self.current_node]
                unvisited_distances = distances[self.unvisited]
                sorted_indices = np.argsort(unvisited_distances)  # 按距离排序，返回排序后的索引
                closest_nodes = self.unvisited[sorted_indices]
            
            # 不在search_list里面的放入搜索列表作为孩子节点
            # 把孩子节点放入搜索列表
            children_list = []
            for i in range(len(closest_nodes)):
                if closest_nodes[i] not in self.search_list:
                    children_list.append(closest_nodes[i])
                    self.search_list.append(closest_nodes[i])
            
            # 计算新加入节点的f(n)；旧节点的f(n)在上一轮的末尾已经更新，不用重新计算
            fn_list = []
            for i in range(len(children_list)):
                # 找到未访问节点中到对应子节点距离最大的点
                current_node_distance = self.dist_matrix[children_list[i]]
                unvisited_current_distances = current_node_distance[self.unvisited]
                max_current_distance = np.max(unvisited_current_distances)
                farest_node = np.where(current_node_distance == max_current_distance)[0][0]

                hn = self.dist_matrix[children_list[i]][farest_node] + \
                        self.dist_matrix[farest_node][self.start_node]
                fn = hn + self.current_gn + self.dist_matrix[self.current_node][children_list[i]]

                fn_list.append(fn)
            
            for i in range(len(fn_list)):# 把f(n)列表放入搜索节点的f(n)列表
                self.search_fn_list.append(fn_list[i])

            if len(self.search_list) != 1: # 如果要搜索的节点不止一个
                # 根据f(n)排序，选取节点
                # 转换为numpy数组
                search_fn_list_np = np.array(self.search_fn_list)
                # 获取fn最小的节点
                min_fn = np.min(search_fn_list_np)
                min_index = int(np.where(search_fn_list_np == min_fn)[0][0])    # 对应的最小fn节点的索引
                # 处理参数
                f_min_node = self.search_list.pop(min_index)
                self.search_fn_list.pop(min_index)
                self.current_gn += self.dist_matrix[self.current_node][f_min_node] # 更新gn
                self.current_node = f_min_node  # 把当前搜索的节点指针指向这个节点
                self.result_node_order.append(f_min_node)  # 将节点存入结果列表
                self.unvisited = np.delete(self.unvisited, np.where(self.unvisited == f_min_node)[0][0])    # 标记为已访问
                num_visited_point += 1
            else:   # 如果只有一个剩下的节点了
                f_min_node = self.search_list.pop()
                self.search_fn_list.pop()
                self.current_gn += (self.dist_matrix[self.current_node][f_min_node] + \
                                    self.dist_matrix[f_min_node][self.start_node]) # 更新gn，需要回到开始的位置
                self.current_node = f_min_node  # 把当前搜索的节点指针指向这个节点
                self.result_node_order.append(f_min_node)  # 将节点存入结果列表
                self.unvisited = np.delete(self.unvisited, np.where(self.unvisited == f_min_node)[0][0])    # 标记为已访问
                num_visited_point += 1
            
            # 更新搜索列表里的f(n);因为当前节点变化了，所以gn的值变化导致了fn的变化，同时hn也可能变化
            for i in range(len(self.search_list)):
                if len(self.unvisited) == 0:
                    break
                current_node_distance = self.dist_matrix[self.search_list[i]]
                unvisited_current_distances = current_node_distance[self.unvisited]
                max_current_distance = np.max(unvisited_current_distances)
                farest_node = np.where(current_node_distance == max_current_distance)[0][0]

                hn = self.dist_matrix[self.search_list[i]][farest_node] + \
                        self.dist_matrix[farest_node][self.start_node]
                self.search_fn_list[i] = hn + self.current_gn + self.dist_matrix[self.current_node][self.search_list[i]]
        
        self.draw_path(self.result_node_order, self.img_file_path + f"/astar_one_way_{nearest_node_num}.jpg")
        print("Node Oreder:", self.result_node_order)
        print("Path length:", self.current_gn)  # 搜索到最后self.current_gn就是路径长度
        return self.result_node_order, self.current_gn


    # 双生成的astar
    def astar_search_two_way(self, nearest_node_num=20):
        # 增加限制：如果当前未访问的节点超过二十个，就只取最近的二十个作为孩子节点；不然就全部拿来算
        # 初始化
        print("Initing...")
        sys.stdout.flush()
        self.search_list.append(self.start_node)
        self.back_search_list.append(self.start_node)
        # 处理初始节点
        self.current_node = self.search_list.pop()  # 将初始节点弹出进行搜索
        self.back_current_node = self.back_search_list.pop()
        self.result_node_order.append(self.current_node)  # 将初始节点存入结果列表
        # self.back_result_node_order = [self.back_current_node] + self.back_result_node_order    # 后面的节点需要倒序存储，但是初始节点不用存到最后
        self.unvisited = np.delete(self.unvisited, np.where(self.unvisited == self.current_node)[0][0])    # 标记为已访问
        num_visited_point = 1

        # 开始搜索
        print("Searching...")
        sys.stdout.flush()
        while len(self.unvisited) != 0:
            # 输出实时信息
            if (num_visited_point-1) % 1000 == 0:
                print("Processed:", f"{int(num_visited_point)}/{int(self.node_num)}...")
                print("Current Length:", self.current_gn + self.back_current_gn)
                print("Current Front List:", self.result_node_order)
                print("Current Back List:", self.back_result_node_order)
                execution_time = time.time() - self.start_time
                print(f"Time Used: {int(execution_time//60):d}min {(execution_time%60):.3f}s")
                sys.stdout.flush()
            
            # 正反双向分别筛选靠近的点作为孩子节点
            if len(self.unvisited) >= nearest_node_num:
                distances = self.dist_matrix[self.current_node]
                closest_nodes, _ = self.choose_closest_nodes(distances, self.unvisited, choose_num=nearest_node_num)
                
                back_distances = self.dist_matrix[self.back_current_node]
                back_closest_nodes, _ = self.choose_closest_nodes(back_distances, self.unvisited, choose_num=nearest_node_num)
            else:
                distances = self.dist_matrix[self.current_node]
                unvisited_distances = distances[self.unvisited]
                sorted_indices = np.argsort(unvisited_distances)  # 按距离排序，返回排序后的索引
                closest_nodes = self.unvisited[sorted_indices]

                back_distances = self.dist_matrix[self.back_current_node]
                back_unvisited_distances = back_distances[self.unvisited]
                back_sorted_indices = np.argsort(back_unvisited_distances)  # 按距离排序，返回排序后的索引
                back_closest_nodes = self.unvisited[back_sorted_indices]
            
            # 不在search_list里面的放入搜索列表作为孩子节点
            # 把孩子节点放入搜索列表
            children_list = []
            back_children_list = []
            for i in range(len(closest_nodes)):
                if closest_nodes[i] not in self.search_list:
                    children_list.append(closest_nodes[i])
                    self.search_list.append(closest_nodes[i])
            for i in range(len(back_closest_nodes)):
                if back_closest_nodes[i] not in self.back_search_list:
                    back_children_list.append(back_closest_nodes[i])
                    self.back_search_list.append(back_closest_nodes[i])
            
            # 计算新加入节点的f(n)；旧节点的f(n)在上一轮的末尾已经更新，不用重新计算
            fn_list = []
            for i in range(len(children_list)):
                # 找到未访问节点中到对应子节点距离最大的点
                current_node_distance = self.dist_matrix[children_list[i]]
                unvisited_current_distances = current_node_distance[self.unvisited]
                max_current_distance = np.max(unvisited_current_distances)
                farest_node = np.where(current_node_distance == max_current_distance)[0][0]

                hn = self.dist_matrix[children_list[i]][farest_node] + \
                        self.dist_matrix[farest_node][self.start_node]
                fn = hn + self.current_gn + self.dist_matrix[self.current_node][children_list[i]]

                fn_list.append(fn)
            
            self.search_fn_list = self.search_fn_list + fn_list # 把f(n)列表放入搜索节点的f(n)列表

            back_fn_list = []
            for i in range(len(back_children_list)):
                # 找到未访问节点中到对应子节点距离最大的点
                back_current_node_distance = self.dist_matrix[back_children_list[i]]
                back_unvisited_current_distances = back_current_node_distance[self.unvisited]
                back_max_current_distance = np.max(back_unvisited_current_distances)
                back_farest_node = np.where(back_current_node_distance == back_max_current_distance)[0][0]

                hn = self.dist_matrix[back_children_list[i]][back_farest_node] + \
                        self.dist_matrix[back_farest_node][self.start_node]
                fn = hn + self.back_current_gn + self.dist_matrix[self.back_current_node][back_children_list[i]]

                back_fn_list.append(fn)
            
            self.back_search_fn_list = self.back_search_fn_list + back_fn_list

            if len(self.unvisited) > 1: # 如果剩在外面的节点不止一个
                # 剩在外面的节点大于等于两个的话就头部取一个，尾部取一个

                # 操作头部

                # 根据f(n)排序，选取节点
                # 转换为numpy数组
                search_fn_list_np = np.array(self.search_fn_list)
                # 获取fn最小的节点
                min_fn = np.min(search_fn_list_np)
                min_index = int(np.where(search_fn_list_np == min_fn)[0][0])    # 对应的最小fn节点的索引
                # 处理参数
                f_min_node = self.search_list.pop(min_index)
                self.search_fn_list.pop(min_index)
                self.current_gn += self.dist_matrix[self.current_node][f_min_node] # 更新gn
                self.current_node = f_min_node  # 把当前搜索的节点指针指向这个节点
                self.result_node_order.append(f_min_node)  # 将节点存入结果列表
                self.unvisited = np.delete(self.unvisited, np.where(self.unvisited == f_min_node)[0][0])    # 标记为已访问
                num_visited_point += 1
                # 如果这个节点在back_search_list里面的位置也要找出来对应删掉
                if f_min_node in self.back_search_list:
                    delete_index = int(np.where(self.back_search_list == f_min_node)[0][0])
                    self.back_search_list.pop(delete_index)
                    self.back_search_fn_list.pop(delete_index)

                # 操作尾部

                # 根据f(n)排序，选取节点
                # 转换为numpy数组
                back_search_fn_list_np = np.array(self.back_search_fn_list)
                # 获取fn最小的节点
                back_min_fn = np.min(back_search_fn_list_np)
                back_min_index = int(np.where(back_search_fn_list_np == back_min_fn)[0][0])    # 对应的最小fn节点的索引
                # 处理参数
                back_f_min_node = self.back_search_list.pop(back_min_index)
                self.back_search_fn_list.pop(back_min_index)
                self.back_current_gn += self.dist_matrix[self.back_current_node][back_f_min_node] # 更新gn
                self.back_current_node = back_f_min_node  # 把当前搜索的节点指针指向这个节点
                self.back_result_node_order = [back_f_min_node] + self.back_result_node_order    # 后面的节点需要倒序存储
                self.unvisited = np.delete(self.unvisited, np.where(self.unvisited == back_f_min_node)[0][0])    # 标记为已访问
                num_visited_point += 1
                # 如果这个节点在search_list里面的位置也要找出来对应删掉
                if back_f_min_node in self.search_list:
                    delete_index = int(np.where(self.search_list == back_f_min_node)[0][0])
                    self.search_list.pop(delete_index)
                    self.search_fn_list.pop(delete_index)
            else:   # 如果只有一个剩下的节点了，就加入头部了事；后续还需要把头尾接起来
                f_min_node = self.search_list.pop()
                self.search_fn_list.pop()
                # 更新gn，需要把前后连接起来
                self.current_gn += self.dist_matrix[self.current_node][f_min_node]
                self.current_gn += self.back_current_gn
                self.current_gn += self.dist_matrix[f_min_node][self.back_current_node]
                self.current_node = f_min_node  # 把当前搜索的节点指针指向这个节点
                self.result_node_order.append(f_min_node)  # 将节点存入结果列表
                # 合并头部尾部的搜索列表
                self.result_node_order = self.result_node_order + self.back_result_node_order
                self.unvisited = np.delete(self.unvisited, np.where(self.unvisited == f_min_node)[0][0])    # 标记为已访问
                num_visited_point += 1
            
            # 更新搜索列表里的f(n);因为当前节点变化了，所以gn的值变化导致了fn的变化，同时hn也可能变化
            for i in range(len(self.search_list)):
                if len(self.unvisited) == 0:
                    break
                current_node_distance = self.dist_matrix[self.search_list[i]]
                unvisited_current_distances = current_node_distance[self.unvisited]
                max_current_distance = np.max(unvisited_current_distances)
                farest_node = np.where(current_node_distance == max_current_distance)[0][0]

                hn = self.dist_matrix[self.search_list[i]][farest_node] + \
                        self.dist_matrix[farest_node][self.start_node]
                self.search_fn_list[i] = hn + self.current_gn + self.dist_matrix[self.current_node][self.search_list[i]]
            
            # 同样的back_list里的也要更新
            for i in range(len(self.back_search_list)):
                if len(self.unvisited) == 0:
                    break
                back_current_node_distance = self.dist_matrix[self.back_search_list[i]]
                back_unvisited_current_distances = back_current_node_distance[self.unvisited]
                back_max_current_distance = np.max(back_unvisited_current_distances)
                back_farest_node = np.where(back_current_node_distance == back_max_current_distance)[0][0]

                hn = self.dist_matrix[self.back_search_list[i]][back_farest_node] + \
                        self.dist_matrix[back_farest_node][self.start_node]
                self.back_search_fn_list[i] = hn + self.back_current_gn + self.dist_matrix[self.back_current_node][self.back_search_list[i]]

        self.draw_path(self.result_node_order, self.img_file_path + f"/astar_two_way_{nearest_node_num}.jpg")
        print("Node Oreder:", self.result_node_order)
        print("Path length:", self.current_gn)  # 搜索到最后self.current_gn就是路径长度
        return self.result_node_order, self.current_gn

    # 单向生成的astar，启发函数是当前节点到未访问节点距离的平均值
    def astar_search_one_way_avg(self, nearest_node_num=20):
        # 增加限制：如果当前未访问的节点超过二十个，就只取最近的二十个作为孩子节点；不然就全部拿来算
        # 初始化
        print("Initing...")
        sys.stdout.flush()
        self.search_list.append(self.start_node)
        # 处理初始节点
        self.current_node = self.search_list.pop()  # 将初始节点弹出进行搜索
        self.result_node_order.append(self.current_node)  # 将初始节点存入结果列表
        self.unvisited = np.delete(self.unvisited, np.where(self.unvisited == self.current_node)[0][0])    # 标记为已访问
        num_visited_point = 1

        # 开始搜索
        print("Searching...")
        sys.stdout.flush()
        while len(self.unvisited) != 0:
            # 输出实时信息
            if num_visited_point % 1000 == 0:
                print("Processed:", f"{int(num_visited_point)}/{int(self.node_num)}...")
                print("Current Length:", self.current_gn)
                print("Current List:", self.result_node_order)
                execution_time = time.time() - self.start_time
                print(f"Time Used: {int(execution_time//60):d}min {(execution_time%60):.3f}s")
                sys.stdout.flush()
            
            # 筛选靠近的点作为孩子节点
            if len(self.unvisited) >= nearest_node_num:
                distances = self.dist_matrix[self.current_node]
                closest_nodes, _ = self.choose_closest_nodes(distances, self.unvisited, choose_num=nearest_node_num)
            else:
                distances = self.dist_matrix[self.current_node]
                unvisited_distances = distances[self.unvisited]
                sorted_indices = np.argsort(unvisited_distances)  # 按距离排序，返回排序后的索引
                closest_nodes = self.unvisited[sorted_indices]
            
            # 不在search_list里面的放入搜索列表作为孩子节点
            # 把孩子节点放入搜索列表
            children_list = []
            for i in range(len(closest_nodes)):
                if closest_nodes[i] not in self.search_list:
                    children_list.append(closest_nodes[i])
                    self.search_list.append(closest_nodes[i])
            
            # 计算新加入节点的f(n)；旧节点的f(n)在上一轮的末尾已经更新，不用重新计算
            fn_list = []
            for i in range(len(children_list)):
                # 找到未访问节点中到对应子节点距离最大的点
                current_node_distance = self.dist_matrix[children_list[i]]
                unvisited_current_distances = current_node_distance[self.unvisited]
                # max_current_distance = np.max(unvisited_current_distances)
                # farest_node = np.where(current_node_distance == max_current_distance)[0][0]

                hn = np.mean(unvisited_current_distances)
                fn = hn + self.current_gn + self.dist_matrix[self.current_node][children_list[i]]

                fn_list.append(fn)
            
            for i in range(len(fn_list)):# 把f(n)列表放入搜索节点的f(n)列表
                self.search_fn_list.append(fn_list[i])

            if len(self.search_list) != 1: # 如果要搜索的节点不止一个
                # 根据f(n)排序，选取节点
                # 转换为numpy数组
                search_fn_list_np = np.array(self.search_fn_list)
                # 获取fn最小的节点
                min_fn = np.min(search_fn_list_np)
                min_index = int(np.where(search_fn_list_np == min_fn)[0][0])    # 对应的最小fn节点的索引
                # 处理参数
                f_min_node = self.search_list.pop(min_index)
                self.search_fn_list.pop(min_index)
                self.current_gn += self.dist_matrix[self.current_node][f_min_node] # 更新gn
                self.current_node = f_min_node  # 把当前搜索的节点指针指向这个节点
                self.result_node_order.append(f_min_node)  # 将节点存入结果列表
                self.unvisited = np.delete(self.unvisited, np.where(self.unvisited == f_min_node)[0][0])    # 标记为已访问
                num_visited_point += 1
            else:   # 如果只有一个剩下的节点了
                f_min_node = self.search_list.pop()
                self.search_fn_list.pop()
                self.current_gn += (self.dist_matrix[self.current_node][f_min_node] + \
                                    self.dist_matrix[f_min_node][self.start_node]) # 更新gn，需要回到开始的位置
                self.current_node = f_min_node  # 把当前搜索的节点指针指向这个节点
                self.result_node_order.append(f_min_node)  # 将节点存入结果列表
                self.unvisited = np.delete(self.unvisited, np.where(self.unvisited == f_min_node)[0][0])    # 标记为已访问
                num_visited_point += 1
            
            # 更新搜索列表里的f(n);因为当前节点变化了，所以gn的值变化导致了fn的变化，同时hn也可能变化
            for i in range(len(self.search_list)):
                if len(self.unvisited) == 0:
                    break
                current_node_distance = self.dist_matrix[self.search_list[i]]
                unvisited_current_distances = current_node_distance[self.unvisited]
                # max_current_distance = np.max(unvisited_current_distances)
                # farest_node = np.where(current_node_distance == max_current_distance)[0][0]

                hn = np.mean(unvisited_current_distances)
                self.search_fn_list[i] = hn + self.current_gn + self.dist_matrix[self.current_node][self.search_list[i]]
        
        self.draw_path(self.result_node_order, self.img_file_path + f"/astar_one_way_avg_{nearest_node_num}.jpg")
        print("Node Oreder:", self.result_node_order)
        print("Path length:", self.current_gn)  # 搜索到最后self.current_gn就是路径长度
        return self.result_node_order, self.current_gn


    # 画路径图
    def draw_path(self, node_order, img_path):
        x, y = [], []
        for i in node_order:
            x.append(self.point_loc[i][0])
            y.append(self.point_loc[i][1])
        # 最后需要画回去
        x.append(x[0])
        y.append(y[0])

        plt.plot(x, y, '-', color='#FF3030', alpha=0.8, linewidth=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        plt.savefig(img_path, format='jpg')


if __name__ == "__main__":
    tsp_path = "/data2/lhy/project/MonaLisaTSP/mona-lisa100K.tsp.txt"

    dist_matrix_path = "/data2/lhy/project/MonaLisaTSP/mona_lisa100K_dist_matrix.npy"
    dist_matrix = np.load(dist_matrix_path)
    dist_matrix = util.upper_matrix_to_symmetric_matrix(dist_matrix)
    
    points_loc = util.read_tsp_txt(tsp_path)[0]

    ga_test = GASearcher(points_loc, dist_matrix)
    ga_test.ga_optimize()

