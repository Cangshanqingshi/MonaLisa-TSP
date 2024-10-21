nearest_node_num=$1

python astar_one_way_avg_optimize_manhatan.py \
    --nearest_node_num $nearest_node_num \
    2>&1 | tee ./logger/astar_one_way_avg_optimize_manhatan.log