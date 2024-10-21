nearest_node_num=$1

python astar_two_way_optimize.py \
    --nearest_node_num $nearest_node_num \
    2>&1 | tee ./logger/astar_two_way_optimize.log