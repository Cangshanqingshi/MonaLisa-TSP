import util

json_path = "/data2/lhy/project/MonaLisaTSP/result/astar_one_way_avg_20_result.json"
tsp_path = "/data2/lhy/project/MonaLisaTSP/mona-lisa100K.tsp.txt"

node_order = util.load_json(json_path)["node order"]
point_loc = util.read_tsp_txt(tsp_path)[0]

img_path = "/data2/lhy/project/MonaLisaTSP/img/astar_one_avg_0.jpg"

util.draw_path(node_order, point_loc, img_path)