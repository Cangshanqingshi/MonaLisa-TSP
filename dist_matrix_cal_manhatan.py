import util

tsp_path = "/data2/lhy/project/MonaLisaTSP/mona-lisa100K.tsp.txt"
matrix_path = "/data2/lhy/project/MonaLisaTSP/mona_lisa100K_dist_matrix_manhatan.npy"
points_loc = util.read_tsp_txt(tsp_path)[0]
util.calculate_distance_matrix_manhatan(points_loc, matrix_path)
