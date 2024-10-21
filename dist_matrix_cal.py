import util

tsp_path = "/data3/lhy/project/MonaLisaTSP/mona-lisa100K.tsp.txt"
matrix_path = "/data3/lhy/project/MonaLisaTSP/mona_lisa100K_dist_matrix.npy"
points_loc = util.read_tsp_txt(tsp_path)[0]
util.calculate_distance_matrix(points_loc, matrix_path)
