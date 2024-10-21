import numpy as np

a = np.array([10, 10, 25, 20, 35, 30, 60, 30, 79])
dist = np.random.rand(len(a), len(a))

print(np.where(a == 30))
print(np.where(a == 30)[0])
print(np.where(a == 30)[0][0])

print('-'*50)
print(np.where(a == 25))
print(np.where(a == 25)[0])
print(np.where(a == 25)[0][0])

print('-'*50)
print(dist[0][np.where(a == 25)[0]])
print(dist[np.where(a == 25)[0][0]][0])
print(dist[0][np.where(a == 25)[0]] + dist[np.where(a == 25)[0][0]][0])
print('-'*50)
print(dist[0][np.where(a == 30)[0][0]] + dist[np.where(a == 30)[0][0]][0])
