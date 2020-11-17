import torch
import math
import matplotlib.pyplot as plt

n_point = 1000000
points = torch.rand((n_point, 2)) * 2 - 1

n_point_in_circle = 0
points_in_circle = []

for point in points:
    r = torch.sqrt(point[0] ** 2 + point[1] ** 2)
    if (r <= 1):
        points_in_circle.append(point)
        n_point_in_circle += 1

points_in_circle = torch.stack(points_in_circle)

plt.plot(points[:, 0].numpy(), points[:, 1].numpy(), 'y.')
plt.plot(points_in_circle[:,0].numpy(), points_in_circle[:,1].numpy(), 'c.')

i = torch.linspace(0, 2 * math.pi)
plt.plot(torch.cos(i).numpy(), torch.sin(i).numpy())
plt.axes().set_aspect('equal')
plt.show()

pi_estimated = 4 * (n_point_in_circle / n_point)
print('Estimated PI value: {}'.format(pi_estimated))