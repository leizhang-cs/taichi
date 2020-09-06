import taichi as ti

ti.init()

screen_res = (400, 200)
boundary = (screen_res[0] / 10, screen_res[1] / 10)
dim = 2
N = 100
x = ti.Vector.field(dim, float, N, needs_grad = True)  # position of particles
gui_x = ti.Vector.field(dim, float, N)
u = ti.Vector.field(dim, float, N)  # velocity of particles
P = ti.field(float, N)  # pressure
rho = ti.field(float, N)  # scalar field
grad_P = ti.Vector.field(dim, float, N)  # grad of pressure
dt = 0.005
m = 1.0 # mass
h = 1.0 # neighbor threshold radius
B = 1119 # P = B ((rho/rho0)**gamma - 1)
rho0 = 1000.0
gamma = 7
pi = 3.14
collision_vec = (-0.8, -0.8)
small_t = 1e-4


#kernel function
@ti.func
def W(r, h):
    if(r > 0 and r < h):
        return 15 / pi / h**6 * (h - r)**3
    else:
        return 0 

@ti.kernel
def compute_rho():
    for i in x:
        for j in range(N):
                r = x[i] - x[j]
                rho[i] += m * W(r.norm(), h)

@ti.kernel
def compute_P():
    for i in x:
        P[i] = B * ((rho[i]/rho0)**gamma - 1)
    for i in x:
        temp = ti.Vector([0.0, 0.0])
        for j in range(N):
            r = x[i] - x[j]
            temp += m * (P[i]/(rho[i]*rho[i]) + P[j]/(rho[j]*rho[j])) * x.grad[i] * W(r.norm(), h)
        grad_P[i] = rho[i] * temp


@ti.kernel
def sympletic_euler():
    g = ti.Vector([0.0, -9.8])
    random_f = ti.Vector([ti.random()-0.5, ti.random()])
    for i in x:
        u[i] += dt * ( (-1/rho[i]) * grad_P[i] + g + random_f)
    for i in x:
        x[i] += dt * u[i]
        # boundary condition
        flag = False
        if x[i][1] < -small_t:
            x[i][1] = small_t
            flag = True
        if x[i][0] < -small_t:
            x[i][0] = small_t
            flag = True
        if x[i][0] > boundary[0]:
            x[i][0] = boundary[0] - small_t
            flag = True
        if flag:
            u[i] *= collision_vec
        gui_x[i] = [x[i][0] / boundary[0], x[i][1] / boundary[1]]

def substep():
    compute_rho()
    compute_P()
    sympletic_euler()

@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random() * boundary[0], ti.random() * boundary[1]]
        rho[i] = rho0


init()

gui = ti.GUI('WCSPH', screen_res)
while gui.running:
    for i in range(10):
        substep()
    gui.circles(gui_x.to_numpy(), radius=2)
    gui.show()

