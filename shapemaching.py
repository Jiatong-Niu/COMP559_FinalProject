import taichi as ti
import math
import numpy as np
ti.init()

# by changing this parameter, you should be able to test different testcases
# each testcase has different number of particles and depends on the complexity, I set different time step since forward euler sometimes explode.:)
testcase = 0
if testcase == 0:
    num_particles = 7
    dt = 1 / 400
if testcase == 1:
    num_particles = 3
    dt = 1 / 1000
if testcase == 2:
    num_particles = 6
    dt = 1 / 1000
if testcase == 3:
    num_particles = 1210
    dt = 1 / 1500

dim = 3
world_scale_factor = 1.0 /100
mass_inv = 1.0
rotation = 30
# create the initial parameter that we may use in the future
positions = ti.Vector.field(dim, float, num_particles)
origin_p = ti.Vector.field(dim, float, num_particles)
radius_vector = ti.Vector.field(dim, float, num_particles)


velocities = ti.Vector.field(dim, float, num_particles)
pos_draw = ti.Vector.field(dim, float, num_particles)
force = ti.Vector.field(dim, float, num_particles)
penalty_force = ti.Vector.field(dim, float, num_particles)

paused = ti.field(ti.i32, shape=())
is_collided = ti.field(ti.i32, num_particles)
any_is_collided = ti.field(ti.i32, shape=())

#  We assume the only externel force in this project is gravity
gravity = ti.Vector([0.0, -9.8, 0.0])


# test case 1 initialization
@ti.kernel
def init_particles0():
    init_pos = ti.Vector([50.0, 50.0, -10.0])
    positions[0] = ti.Vector([0, 0, 0]) + init_pos
    positions[1] = ti.Vector([0, 0, 1.5]) + init_pos
    positions[2] = ti.Vector([0, 1.5, 0]) + init_pos
    positions[3] = ti.Vector([1.5, 0, 0]) + init_pos
    positions[4] = ti.Vector([0, 0, -1.5]) + init_pos
    positions[5] = ti.Vector([0, -1.5, 0]) + init_pos
    positions[6] = ti.Vector([-1.5, 0, 0]) + init_pos


# test case 2 initialization
@ti.kernel
def init_particles1():
    init_pos = ti.Vector([50.0, 50.0, -50.0])
    positions[0] = ti.Vector([0, 0, 0]) + init_pos
    positions[1] = ti.Vector([2, 0, 0]) + init_pos
    positions[2] = ti.Vector([4, 0, 0]) + init_pos


# test case 3 initialization
@ti.kernel
def init_particles2():
    init_pos = ti.Vector([50.0, 50.0, -50.0])
    positions[0] = ti.Vector([0, 0, 0]) + init_pos
    positions[1] = ti.Vector([2, 0, 0]) + init_pos
    positions[2] = ti.Vector([4, 0, 0]) + init_pos
    positions[3] = ti.Vector([0, 2, 0]) + init_pos
    positions[4] = ti.Vector([2, 2, 0]) + init_pos
    positions[5] = ti.Vector([4, 2, 0]) + init_pos


# test case 4 initialization
@ti.kernel
def init_particles3():
    init_pos = ti.Vector([50.0, 50.0, 0.0])
    cube_size = 20
    spacing = 2
    num_per_row = (int)(cube_size // spacing) + 1
    num_per_floor = num_per_row * num_per_row
    for i in range(num_particles):
        floor = i // (num_per_floor)
        row = (i % num_per_floor) // num_per_row
        col = (i % num_per_floor) % num_per_row
        positions[i] = ti.Vector([col * spacing, floor * spacing, row * spacing]) + init_pos


# Detect whether the object hit the floor
@ti.kernel
def collision_detection():
    for i in range(num_particles):
        if positions[i].y < 0:
            is_collided[i] = True
            any_is_collided[None] = True


@ti.kernel
def find_center_of_mass():
    total_mass = ti.Vector([0.0, 0.0, 0.0])
    # we assume the particles has the same weight mass_inv
    for i in range(num_particles):
        total_mass += positions[i]*mass_inv
    center_mass = total_mass/(num_particles*mass_inv)
    for i in range(num_particles):
        radius_vector[i] = positions[i] - center_mass


@ti.kernel
def shape_matching():

    # We use forward euler rule to update the position of particles
    for i in range(num_particles):
        origin_p[i] = positions[i]
        force[i] = gravity + penalty_force[i]
        velocities[i] += mass_inv * force[i] * dt
        positions[i] += velocities[i] * dt

    # compute the new(matched shape) mass center
    total_mass = ti.Vector([0.0, 0.0, 0.0])
    # we assume the particles has the same weight mass_inv
    for i in range(num_particles):
        total_mass += positions[i] * mass_inv
    center_mass = total_mass / (num_particles * mass_inv)

    # compute transformation matrix and extract rotation
    sum1 = sum2 = ti.Matrix([[0.0] * 3 for _ in range(3)], ti.f64)
    for i in range(num_particles):
        sum1 += (positions[i] - center_mass).outer_product(radius_vector[i])
        sum2 += radius_vector[i].outer_product(radius_vector[i])
    A = sum1 @ sum2.inverse()
    R, _ = ti.polar_decompose(A)

    # update velocities and positions
    for i in range(num_particles):
        positions[i] = center_mass + R @ radius_vector[i]
        velocities[i] = (positions[i] - origin_p[i]) / dt


@ti.kernel
def update_vel_pos():
    ## updateing with forward euler
    for i in range(num_particles):
        force[i] = gravity + penalty_force[i]
        velocities[i] += mass_inv * force[i] * dt
        positions[i] += velocities[i] * dt

# define how the particles interact with the floor
@ti.kernel
def collision_response():
    eps = 2.0  # the padding to prevent penatrating
    k = 100.0  # stiffness of the penalty force
    boundary = ti.Matrix([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]], ti.f64)
    boundary[0, :] = boundary[0, :] + eps
    boundary[1, :] = boundary[1, :] - eps
    for i in range(num_particles):
        if positions[i].y < boundary[0, 1]:
            n_dir = ti.Vector([0.0, 1.0, 0.0])
            phi = positions[i].y - boundary[0, 1]
            penalty_force[i] = k * ti.abs(phi) * n_dir

        if positions[i].x < boundary[0, 1]:
            n_dir = ti.Vector([-1.0, 0.0, 0.0])
            phi = positions[i].x - boundary[1, 0]
            penalty_force[i] = k * ti.abs(phi) * n_dir




def substep():
    #before detecting collision clear everything
    penalty_force.fill(0.0)
    force.fill(0.0)
    any_is_collided.fill(0)
    is_collided.fill(0)
    collision_detection()

    if any_is_collided[None]:
        # if collision happens response and do shape matching
        collision_response()
        shape_matching()
    else:
        update_vel_pos()


# init the window, canvas, scene and camerea
window = ti.ui.Window("rigidbody", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1.0, 1.0, 1.0))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# initial camera position
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)


def main():
    if testcase == 0:
        init_particles0()
    if testcase == 1:
        init_particles1()
    if testcase == 2:
        init_particles2()
    if testcase == 3:
        init_particles3()

    if(rotation != 0):

        theta = rotation/ 180.0 * math.pi
        R = ti.Matrix([
            [ti.cos(theta), -ti.sin(theta), 0.0],
            [ti.sin(theta), ti.cos(theta), 0.0],
            [0.0, 0.0, 1.0]
        ])
        for i in range(num_particles):
            positions[i] = R @ positions[i]
    find_center_of_mass()

    paused[None] = True
    while window.running:
        if window.get_event(ti.ui.PRESS):
            # press space to pause the simulation
            if window.event.key == ti.ui.SPACE:
                paused[None] = not paused[None]


        # do the simulation in each step
        if (paused[None] == False):
            for i in range(int(0.05 / dt)):
                substep()

        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.ambient_light((0.5, 0.5, 0.5))

        # draw particles for better visualization add scale factor
        for i in range(num_particles):
            pos_draw[i] = positions[i] * world_scale_factor

        scene.particles(pos_draw, radius=0.03, color=(0.6, 0.5, 0.5))

        # show the frame
        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()
