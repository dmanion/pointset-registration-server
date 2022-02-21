import copy
import os
from io import BytesIO
import datetime
import numpy as np
import open3d as o3d
from flask import Flask, request, render_template
from probreg import gmmtree


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def reg_pts_mesh(pcd_path, mesh_path, trans_init):
    source = o3d.io.read_point_cloud(mesh_path)
    target = o3d.io.read_point_cloud(pcd_path)
    threshold = 0.02

    ## show initial
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    ev = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(ev)

    ## start
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000), )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)
    return reg_p2p.transformation


if __name__ == '__main__':
    app = Flask(__name__)
    result_p = np.zeros(3)
    result_R = np.eye(3)
    mesh_location=f"{os.getcwd()}/body_upper.wrl"

    @app.route("/")
    def index():
        return "Hello Flask!"
    @app.route("/upload/pcd-bin", methods=["POST"])
    def get_pcd():
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        pcd_filename = f"{os.getcwd()}/{now:%Y%m%d%H%M%S}.bin"
        with open(pcd_filename, "wb") as f:
            f.write(request.data)

        ## pcd is rgbd input, so it's pinned along the world coordinate.
        ## mesh is aligned manually, so the initial pose is stored in mesh_initpose.
        init_R = np.ndarray([
            request.args.get("rot_w"),
            request.args.get("rot_x"),
            request.args.get("rot_y"),
            request.args.get("rot_z"),
        ])
        init_p = np.array([
            request.args.get("cam_x"),
            request.args.get("cam_y"),
            request.args.get("cam_z"),
        ])
        homomat = np.eye(4)
        homomat[:3, :3] = init_R
        homomat[:3, 3] = init_p
        mesh_initpose = o3d.geometry.get_rotation_matrix_from_quaternion(init_R)
        reg_pts_mesh(pcd_filename, mesh_location, mesh_initpose)

    app.run()
