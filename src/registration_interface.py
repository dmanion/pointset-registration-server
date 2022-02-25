import copy
import math
import os
from io import BytesIO
import datetime
import numpy as np
import open3d as o3d
from flask import Flask, request, jsonify
from probreg import gmmtree

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                         [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                         [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                         [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


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
    source = o3d.io.read_triangle_mesh(mesh_path).sample_points_uniformly()
    target = o3d.io.read_point_cloud(pcd_path)
    threshold = 0.02

    ## show initial
    print("Initial alignment is:")
    print(trans_init)
    #ev = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    #print(ev)

    ## start
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20), )
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    #draw_registration_result(source, target, reg_p2p.transformation)
    return reg_p2p.transformation


if __name__ == '__main__':
    app = Flask(__name__)
    result_p = np.zeros(3)
    result_R = np.eye(3)
    mesh_location=f"{os.getcwd()}/body_upper.STL.ply"

    @app.route("/")
    def index():
        return "Hello Flask!"
    @app.route("/upload/pcd-bin", methods=["POST"])
    def get_pcd():
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        pcd_filename = f"{os.getcwd()}/{now:%Y%m%d%H%M%S}.pcd"
        with open(pcd_filename, "wb") as f:
            f.write(request.data)

        ## pcd is rgbd input, so it's pinned along the world coordinate.
        ## mesh is aligned manually, so the initial pose is stored in mesh_initpose.
        init_R = np.array([
            float(request.args.get("rot_w")),
            float(request.args.get("rot_x")),
            float(request.args.get("rot_y")),
            float(request.args.get("rot_z")),
        ])
        init_p = np.array([
            float(request.args.get("cam_x")),
            float(request.args.get("cam_y")),
            float(request.args.get("cam_z")),
        ])
        homomat = np.eye(4)
        homomat[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(init_R)
        homomat[:3, 3] = init_p
        result = reg_pts_mesh(pcd_filename, mesh_location, homomat)
        result_p = result[:3, 3]
        result_R = result[:3, 3]

        return jsonify([0])


    @app.route("/0/position")
    def sendback_pos():
        key = ["x", "y", "z"]
        data = jsonify([dict(zip(key, result_p))])
        return data

    @app.route("/0/orientation")
    def sendback_rot():
        key = ["w", "x", "y", "z"]
        data = jsonify([dict(zip(key, quaternion_from_matrix(result_R)))])
        return data

    app.run()
