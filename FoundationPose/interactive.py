import pyrealsense2 as rs
from estimater import *
from FoundationPose.mask import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=4)
parser.add_argument('--track_refine_iter', type=int, default=2)
args = parser.parse_args()

set_logging_format()
set_seed(0)

root = tk.Tk()
root.withdraw()

mesh_path = filedialog.askopenfilename()
if not mesh_path:
    print("No mesh file selected")
    exit(0)
mask_file_path = create_mask()
mesh = trimesh.load(mesh_path)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,glctx=glctx)
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
align_to = rs.stream.color
align = rs.align(align_to)

i = 0

mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)
cam_K = np.array([[608.94921875, 0., 326.4563293457031],
                   [0., 607.6509399414062, 240.90924072265625],
                   [0., 0., 1.]])
Estimating = True

# Provided position of the hole in the object's local coordinate system
hole_local = np.array([-0.073616,0.107612,-0.010297])
# Generate a random rotation matrix for the hole's orientation
random_rotation = R.random().as_matrix()  # 3x3 random rotation matrix
# Create a 4x4 transformation matrix for the hole
hole_pose = np.eye(4)  # Start with an identity matrix
# Apply the random orientation (3x3) to the top-left corner of the 4x4 matrix
#hole_pose[:3, :3] = random_rotation
# Set the position of the hole in the object's local coordinates
hole_pose[:3, 3] = hole_local

time.sleep(3)
# Streaming loop
try:
    while Estimating:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)
        if cv2.waitKey(1) == 13:
            Estimating = False
            break        
        H, W = color_image.shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.1) | (depth>=np.inf)] = 0
        if i==0:
            if len(mask.shape)==3:
                for c in range(3):
                    if mask[...,c].sum()>0:
                        mask = mask[...,c]
                        break
            mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
        center_pose = pose@np.linalg.inv(to_origin)
        # Apply the object's center_pose to the hole's transformation matrix
        cam_T_point = pose@hole_pose #(cam_T_CAD) x (CAD_T_point)
        cam_T_point_rotated = cam_T_point@np.linalg.inv(to_origin)
        #vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
        #vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
        # Visualize the object's pose first (bounding box and pose)
        vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
        # Then, visualize the hole's pose on the same image
        vis = draw_xyz_axis(vis, ob_in_cam=cam_T_point_rotated, scale=0.05, K=cam_K, thickness=2, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)        
        i += 1
        
finally:
    pipeline.stop()