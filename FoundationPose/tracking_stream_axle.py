import pyrealsense2 as rs
from estimater import *
from FoundationPose.mask import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

from seg_axle import generate_combined_mask

parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=4)
parser.add_argument('--track_refine_iter', type=int, default=2)
args = parser.parse_args()

set_logging_format()
set_seed(0)

root = tk.Tk()
root.withdraw()

#mesh_path = filedialog.askopenfilename()
mesh_path = "/home/martyn/martyn/data/sdit01888D53e5s6_Meshed_Decimated_Scaled.ply"
if not mesh_path:
    print("No mesh file selected")
    exit(0)
#mask_file_path = create_mask()
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

#mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)
cam_K = np.array([[605.5885009765625, 0., 326.1221008300781],
                   [0., 603.9918212890625, 253.0368194580078],
                   [0., 0., 1.]])
Estimating = True

# Load the point cloud file using trimesh
point_cloud_file = "/home/martyn/martyn/data/point_cloud_medium.ply"  # Replace with your point cloud file
point_cloud_mesh = trimesh.load(point_cloud_file)
# Extract points from the point cloud
model_points = np.array(point_cloud_mesh.vertices)  # Shape: (N, 3)

# Define the output path for the video
output_video_path = "/home/martyn/martyn/foundationpose_results/live_results/axle/output_video.mp4"  # Replace with your desired file path
# Set video properties
frame_width, frame_height = 640, 480  # Resolution of the video
fps = 30  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
# Initialize the video writer
#video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Define the path for the first frame image
first_frame_path = "/home/martyn/martyn/foundationpose_results/live_results/axle/first_frame.png"  # Replace with your desired path
first_frame_saved = False  # Flag to track if the first frame is saved

# Load Segmentation Model
model = YOLO("/home/martyn/martyn/data/axle_seg.pt")

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
        start_time = time.time()
        if i==0:
            mask = generate_combined_mask(color_image, model)
            if len(mask.shape)==3:
                for c in range(3):
                    if mask[...,c].sum()>0:
                        mask = mask[...,c]
                        break
            mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
            pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
            clean_runtime = time.time() - start_time
            # Print the clean runtime for estimation
            print(f"Clean runtime for estimation: {clean_runtime:.4f} seconds")
        else:
            pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
        #center_pose = pose@np.linalg.inv(to_origin)

        # Measure processing time
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time
        print(f"Frame processing time: {elapsed_time:.4f} seconds, FPS: {fps:.2f}")

        # Visualize projected points
        vis = project_and_draw_points(
            rgb_image=color_image,         # RGB image
            point_cloud=model_points,      # Point cloud from trimesh
            K=cam_K,                       # Camera intrinsic matrix
            pose=pose,                     # Object pose
            point_color=(0, 255, 0),       # Green points
            point_radius=2,                # Small radius for points
            alpha=0.3                      # Transparency level (50% opaque)
        )

        #vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose, bbox=bbox)
        #vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
        
        # Save the first frame
        if not first_frame_saved:
            cv2.imwrite(first_frame_path, vis[..., ::-1])  # Convert RGB to BGR
            first_frame_saved = True
            print(f"First frame saved to {first_frame_path}")
        
        cv2.imshow('1', vis[...,::-1])

        """
        # Write the frame to the video
        video_writer.write(vis[..., ::-1])  # Convert RGB to BGR for OpenCV
        """
        
        # Check for keypress to exit
        key = cv2.waitKey(1) & 0xFF  # Capture the keypress
        if key == ord('q'):  # If 'q' is pressed
            print("Exiting loop...")
            break

        i += 1
        
finally:
    pipeline.stop()
    #video_writer.release()  # Finalize the video
    cv2.destroyAllWindows()  # Close all OpenCV windows
    #print(f"Video saved to {output_video_path}")