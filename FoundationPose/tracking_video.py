from estimater import *
from FoundationPose.mask import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.spatial.transform import Rotation as R

# ------------------------------------------------------------------------------
# ARGUMENT PARSING
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=4)
parser.add_argument('--track_refine_iter', type=int, default=2)
args = parser.parse_args()

set_logging_format()
set_seed(0)

# ------------------------------------------------------------------------------
# GUI/FILE DIALOG FOR MESH SELECTION
# ------------------------------------------------------------------------------
root = tk.Tk()
root.withdraw()

mesh_path = filedialog.askopenfilename()
if not mesh_path:
    print("No mesh file selected")
    sys.exit(0)

mask_file_path = "/home/martyn/Thesis/pose-tracking/data/frames/frames_part/scene_02/masks/1737366013729.png"

# ------------------------------------------------------------------------------
# LOAD THE MESH & PREPARE BOUNDING BOX
# ------------------------------------------------------------------------------
mesh = trimesh.load(mesh_path)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

# ------------------------------------------------------------------------------
# CREATE SCORER, REFINER, FOUNDATIONPOSE INSTANCE
# ------------------------------------------------------------------------------
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    glctx=glctx
)

# ------------------------------------------------------------------------------
# INTRINSIC MATRIX (SAMPLE VALUES)
# Adjust if your recorded images have different intrinsics
# ------------------------------------------------------------------------------
cam_K = np.array([
    [605.5885,    0.,      326.1221],
    [   0.,    603.9918,  253.0368],
    [   0.,        0.,      1.     ]
])

# ------------------------------------------------------------------------------
# POINT CLOUD FILE (FOR VISUALIZATION)
# ------------------------------------------------------------------------------
point_cloud_file = "/home/martyn/Thesis/pose-estimation/data/point-clouds/A6544132042_003_point_cloud_scaled.ply"
point_cloud_mesh = trimesh.load(point_cloud_file)
model_points = np.array(point_cloud_mesh.vertices)  # (N, 3)

# ------------------------------------------------------------------------------
# OUTPUT VIDEO SETUP (OPTIONAL)
# ------------------------------------------------------------------------------
output_dir = "/home/martyn/Thesis/pose-tracking/results/part/methods/foundationpose_test/"
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, "output_video.mp4")

# ------------------------------------------------------------------------------
# OUTPUT FRAMES SETUP (OPTIONAL)
# ------------------------------------------------------------------------------
output_frames_dir = os.path.join(output_dir, "output_frames")
os.makedirs(output_frames_dir, exist_ok=True)

frame_width, frame_height = 640, 480  # Adjust if necessary
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# ------------------------------------------------------------------------------
# OPTIONAL FIRST-FRAME SAVING
# ------------------------------------------------------------------------------
first_frame_path = os.path.join(output_dir, "first_frame.png")
first_frame_saved = False

# ------------------------------------------------------------------------------
# LOAD YOUR MASK
# ------------------------------------------------------------------------------
mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)

# ------------------------------------------------------------------------------
# FOLDERS WITH PRE-RECORDED FRAMES
# ------------------------------------------------------------------------------
scene_dir = "/home/martyn/Thesis/pose-tracking/data/frames/frames_part/scene_02/"
color_folder = os.path.join(scene_dir, "rgb/")
depth_folder = os.path.join(scene_dir, "depth/")

color_files = sorted([
    f for f in os.listdir(color_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])
depth_files = sorted([
    f for f in os.listdir(depth_folder)
    if f.lower().endswith(('.png', '.tiff', '.tif', '.npy', '.exr'))
])

if len(color_files) != len(depth_files):
    print("Mismatch in number of color vs. depth frames!")
    sys.exit(1)

# ------------------------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------------------------
est_time = None           # Time for the first-frame estimation
tracking_times = []       # List of tracking times for subsequent frames
fps_values = []           # FPS for each processed frame

# ------------------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------------------
Estimating = True
i = 0

try:
    for color_filename, depth_filename in zip(color_files, depth_files):
        if not Estimating:
            break

        color_path = os.path.join(color_folder, color_filename)
        depth_path = os.path.join(depth_folder, depth_filename)

        # Read color (BGR by default in OpenCV)
        color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
        if color_bgr is None:
            print(f"Could not read color file: {color_path}")
            continue

        # Convert BGR to RGB for your pipeline
        color_image = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # Read depth (UNCHANGED to preserve 16-bit if needed)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            print(f"Could not read depth file: {depth_path}")
            continue

        # Convert from millimeters to meters if needed
        depth_image = depth_raw.astype(np.float32) / 1000.0

        # Resize depth to match color (if needed)
        H, W = color_image.shape[:2]
        depth_image = cv2.resize(depth_image, (W, H), interpolation=cv2.INTER_NEAREST)

        # Filter invalid depths
        depth_image[(depth_image < 0.1) | (depth_image >= np.inf)] = 0

        # Time the pose operation
        start_time = time.time()

        # First frame => register
        if i == 0:
            # Ensure mask matches the image size
            if len(mask.shape) == 3:
                for c in range(3):
                    if mask[..., c].sum() > 0:
                        mask = mask[..., c]
                        break
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)

            pose = est.register(
                K=cam_K,
                rgb=color_image,
                depth=depth_image,
                ob_mask=mask,
                iteration=args.est_refine_iter
            )
            clean_runtime = time.time() - start_time
            est_time = clean_runtime  # Save the estimation time
            print(f"Clean runtime for estimation (frame 0): {clean_runtime:.4f} seconds")

        # Subsequent frames => track
        else:
            pose = est.track_one(
                rgb=color_image,
                depth=depth_image,
                K=cam_K,
                iteration=args.track_refine_iter
            )

        # Compute pose in the "center" frame of reference
        center_pose = pose @ np.linalg.inv(to_origin)

        elapsed_time = time.time() - start_time
        current_fps = 1.0 / elapsed_time

        # If not the first frame, accumulate tracking time
        if i > 0:
            tracking_times.append(elapsed_time)

        # For any frame, accumulate FPS
        fps_values.append(current_fps)

        # Diagnostic prints for each frame
        print(f"Frame {i}: processing time {elapsed_time:.4f}s, FPS: {current_fps:.2f}")

        # Visualization
        vis = project_and_draw_points(
            rgb_image=color_image,
            point_cloud=model_points,
            K=cam_K,
            pose=pose,
            point_color=(0, 255, 0),
            point_radius=2,
            alpha=0.5
        )

        # Save the first frame if not done
        if (i == 0) and (not first_frame_saved):
            cv2.imwrite(first_frame_path, vis[..., ::-1])  # Convert RGB->BGR
            first_frame_saved = True
            print(f"First frame saved to {first_frame_path}")

        # Draw FPS on the frame for display
        vis_bgr = vis[..., ::-1].copy()  # Convert RGB->BGR
        # Show in a window (optional)
        cv2.imshow('Pose Estimation', vis_bgr)

        # Write frame to output video
        video_writer.write(vis_bgr)  # BGR format

        # Save each frames as PNG
        frame_png = os.path.join(output_frames_dir, f"frame_{i:06d}.png")
        cv2.imwrite(frame_png, vis_bgr)

        # Check for 'q' to quit
        key = cv2.waitKey(max(1, int(1000 / fps - elapsed_time * 1000))) & 0xFF
        if key == ord('q'):
            print("Exiting loop...")
            break

        i += 1

except KeyboardInterrupt:
    print("User interrupted the process.")

finally:
    # Final cleanup
    Estimating = False
    cv2.destroyAllWindows()
    video_writer.release()
    print(f"Video saved to {output_video_path}")

    # --------------------------------------------------------------------------
    # COMPUTE AND SAVE STATS
    # --------------------------------------------------------------------------
    # If there is at least one registration frame
    if est_time is not None:
        first_frame_est = est_time
    else:
        first_frame_est = 0.0

    # Compute average tracking time
    if len(tracking_times) > 0:
        avg_track_time = sum(tracking_times) / len(tracking_times)
    else:
        avg_track_time = 0.0

    # Compute average FPS
    if len(fps_values) > 0:
        avg_fps = sum(fps_values) / len(fps_values)
    else:
        avg_fps = 0.0

    # Prepare results
    results = [
        f"Estimation time on first frame: {first_frame_est:.4f} seconds",
        f"Average tracking time (subsequent frames): {avg_track_time:.4f} seconds",
        f"Average FPS (all frames): {avg_fps:.2f}"
    ]

    # Print to console
    print("\n----- Performance Summary -----")
    for line in results:
        print(line)

    # Save to a text file
    stats_file = os.path.join(output_dir, "runtime.txt")
    with open(stats_file, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Runtime saved to {stats_file}")