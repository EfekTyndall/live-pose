import pyrealsense2 as rs
from estimater import *
from FoundationPose.mask import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_pose_metrics(
    pose_4x4, 
    ground_truth_txt, 
    model_points, 
    scene_name=""
):
    """
    Loads ground-truth transformation from ground_truth_txt,
    computes rotation error, translation error, ADD, etc.
    Returns a dict containing the metrics.
    """
    # 1) Load GT transform (4x4) from text file
    # Suppose we have a 4x4 format in tf_ground_truth.txt:
    gt_transform = np.loadtxt(ground_truth_txt)  # shape (4,4)

    # Ensure shape correctness
    assert gt_transform.shape == (4,4), f"Ground truth in {ground_truth_txt} must be 4x4"

    # Extract R_gt, t_gt
    R_gt = gt_transform[:3, :3]
    t_gt = gt_transform[:3, 3]

    # Extract R_est, t_est from pose_4x4
    R_est = pose_4x4[:3, :3]
    t_est = pose_4x4[:3, 3]

    # Compute rotation error (in degrees)
    # A simple geodesic distance
    R_diff = R.from_matrix(R_gt.T @ R_est)
    rot_err = np.degrees(np.abs(R_diff.magnitude()))

    # Compute translation error (mm) if your data is in mm
    trans_err = np.linalg.norm(t_est - t_gt)

    # Compute ADD
    # transform model_points by GT and by EST, compute average distance
    # model_points shape (N,3)
    model_est = (R_est @ model_points.T).T + t_est
    model_gt  = (R_gt @ model_points.T).T + t_gt
    add = np.mean(np.linalg.norm(model_est - model_gt, axis=1))

    metrics_dict = {
        "Rotation Error (deg)":  rot_err,
        "Translation Error (mm)": trans_err,
        "ADD Metric (mm)": add
    }
    return metrics_dict

def overlay_points_on_image(
    image_bgr, points_3d, pose_4x4, K,
    color=(0,255,0), radius=1
):
    """
    Projects points_3d by pose_4x4 onto image_bgr using camera intrinsics K.
    Draw small circles in 'color' with 'radius'.
    Returns the overlaid image.
    """
    n = points_3d.shape[0]
    ones = np.ones((n,1), dtype=np.float32)
    pts_hom = np.hstack([points_3d, ones])  # (N,4)
    pts_cam = (pose_4x4 @ pts_hom.T).T[:, :3]  # shape (N,3)

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    zs = pts_cam[:,2]
    xs = pts_cam[:,0]*fx/zs + cx
    ys = pts_cam[:,1]*fy/zs + cy

    out_img = image_bgr.copy()
    for x, y, z in zip(xs, ys, zs):
        if z > 0:  # only draw if in front of camera
            cv2.circle(out_img, (int(x), int(y)), radius, color, -1)
    return out_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--est_refine_iter', type=int, default=4, help="Iterations for refine in est.register()")
    parser.add_argument('--scenes_root', type=str,
        default="/home/martyn/Thesis/pose-estimation/data/scenes/",
        help="Root directory containing scene_01, scene_02, ... scene_10"
    )
    parser.add_argument('--output_root', type=str,
        default="/home/martyn/Thesis/pose-estimation/results/methods/foundationpose/",
        help="Root directory for saving results"
    )
    parser.add_argument('--num_scenes', type=int, default=5, help="Number of scenes to process")
    parser.add_argument('--runs_per_scene', type=int, default=5, help="Number of runs per scene")
    args = parser.parse_args()

    # Constants
    mesh_path = "/home/martyn/martyn/data/sdit01888D53e5s6_Meshed_Decimated_Scaled.ply"
    point_cloud_path = "/home/martyn/martyn/data/point_cloud_medium.ply"

    # Intrinsic matrix
    cam_K = np.array([
        [605.5885009765625, 0., 326.1221008300781],
        [0., 603.9918212890625, 253.0368194580078],
        [0., 0., 1.]
    ])

    # Depth scale factor if needed
    depth_scale = 0.001

    # Load the CAD mesh for FoundationPose
    mesh = trimesh.load(mesh_path)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    # Create the FoundationPose object once (same mesh used for all scenes)
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx
    )

    # Load the separate point cloud for overlay
    cad_cloud = trimesh.load(point_cloud_path)
    model_points = np.array(cad_cloud.vertices)  # shape (N, 3), presumably mm

    # Prepare a list of dictionaries for final CSV across all scenes
    all_scenes_summary = []

    for scene_idx in range(1, args.num_scenes + 1):
        scene_name = f"scene_{scene_idx:02d}"
        scene_dir = os.path.join(args.scenes_root, scene_name)

        # Prepare a list to hold run-level metrics for this scene
        run_metrics = []

        # We'll store the path to output for this scene
        scene_output_dir = os.path.join(args.output_root, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)

        # Identify the necessary files in the scene directory
        # e.g. "rgb.png", "depth.png", "mask.png", "tf_ground_truth.txt"
        rgb_path    = os.path.join(scene_dir, "rgb.png")
        depth_path  = os.path.join(scene_dir, "depth.png")
        mask_path   = os.path.join(scene_dir, "mask.png")
        gt_path     = os.path.join(scene_dir, "tf_ground_truth.txt")

        # Pre-load the static data that won't change across runs
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[ERROR] Missing RGB for {scene_name}. Skipping scene.")
            continue

        depth_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            print(f"[ERROR] Missing Depth for {scene_name}. Skipping scene.")
            continue
        depth = depth_raw.astype(np.float32) * depth_scale

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[ERROR] Missing mask for {scene_name}. Skipping scene.")
            continue
        if len(mask.shape) == 3:
            # pick channel
            for c in range(3):
                if mask[...,c].sum() > 0:
                    mask = mask[...,c]
                    break
        mask = mask.astype(bool).astype(np.uint8)

        # Check ground truth file
        #if not os.path.isfile(gt_path):
        #    print(f"[ERROR] Missing ground truth for {scene_name}. Skipping scene.")
        #    continue

        # Now do multiple runs
        for run_idx in range(1, args.runs_per_scene + 1):
            print(f"\n--- {scene_name}, Run {run_idx} ---")
            run_dir = os.path.join(scene_output_dir, f"run_{run_idx:02d}")
            os.makedirs(run_dir, exist_ok=True)

            # Start timing
            start_time = time.time()

            # Pose estimation
            pose = est.register(
                K=cam_K,
                rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                depth=depth,
                ob_mask=mask,
                iteration=args.est_refine_iter
            )

            inference_runtime = time.time() - start_time
            print(f"Inference time: {inference_runtime:.4f}s")

            # Convert (3,4) -> (4,4) if needed
            if pose.shape == (3,4):
                tmp = np.eye(4, dtype=np.float32)
                tmp[:3,:4] = pose
                pose = tmp

            print("Estimated Pose:\n", pose)

            # Compute pose metrics
            #metrics_dict = compute_pose_metrics(
            #    pose_4x4=pose,
            #    ground_truth_txt=gt_path,
            #    model_points=model_points,
            #    scene_name=scene_name
            #)

            # Add runtime
            #metrics_dict["Inference Runtime (s)"] = inference_runtime

            metrics_dict = {"Inference Runtime (s)":  inference_runtime}

            # Save overlay for visualization
            # We'll overlay onto the BGR 'rgb' image
            overlayed = overlay_points_on_image(
                rgb,  # BGR
                model_points,
                pose,
                cam_K,
                color=(0,255,0),
                radius=1
            )
            overlay_path = os.path.join(run_dir, "overlay.png")
            cv2.imwrite(overlay_path, overlayed)
            print(f"Overlay saved to {overlay_path}")

            # Save run-level metrics to a CSV
            # We'll create a small DataFrame with 2 columns: "Metric" & "Value"
            metric_items = list(metrics_dict.items())  # [("Rotation Error (deg)", X), ...]
            df_run = pd.DataFrame(metric_items, columns=["Metric","Value"])
            eval_csv = os.path.join(run_dir, "evaluation_metrics.csv")
            df_run.to_csv(eval_csv, index=False)
            print(f"Metrics saved to {eval_csv}")

            # Also keep them in memory to later compute scene-level stats
            run_metrics.append(metrics_dict)

        # After finishing all runs for this scene, compute average and std
        df_runs = pd.DataFrame(run_metrics)  # each row is a run
        scene_avg = df_runs.mean()  # mean for each metric
        scene_std = df_runs.std()   # std for each metric

        # Save scene-level metrics to scene_metrics.csv
        # We'll create a new DataFrame that has "Metric", "Mean", "Std"
        metric_names = df_runs.columns  # e.g. "Rotation Error (deg)", "ADD Metric (mm)", ...
        rows = []
        for mn in metric_names:
            avg_val = scene_avg[mn]
            std_val = scene_std[mn]
            rows.append([mn, avg_val, std_val])
        df_scene = pd.DataFrame(rows, columns=["Metric","Mean","Std"])
        scene_metrics_csv = os.path.join(scene_output_dir, "scene_metrics.csv")
        df_scene.to_csv(scene_metrics_csv, index=False)
        print(f"Scene-level metrics for {scene_name} saved to {scene_metrics_csv}")

        # We'll store the mean for each metric in a single dictionary (for the all_scenes CSV)
        # Key them as "MetricName Mean", or keep them separate
        scene_summary_dict = {"Scene": scene_name}
        for mn in metric_names:
            scene_summary_dict[f"{mn} Mean"] = scene_avg[mn]
            scene_summary_dict[f"{mn} SD"]  = scene_std[mn]
        all_scenes_summary.append(scene_summary_dict)

    # After all scenes, save a combined CSV
    df_all_scenes = pd.DataFrame(all_scenes_summary)
    all_csv = os.path.join(args.output_root, "all_scenes_average_metrics.csv")
    df_all_scenes.to_csv(all_csv, index=False)
    print(f"\nAll scenes average metrics saved to: {all_csv}")

if __name__ == "__main__":
    main()