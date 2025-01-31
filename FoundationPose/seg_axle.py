import numpy as np
import time

def generate_combined_mask(
    rgb_image, model, device='cuda:0', imgsz=(480, 640), conf=0.5
):
    # Start timer
    start_time = time.time()

    # Run inference
    results = model.predict(rgb_image, imgsz=imgsz, conf=conf, device=device)
    result = results[0]

    # Combine masks and calculate the bounding box
    height, width = result.masks.data[0].shape[-2:]
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for mask in result.masks.data:
        binary_mask = mask.cpu().numpy().astype(np.uint8)
        combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)

    # Convert to binary format
    combined_mask = (combined_mask > 0).astype(np.uint8)

    # End timer
    segmentation_time = time.time() - start_time
    print("Segmentation time (s):", segmentation_time)

    # Logging
    print(f"Combined mask shape: {combined_mask.shape}")
    print(f"Number of non-zero pixels in mask: {np.sum(combined_mask)}")

    # Return only the combined mask and segmentation time
    return combined_mask