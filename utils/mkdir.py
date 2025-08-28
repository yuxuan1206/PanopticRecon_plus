import os


def mkdir_result_folder(root_path):
    os.makedirs(root_path, exist_ok=True)

    os.makedirs(os.path.join(root_path, "grid"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "ply"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "normal_st"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "instance_st"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "semantic_st"), exist_ok=True)
    os.makedirs(os.path.join(root_path, "panoptic_st"), exist_ok=True)

    gaussian_path = os.path.join(root_path, "../rgb")
    os.makedirs(gaussian_path, exist_ok=True)