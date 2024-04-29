from pathlib import Path

# Folder structure: ------------------------------------------------------
# ANALYSIS_FOLDER
#     analysis_info.csv
#     best_cameras.csv
#     data_quality.csv

# HEAD_FOLDER
#     ppt_folder
#         ppt_folder
#             labels
#                 .txt
#             .mp4
#         .csv (output file)

# POSE_FOLDER
#     ppt_folder
#         csv_files
#         json files
#         output_videos
#             vid.mp4
#         video_info
# ----------------------------------------------------------------------

FOLDER = Path("D:\\KHULA_jsons")  # XXX: edit this path
HEAD_FOLDER = FOLDER / "head_output"
POSE_FOLDER = FOLDER / "pose_output"
ANALYSIS_FOLDER = FOLDER / "analysis_info"
FRAME_CHECKS = ANALYSIS_FOLDER / "frame_checks"
BEST_CAMERAS = ANALYSIS_FOLDER / "best_cameras"
MODEL1_FOLDER = FOLDER / "cross_corr" # fps automatically appended to this name
MODEL2_FOLDER = FOLDER / "mdrqa" # fps automatically appended to this name
MODEL3_FOLDER = FOLDER / "graph_network"

KEYPOINTS_DICT = {0: "Nose",
                  1: "Neck",
                  2: "RShoulder",
                  3: "RElbow",
                  4: "RWrist",
                  5: "LShoulder",
                  6: "LElbow",
                  7: "LWrist",
                  8: "MidHip",
                  9: "RHip",
                  10: "RKnee",
                  11: "RAnkle",
                  12: "LHip",
                  13: "LKnee",
                  14: "LAnkle",
                  15: "REye",
                  16: "LEye",
                  17: "REar",
                  18: "LEar",
                  19: "LBigToe",
                  20: "LSmallToe",
                  21: "LHeel",
                  22: "RBigToe",
                  23: "RSmallToe",
                  24: "RHeel"}