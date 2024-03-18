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

FOLDER = Path("D:\\test")  # XXX: edit this path
HEAD_FOLDER = FOLDER / "head_output"
POSE_FOLDER = FOLDER / "pose_output"
ANALYSIS_FOLDER = FOLDER / "analysis_info"
BEST_CAMERAS = ANALYSIS_FOLDER / "best_cameras.csv"
MODEL1_FOLDER = FOLDER / "cross_corr" # fps automatically appended to this name
