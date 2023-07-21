import cv2
from tqdm import tqdm
import os
from PIL import Image



def get_duration_per_subject(video_list):
    duration_per_subject = {}

    for idx, video_name in enumerate(tqdm(video_list, total=len(video_list))):

        filename = video_name.split(os.sep)[-1][:-4]
        subject = filename.split("_")[0]
        task = filename.split("_")[1]

        if subject not in duration_per_subject:
            duration_per_subject[subject] = 0

        cap = cv2.VideoCapture(video_name)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        duration = frame_count / fps

        duration_per_subject[subject] += duration

    return duration_per_subject


def load_video_by_opencv(video_path, fps=None, step_secs=None, step_in_frame_nums=False):
    frames = []
    video = cv2.VideoCapture(video_path)

    if fps is None:
        fps = video.get(cv2.CAP_PROP_FPS)

    if step_secs is None:
        step_frame = 1 / fps
    else:
        step_frame = int(fps * step_secs)

    if step_in_frame_nums:
        step_frame = step_secs

    count = 0

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            frames.append(frame)

            if step_secs is not None:
                count += step_frame
                video.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            video.release()

    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    return frames


if __name__ == "__main__":

    from utils import get_all_files_recursively_by_ext, ensure_dir

    # Set 1% data of a subject for manual labelling. Adjust it so that you can get about 100-150 jpgs for each subject.
    ratio = 1 / 100

    # Where do you want to output your samples? Please use full path, not relative path for convenience.
    output_directory = r"F:\\Isabella\yolo_samples"

    video_list = get_all_files_recursively_by_ext("videos", "mp4")
    duration_per_subject = get_duration_per_subject(video_list)
    seen_subject = []
    count = 0


    for idx, video_name in enumerate(tqdm(video_list, total=len(video_list))):

        filename = video_name.split(os.sep)[-1][:-4]
        subject = filename.split("_")[0]
        task = filename.split("_")[1]

        if subject not in seen_subject:
            seen_subject += subject
            count = 0

        cap = cv2.VideoCapture(video_name)
        fps = cap.get(cv2.CAP_PROP_FPS)

        step_secs = duration_per_subject[subject] / 100
        frames = load_video_by_opencv(video_name, fps=fps, step_secs=step_secs)

        for frame in frames:

            output_filename = os.path.join(output_directory, subject,
                                           "{}_{}.jpg".format(task, count))
            ensure_dir(output_filename)

            if os.path.isfile(output_filename):
                continue
            frame.save(output_filename)
            count += 1