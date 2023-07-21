from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("data/c006/20220310104945.MTS", 1*3600+48*60, 1*3600+53*60, targetname="data/c006/20220310104945_stt.avi")