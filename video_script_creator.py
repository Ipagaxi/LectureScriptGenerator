from mmap import PAGESIZE
from os import name
from turtle import width
from scenedetect import VideoManager, SceneManager, detect, AdaptiveDetector, ContentDetector, save_images, open_video, split_video_ffmpeg
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import cm, inch
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.rl_config import defaultPageSize
from reportlab.lib.styles import getSampleStyleSheet
from faster_whisper import WhisperModel
import os
import sys
import time


# Dividing video into sub scenes and taking screenshots
def scene_splitting_and_screenshot(path_to_video):
    print("Detecting scenes ...")
    video_file = path_to_video.split(".")
    video = open_video(path_to_video)

    video_manager = VideoManager([path_to_video])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=7.0, min_scene_len=90))

    video_manager.set_downscale_factor(2)  # Process at half resolution
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    num_scenes = len(scene_list)
    print("Fount %d scenes in Video!" % (num_scenes))
    print("Take scene screenshots ...")
    save_images(scene_list=scene_list, video=video, num_images=3, output_dir="%s_screenshots" % (video_file[0]), show_progress=True)
    print("Scene screenshots taken!")
    print("Split video ...")
    if not os.path.exists("%s_scenes" % (video_file[0])):
        os.makedirs("%s_scenes" % (video_file[0]))
    split_video_ffmpeg(path_to_video, scene_list, output_file_template='%s_scenes/%s-Scene-$SCENE_NUMBER.%s' % (video_file[0], video_file[0], video_file[1]), show_progress=True)
    print("Video splitted!")
    return num_scenes


# models: tiny, base, small, medium, large
def transcribe_videos(model_name, path_to_video, num_scenes):
    print("Transcribe scenes ...")
    print("This may take a while ...")
    video_file = path_to_video.split(".")
    scene_texts = []
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    for i in range(num_scenes):
        num = str(i+1).zfill(3)
        segments, info = model.transcribe("%s_scenes/%s-Scene-%s.%s" % (video_file[0], video_file[0], num, video_file[1]))
        if not os.path.exists("%s_texts" % (video_file[0])):
            os.makedirs("%s_texts" % (video_file[0]))
        combined_segments = ""
        with open("%s_texts/%s-%s.txt" % (video_file[0], video_file[0], num), "w", encoding="utf-8") as f:
            for segment in segments:
                combined_segments += segment.text
                f.write(segment.text)
        scene_texts.append("%s " % (combined_segments))
        print("%d. scene transcribed ..." % (i+1))
    print("All scenes transcribed!")
    return scene_texts

def load_transcripts(path_to_video, num_scenes):
    print("Load transcripts ...")
    video_file = path_to_video.split(".")
    scene_texts = []
    for i in range(num_scenes) :
        num = str(i+1).zfill(3)
        f_read = open("%s_texts/%s-%s.txt" % (video_file[0], video_file[0], num))
        scene_texts.append(f_read.read())
        f_read.close()
        print("%d.scene transcript loaded ..." % (i))
    print("All transcripts loaded!")
    return scene_texts


def generatePDF(path_to_video, scene_texts):
    print("Generate PDF ...")
    video_file = path_to_video.split(".")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate("%s.pdf" % (video_file[0]))
    Story = []
    style = styles["Normal"]
    tmp_img = ImageReader("%s_screenshots/%s-Scene-001-01.jpg" % (video_file[0], video_file[0]))
    img_w, img_h = tmp_img.getSize()
    scale = 240.0 / img_h
    img_w *= scale
    img_h = 240.0
    print("scaled width: " + str(img_w))
    print("scale height: " + str(img_h))
    for i, text in enumerate(scene_texts) :
        num = str(i+1).zfill(3)
        img = Image("%s_screenshots/%s-Scene-%s-03.jpg" % (video_file[0], video_file[0], num), width=img_w, height=img_h)
        p = Paragraph(text, style)
        Story.append(img)
        Story.append(Spacer(1,0.2*inch))
        Story.append(p)
        Story.append(Spacer(1,0.2*inch))
    doc.build(Story)
    print("Generated %s.pdf successfully!" % (video_file[0]))

start = time.time()
path_to_video = sys.argv[1]
whisper_model = sys.argv[2]
video_file = path_to_video.split(".")
num_scenes = scene_splitting_and_screenshot(path_to_video)
if os.path.exists("%s_texts/%s-001.txt" % (video_file[0], video_file[0])):
    print("Found existing transcripts!")
    scene_texts = load_transcripts(path_to_video, num_scenes)
else:
    scene_texts = transcribe_videos(whisper_model, path_to_video, num_scenes)
generatePDF(path_to_video, scene_texts)
end = time.time()
print(f"Total runtime of the program is {(end - start) / 60.0} minutes")