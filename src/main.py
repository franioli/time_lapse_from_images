import cv2
import numpy as np
import os
import exifread
import argparse
import fnmatch

from PIL import Image
from pathlib import Path
from typing import List, Tuple, Union
from tqdm import tqdm
from datetime import datetime

# Video Generating function


def generate_video(
    image_dir: Union[str, Path],
    video_name: str = "out.avi",
    im_ext: str = None,
    resize_fct=1,
    fourcc: int = 0,
    fps: float = 10,
    overlay_date: bool = True,
    logo_path: str = None,
) -> None:

    image_dir = Path(image_dir)
    if im_ext:
        # Make glob search case-insensitive (using bash expansions)
        assert (
            len(im_ext) == 3
        ), "Wrong extension input. Please, provide extension with 3 characters only (e.g., jpg)"
        suf = [f"[{x.lower()}{x.upper()}]" for x in im_ext]
        im_list = sorted(image_dir.glob(f"*.{suf[0]}{suf[1]}{suf[2]}"))
    else:
        im_list = sorted(image_dir.glob("*"))

    # setting the frame width, height width
    # the width, height of first image
    frame = cv2.imread(str(im_list[0]))
    height, width, _ = frame.shape
    size = tuple([width, height])

    # Size casted to int
    if resize_fct != 1:
        size = tuple([int(x / resize_fct) for x in size])
        print(f"Resizing images to {size[0]}x{size[1]}")

    video = cv2.VideoWriter(video_name, fourcc, fps, size)

    print("Generating video: ")
    # Appending the images to the video one by one
    for file in tqdm(im_list):
        if not file.is_file():
            continue

        if overlay_date:
            im = overlay_info_on_image(str(file), logo_path=logo_path)
        else:
            im = cv2.imread(str(file))

        if resize_fct != 1:
            im = cv2.resize(im, size, cv2.INTER_LANCZOS4)

        video.write(im)

    video.release()


def overlay_info_on_image(
    im_path: Union[Path, str],
    logo_path: Union[Path, str] = None,
) -> np.ndarray:
    f = open(im_path, "rb")
    tags = exifread.process_file(f, details=False, stop_tag="DateTimeOriginal")
    # if "Image DateTime" in tags.keys():
    date_str = tags["EXIF DateTimeOriginal"].values
    date_fmt = "%Y:%m:%d %H:%M:%S"
    date_time = datetime.strptime(date_str, date_fmt)
    over_str = f"{date_time.year}-{date_time.month:02}-{date_time.day:02}"

    image = cv2.imread(str(im_path))
    h, w, _ = image.shape
    font = cv2.FONT_HERSHEY_DUPLEX
    bottomLeftCornerOfText = (int(w / 2 - 800), h - 100)
    fontScale = 8
    fontColor = (255, 255, 255)
    thickness = 10
    lineType = cv2.LINE_8
    # Text border
    cv2.putText(
        image,
        over_str,
        bottomLeftCornerOfText,
        font,
        fontScale,
        (0,),
        thickness + 10,
        lineType,
    )
    # Inner text
    cv2.putText(
        image,
        over_str,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )

    if logo_path:
        overlay = cv2.imread(logo_path)
        image = overlay_logo(image, overlay, pading=50, alpha=1)
    # cv2.imwrite("out.jpg", image)

    return image


def overlay_logo(
    img: np.ndarray,
    img_overlay: np.ndarray,
    pading: int = 0,
    alpha: float = 1,
):
    h, w = img_overlay.shape[:2]
    shapes = np.zeros_like(img, np.uint8)
    shapes[
        img.shape[0] - h - pading : -pading, img.shape[1] - w - pading : -pading
    ] = img_overlay
    mask = shapes.astype(bool)
    img[mask] = cv2.addWeighted(img, 1 - alpha, shapes, alpha, 0)[mask]

    # cv2.imwrite("out.jpg", img)
    # cv2.namedWindow("a", cv2.WINDOW_NORMAL)
    # cv2.imshow('a', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img


def resize_all_images(
    image_dir: Union[str, Path],
    size: Tuple,
    im_ext: str = None,
    out_path: Union[str, Path] = None,
    out_fmt: str = "JPEG",
    jpg_quality: int = 95,
) -> Path:

    image_dir = Path(image_dir)
    if im_ext:
        im_list = sorted(image_dir.glob("*." + im_ext))
    else:
        im_list = sorted(image_dir.glob("*"))

    # Size casted to int
    size = tuple([int(x) for x in size])

    im = Image.open(im_list[0])
    print(f"Resizing images from {im.size[0]}x{im.size[1]} to {size[0]}x{size[1]}: ")

    if out_path is None:
        out_path = im_list[0].parent / "resized"
    else:
        out_path = Path(out_path)
    if out_path.is_dir():
        print("Output directory already exists. Will write into it.")
    else:
        out_path.mkdir(parents=True)

    for file in tqdm(im_list):
        if not file.is_file():
            continue
        # print('*', end='')
        im = Image.open(file)
        imResize = im.resize(size, Image.Resampling.LANCZOS)
        imResize.save(
            out_path / file.name, out_fmt, quality=jpg_quality
        )  # setting quality

    return out_path


if __name__ == "__main__":

    from easydict import EasyDict as edict

    parser = argparse.ArgumentParser(
        prog="CreateTimeLapse",
        description="Create Time Lapse video from series of images",
        epilog="\n",
    )

    parser.add_argument(
        "-i", "--image_dir", type=str, help="Path to directory containing the images"
    )
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        default="out",
        help="Name of the output video (without extension).",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default=None,
        help="Extension of the images to be included in the video. If None is provided, no checks on the type of files in IMAGE_DIR are carried out. Default=None",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Video Frame Rate. Default=30 fps"
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="avc1",
        help="Codec string according to OpenCV. Default='avc1'",
    )
    parser.add_argument(
        "--resize_fct",
        type=float,
        default=1.0,
        help="Factor for resizing the images (2 means downsampling the images by a factor 2). Default=1",
    )
    parser.add_argument(
        "--logo",
        type=str,
        default=None,
        help="Path to logo image (with ext) to add at the bottom of the video. Default=None",
    )
    args = parser.parse_args()

    print("-------------------------------------")
    print("Creating time-lapse video from images")
    print("-------------------------------------")

    # For testing purposes
    # args = edict(
    #     {
    #         "image_dir": "test",
    #         "image_ext": "JPG",
    #         "output_name": "test",
    #         "fps": 30.0,
    #         "resize_fct": 4,
    #         "logo": "data/logo_polimi.jpg",
    #         "codec": "avc1",
    #     }
    # )

    # Check Paths
    args.image_dir = Path(args.image_dir)
    assert (
        args.image_dir.is_dir()
    ), "Image directory does not exist. Provide a valid path."
    if args.logo:
        assert Path(
            args.logo
        ).is_file(), "Logo image does not exist. Provide a valid path."

    video_name = f"{args.output_name}_{int(args.fps)}fps.mp4"
    num_of_images = len(os.listdir(args.image_dir))
    print(f"Images found: {num_of_images}")

    fourcc = cv2.VideoWriter_fourcc(
        args.codec[0],
        args.codec[1],
        args.codec[2],
        args.codec[3],
    )
    generate_video(
        args.image_dir,
        video_name=video_name,
        resize_fct=args.resize_fct,
        im_ext=args.image_ext,
        fourcc=fourcc,
        fps=args.fps,
        logo_path=args.logo,
    )

    print("Done")
