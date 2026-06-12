import cv2
import json
import logging
import shutil
import subprocess
from pathlib import Path

from vest_detection.detector import VestDetector
from vest_detection.visualizer import DetectionVisualizer

logger = logging.getLogger(__name__)


class VideoPipeline:
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        enable_tracking: bool = True,
        class_filter: list | None = None,
        display_names: dict | None = None,
    ):
        self.detector = VestDetector(
            model_path=model_path,
            confidence=confidence,
            class_filter=class_filter,
        )
        self.visualizer = DetectionVisualizer(display_names=display_names)
        self.enable_tracking = enable_tracking

    def run(self, video_path: str, output_path: str, json_path: str = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"视频读取失败：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_process = self._open_ffmpeg_writer(output_file, fps, width, height)
        out = None
        if ffmpeg_process is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
            if not out.isOpened():
                cap.release()
                raise RuntimeError(f"视频写入器初始化失败：{output_file}")

        all_results = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.predict(frame, verbose=False)
                annotated = self.visualizer.draw(frame, detections)
                self._write_frame(annotated, ffmpeg_process, out)

                result_data = self._detections_to_dict(detections, frame_idx)
                all_results.append(result_data)

                frame_idx += 1
                if frame_idx % 30 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
        finally:
            cap.release()
            if out is not None:
                out.release()
            self._close_ffmpeg_writer(ffmpeg_process)

        if json_path:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Video saved to {output_path}")
        return all_results

    def _open_ffmpeg_writer(self, output_file: Path, fps: float, width: int, height: int):
        if output_file.suffix.lower() != ".mp4":
            return None

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logger.warning("ffmpeg not found, falling back to OpenCV mp4v output")
            return None

        cmd = [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps:.6f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_file),
        ]
        logger.info("Writing H.264 video with ffmpeg/libx264")
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _write_frame(self, frame, ffmpeg_process, opencv_writer) -> None:
        if ffmpeg_process is not None:
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                stderr = ffmpeg_process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg H.264 写入失败：{stderr}") from exc
            return

        opencv_writer.write(frame)

    def _close_ffmpeg_writer(self, ffmpeg_process) -> None:
        if ffmpeg_process is None:
            return

        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        stderr = ffmpeg_process.stderr.read().decode("utf-8", errors="replace")
        return_code = ffmpeg_process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg H.264 编码失败：{stderr}")

    def _detections_to_dict(self, detections, frame_idx: int):
        class_names = detections.data.get("class_name", [])

        results = []
        for i in range(len(detections)):
            results.append({
                "class_name": str(class_names[i]) if len(class_names) > i else str(detections.class_id[i]),
                "confidence": float(detections.confidence[i]),
                "bbox": detections.xyxy[i].tolist()
            })

        return {
            "frame": frame_idx,
            "count": len(results),
            "detections": results
        }
