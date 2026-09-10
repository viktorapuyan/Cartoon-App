import subprocess
import sys
import torch
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
import cv2
import numpy as np

from object_detector import ObjectDetector

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
ARUCO_MARKER_SIZE_MM = 50.0
DETECTION_CONFIDENCE = 0.50
NOT_ALLOWED_CLASS = 'Not allowed'
CLEARANCE_MM = 10.0


def resource_path(filename: str) -> Path:
    """Return the path to a bundled resource in source or PyInstaller mode."""
    if getattr(sys, 'frozen', False):
        bundle_dir = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
    else:
        bundle_dir = Path(__file__).resolve().parent
    return bundle_dir / filename


def cv2_to_photoimage(frame: np.ndarray) -> tk.PhotoImage:
    """Convert an OpenCV BGR frame to a Tkinter PhotoImage."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    ppm_header = f'P6\n{width} {height}\n255\n'.encode('ascii')
    return tk.PhotoImage(data=ppm_header + rgb.tobytes(), format='PPM')


class DualCameraApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Cartoon')
        self.root.geometry('1000x650')
        self.root.minsize(860, 560)
        self.root.configure(bg='#f4f6f8')

        self.cap1 = None
        self.cap2 = None
        self.detector1 = None
        self.detector2 = None
        self.photo1 = None
        self.photo2 = None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.width = None
        self.height = None
        self.length = None
        self.pixels_per_mm_cam1 = None
        self.pixels_per_mm_cam2 = None
        self.not_allowed_active = False

        self._configure_styles()
        self._setup_ui()
        self._init_cameras()
        self._init_detectors()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self._update_frames()

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('App.TFrame', background='#f4f6f8')
        style.configure('Title.TLabel', background='#f4f6f8', foreground='#18212f',
                        font=('Segoe UI', 20, 'bold'))
        style.configure('CameraTitle.TLabel', background='#ffffff', foreground='#263445',
                        font=('Segoe UI', 13, 'bold'))
        style.configure('Status.TLabel', background='#f4f6f8', foreground='#64748b',
                        font=('Segoe UI', 10))
        style.configure('Primary.TButton', font=('Segoe UI', 11, 'bold'), padding=(22, 12))
        style.configure('Secondary.TButton', font=('Segoe UI', 11, 'bold'), padding=(22, 12))

    def _setup_ui(self):
        outer = ttk.Frame(self.root, style='App.TFrame', padding=18)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text='Carton Measurement', style='Title.TLabel').pack(pady=(0, 4))
        ttk.Label(outer, text='Position the ArUco markers and carton in both camera views',
                  style='Status.TLabel').pack(pady=(0, 14))

        camera_row = ttk.Frame(outer, style='App.TFrame')
        camera_row.pack(fill=tk.BOTH, expand=True)
        camera_row.columnconfigure(0, weight=1)
        camera_row.columnconfigure(1, weight=1)
        camera_row.rowconfigure(0, weight=1)

        self.camera1_view = self._create_camera_panel(camera_row, 'Camera 1', 'Length', 0)
        self.camera2_view = self._create_camera_panel(camera_row, 'Camera 2', 'Width & Height', 1)

        controls = ttk.Frame(outer, style='App.TFrame')
        controls.pack(pady=(18, 10))

        self.capture_btn = ttk.Button(
            controls, text='Capture Measurements', style='Primary.TButton',
            command=self.capture_measurements
        )
        self.capture_btn.grid(row=0, column=0, padx=8)

        self.generate_btn = ttk.Button(
            controls, text='Generate Dieline', style='Secondary.TButton',
            command=self.generate_dieline, state=tk.DISABLED
        )
        self.generate_btn.grid(row=0, column=1, padx=8)

        self.measurements_label = ttk.Label(
            outer, text='Measurements: not captured', style='Status.TLabel'
        )
        self.measurements_label.pack(pady=(0, 4))

        self.status_label = ttk.Label(
            outer, text='Ready - place ArUco markers in both camera views, then capture',
            style='Status.TLabel'
        )
        self.status_label.pack(pady=(0, 2))

    def _create_camera_panel(self, parent, camera_name, measurement_name, column):
        panel = tk.Frame(parent, bg='#ffffff', highlightthickness=1,
                         highlightbackground='#d7dee7')
        panel.grid(row=0, column=column, sticky='nsew', padx=7)
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text=f'{camera_name} - {measurement_name}',
                  style='CameraTitle.TLabel', anchor=tk.CENTER).grid(
                      row=0, column=0, sticky='ew', pady=(10, 8)
                  )

        view = tk.Label(panel, text=f'{camera_name}\nWaiting for camera...',
                        bg='#202936', fg='#d7dee7', font=('Segoe UI', 12), relief=tk.FLAT)
        view.grid(row=1, column=0, sticky='nsew', padx=10, pady=(0, 10))
        return view

    def _init_cameras(self):
        self.cap1 = self._open_camera(0)
        self.cap2 = self._open_camera(1)

    @staticmethod
    def _open_camera(index):
        camera = cv2.VideoCapture(index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        return camera

    def _init_detectors(self):
        self.detector1 = self._load_detector(resource_path('camera1_segmodel.pt'))
        self.detector2 = self._load_detector(resource_path('camera2_segmodel.pt'))

    @staticmethod
    def _load_detector(model_path):
        try:
            model_path = Path(model_path)
            if not model_path.is_file():
                print(f'Error: Model file not found at {model_path}')
                return None

            detector = ObjectDetector(
                model_path=str(model_path),
                conf_threshold=DETECTION_CONFIDENCE
            )
            return detector if detector.load_model() else None
        except Exception as exc:
            print(f'Error loading {model_path}: {exc}')
            return None

    def _update_frames(self):
        not_allowed_this_frame = self._process_camera_frame(
            self.cap1, self.detector1, self.camera1_view, 'pixels_per_mm_cam1', 1
        )
        not_allowed_this_frame |= self._process_camera_frame(
            self.cap2, self.detector2, self.camera2_view, 'pixels_per_mm_cam2', 2
        )
        self._update_safety_state(not_allowed_this_frame)
        self.root.after(30, self._update_frames)

    def _process_camera_frame(self, camera, detector, view, calibration_attribute, camera_number):
        if camera is None or not camera.isOpened():
            view.configure(text=f'Camera {camera_number}\nUnavailable', image='')
            return False

        received, frame = camera.read()
        if not received:
            view.configure(text=f'Camera {camera_number}\nNo frame received', image='')
            return False

        not_allowed_detected = False
        if detector is not None:
            frame, detections = detector.detect(frame, draw_boxes=True)
            not_allowed_detected = any(
                detection['class_name'] == NOT_ALLOWED_CLASS for detection in detections
            )

        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            marker_corners = corners[0][0]
            top_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
            bottom_width = np.linalg.norm(marker_corners[3] - marker_corners[2])
            marker_width_pixels = (top_width + bottom_width) / 2
            setattr(self, calibration_attribute, marker_width_pixels / ARUCO_MARKER_SIZE_MM)

        photo = cv2_to_photoimage(frame)
        view.configure(image=photo, text='')
        if camera_number == 1:
            self.photo1 = photo
        else:
            self.photo2 = photo
        return not_allowed_detected

    def _update_safety_state(self, not_allowed_this_frame):
        if not_allowed_this_frame != self.not_allowed_active:
            self.not_allowed_active = not_allowed_this_frame
            if not_allowed_this_frame:
                self.capture_btn.configure(state=tk.DISABLED)
                self.generate_btn.configure(state=tk.DISABLED)
                self.status_label.configure(
                    text='WARNING: "Not allowed" object detected - capture disabled',
                    foreground='#c62828'
                )
            else:
                self.capture_btn.configure(state=tk.NORMAL)
                self.generate_btn.configure(
                    state=tk.NORMAL if self._has_measurements() else tk.DISABLED
                )
                self.status_label.configure(
                    text='Ready - place ArUco markers in both camera views, then capture',
                    foreground='#64748b'
                )

    def _has_measurements(self):
        return all(value is not None for value in (self.width, self.height, self.length))

    def capture_measurements(self):
        if self.cap1 is None or not self.cap1.isOpened():
            messagebox.showwarning('Camera unavailable', 'Camera 1 is not available', parent=self.root)
            return
        if self.cap2 is None or not self.cap2.isOpened():
            messagebox.showwarning('Camera unavailable', 'Camera 2 is not available', parent=self.root)
            return

        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        if not ret1 or not ret2:
            messagebox.showwarning('Capture failed', 'Could not capture frames from both cameras.', parent=self.root)
            return
        if self.pixels_per_mm_cam1 is None:
            messagebox.showwarning('Camera 1 not calibrated', 'Place an ArUco marker in Camera 1 view.', parent=self.root)
            return
        if self.pixels_per_mm_cam2 is None:
            messagebox.showwarning('Camera 2 not calibrated', 'Place an ArUco marker in Camera 2 view.', parent=self.root)
            return
        if self.detector1 is None or self.detector2 is None:
            messagebox.showwarning('Detector unavailable', 'One or more object detectors could not be loaded.', parent=self.root)
            return

        _, detections1 = self.detector1.detect(frame1, draw_boxes=False)
        _, detections2 = self.detector2.detect(frame2, draw_boxes=False)
        if not detections1:
            messagebox.showwarning('Object not detected', 'No object detected in Camera 1.', parent=self.root)
            return
        if not detections2:
            messagebox.showwarning('Object not detected', 'No object detected in Camera 2.', parent=self.root)
            return

        x1, _, x2, _ = map(int, detections1[0]['bbox'])
        self.length = (x2 - x1) / self.pixels_per_mm_cam1
        x1, y1, x2, y2 = map(int, detections2[0]['bbox'])
        self.width = (x2 - x1) / self.pixels_per_mm_cam2
        self.height = (y2 - y1) / self.pixels_per_mm_cam2

        self.measurements_label.configure(
            text=f'Length: {self.length:.2f} mm    Width: {self.width:.2f} mm    Height: {self.height:.2f} mm',
            foreground='#176b3a'
        )
        self.status_label.configure(text='Measurements captured successfully', foreground='#176b3a')
        self.generate_btn.configure(state=tk.NORMAL)

    def generate_dieline(self):
        if not self._has_measurements():
            messagebox.showwarning('Missing measurements', 'Please capture measurements first.', parent=self.root)
            return

        try:
            dimensions = {
                'length': float(self.length) + CLEARANCE_MM,
                'width': float(self.width) + CLEARANCE_MM,
                'height': float(self.height) + CLEARANCE_MM,
            }
            if getattr(sys, 'frozen', False):
                generator_path = Path(sys.executable).resolve().parent.parent / 'CartonDieline' / 'CartonDieline.exe'
                args = [str(generator_path)]
            else:
                generator_path = Path(__file__).resolve().parent / 'gen_cartondieline.py'
                args = [sys.executable, str(generator_path)]
            for name, value in dimensions.items():
                args.extend([f'--{name}', str(value)])
            subprocess.Popen(args)
        except Exception as exc:
            messagebox.showerror('Generation failed', f'Failed to generate dieline:\n{exc}', parent=self.root)

    def close(self):
        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    DualCameraApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
