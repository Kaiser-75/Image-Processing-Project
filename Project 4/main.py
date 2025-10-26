import cv2
import time
import tkinter as tk
from tkinter import filedialog
from vision import CardDetector, CardMatcher

class HUD:
    INSTR = "Press D to detect. Q to quit."

    def draw(self, frame, msg, show_instr=True):
        vis = frame.copy()
        if show_instr:
            cv2.rectangle(vis, (10, 10), (600, 80), (0, 0, 0), -1)
            cv2.putText(vis, self.INSTR, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230,230,230), 2)
        if msg:
            cv2.putText(vis, msg, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        return vis

class CardRecognizer:
    def __init__(self):
        self.detector = CardDetector()
        self.matcher = CardMatcher()
        self.hud = HUD()

    def detect(self, frame):
        crops = self.detector.from_frame(frame)
        if not crops.ok:
            return False, "Cannot detect", crops
        res = self.matcher.match(crops.rank_img, crops.suit_img)
        if not res.ok:
            return False, "Cannot detect", crops
        return True, f"{res.rank} of {res.suit}", crops

class WebcamApp:
    def __init__(self):
        self.recognizer = CardRecognizer()

    def _open_camera(self):
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(0, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            for _ in range(5):
                cap.read()
            return cap
        return None

    def run(self):
        cap = self._open_camera()
        if not cap or not cap.isOpened():
            print("Cannot open webcam")
            return
        cv2.namedWindow("Card Match", cv2.WINDOW_NORMAL)
        label, freeze_until = "", 0
        print("Webcam ready. Hold card fully visible, bright light, dark background.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()
            msg = label if now < freeze_until else ""
            cv2.imshow("Card Match", self.recognizer.hud.draw(frame, msg))
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('d'):
                ok, label, crops = self.recognizer.detect(frame)
                label = label if ok else "Cannot detect"
                freeze_until = time.time() + 2
                if crops.warp_gray is not None:
                    side = cv2.cvtColor(crops.warp_gray, cv2.COLOR_GRAY2BGR)
                    new_w, new_h = int(200 * 1.5), int(300 * 1.5)
                    side = cv2.resize(side, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    combo = self.recognizer.hud.draw(side, label, show_instr=False)
                    cv2.imshow("Result", combo)
        cap.release()
        cv2.destroyAllWindows()

class ImageApp:
    def __init__(self):
        self.recognizer = CardRecognizer()

    def run(self):
        path = filedialog.askopenfilename(title="Select image")
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            print("Cannot read image file.")
            return
        ok, label, crops = self.recognizer.detect(img)
        label = label if ok else "Cannot detect"
        if crops.warp_gray is not None:
            side = cv2.cvtColor(crops.warp_gray, cv2.COLOR_GRAY2BGR)
            new_w, new_h = int(200 * 1.5), int(300 * 1.5)
            side = cv2.resize(side, (new_w, new_h), interpolation=cv2.INTER_AREA)
            combo = self.recognizer.hud.draw(side, label, show_instr=False)
        else:
            preview = cv2.resize(img, (300, 200), interpolation=cv2.INTER_AREA)
            combo = self.recognizer.hud.draw(preview, label, show_instr=False)
        cv2.imshow("Result", combo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class CardApp:
    def __init__(self):
        self.win = tk.Tk()
        self.win.title("Card Matcher")
        tk.Button(self.win, text="Webcam", width=20,
                  command=lambda: [self.win.withdraw(), WebcamApp().run(), self.win.deiconify()]).pack(padx=10, pady=10)
        tk.Button(self.win, text="Choose Image", width=20,
                  command=lambda: [self.win.withdraw(), ImageApp().run(), self.win.deiconify()]).pack(padx=10, pady=10)

    def run(self):
        self.win.mainloop()

if __name__ == "__main__":
    CardApp().run()

