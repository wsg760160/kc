import cv2
from manager import WindowManager, CaptureManager

num_image = 1
num_video = 1


class Cameo(object):

    def __init__(self):
        self._windowManger = WindowManager("cameo", self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0, cv2.CAP_DSHOW), self._windowManger, True)

    def onKeypress(self, keycode):
        """Handle a keypress.
        space -> Take a screenshot.
        tab -> Start /stop recording a screencast.
        escape -> Quit
        enter  -> Mode changes(have edges or not: 0, have edge;1,have no edge)
        """
        global num_image, num_video
        if keycode == 32:  # space
            self._captureManager.writeImage(f'screenshot{num_image}.png')
            num_image += 1
        elif keycode == 9:  # tab
            num = (num_video + 1) // 2
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo(f'screencast{num}.avi')
            if num_video % 2 == 0:
                self._captureManager.stopWritingVideo()
            num_video += 1
        elif keycode == 13:  # enter
            self._captureManager.modeChange()
        elif keycode == 37:  # 精度减小， 漏检率减小
            self._captureManager.dwNms_threshold()
        elif keycode == 38:  # nms上升，候选框增多
            self._captureManager.upScope_threshold()
        elif keycode == 39:  # 精度上升， 漏检率上升
            self._captureManager.upNms_threshold()
        elif keycode == 40:  # nms降低， 候选框减少
            self._captureManager.dwScope_threshold()
        elif keycode == 16:  # 重置detectMultiScale参数
            self._captureManager.reset()
        elif keycode == 27:  # ESC
            self._windowManger.destroyWindow()
        else:
            pass

    def run(self):
        """Run the main loop"""
        self._windowManger.createWindow()
        while self._windowManger.isWindowCreated:
            self._captureManager.enterFrame()
            self._captureManager.exitFrame()
            self._windowManger.processEvents()


if __name__ == '__main__':
    Cameo().run()