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
        elif keycode == 37:  # detectMultiScale minNeighbor 减小 -> 目标被检测次数减小，被检测出的数量增加，精度降低
            self._captureManager.dwMinNeighbor()
        elif keycode == 38:  # detectMultiScale scaleFactor 增加 -> 图片缩放比例减小，迭代次数（可以理解为检测次数）增加，精度上升（计算量急速上升）
            self._captureManager.dwSFactor()
        elif keycode == 39:  # detectMultiScale minNeighbor 增加 -> 目标被检测次数增加，被检测出的数量增加，精度升高
            self._captureManager.upMinNeighbor()
        elif keycode == 40:  # detectMultiScale scaleFactor 降低 -> 图片缩放比例提高，迭代次数减少，精度下降，计算量减少
            self._captureManager.upSFactor()
        elif keycode == 16:  # 重置detectMultiScale参数
            self._captureManager.resetDetectScale()
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
