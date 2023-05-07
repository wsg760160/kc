import cv2
import numpy as np
import time

satellite_detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")  # 可以改识别模型
text = "satellite"


def satellite_detect(img_processed, scan, distance):
    img_copy = img_processed
    img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
    img_processed = cv2.GaussianBlur(img_processed, (3, 3), 0)
    satellite = satellite_detector.detectMultiScale(img_processed,
                                                    scaleFactor=scan,  # 缩放，越小越细
                                                    minNeighbors=distance,  # 邻距，同上
                                                    minSize=(60, 60))  # 图层最小尺寸
    for x, y, w, h in satellite:
        if x > 10000 or y > 10000 or x < 0 or y < 0:
            continue
        else:
            cv2.rectangle(img_copy, pt1=(int(x), int(y)), pt2=(int(x) + int(w), int(y) + int(h)),
                          color=[71, 81, 101],
                          thickness=2)
            cv2.putText(img_copy, text, (int(x + 2 * w / 3), int(y + h + 50)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                        (0, 0, 123))
    return img_copy


class CaptureManager(object):
    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):  # 对该类添加属性
        self.previewWindowManager = previewWindowManager  # 预览窗口
        self.shouldMirrorPreview = shouldMirrorPreview  # 镜像窗口
        self._capture = capture  # 打开摄像头
        self._channel = 0  # 摄像头通道，
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        self._startTime = None
        self._framesElapsed = 0
        self._fpsEstimate = None
        self._mode = True
        self._num1 = 1.2
        self._num2 = 3

    @property  # 定义只读属性channel（0）
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):  # 为channel赋值
        if self._channel != value:  # 判断摄像头通道是否正确（0）
            self._channel = value
            self._frame = None

    @property
    def frame(self):  # 定义只读属性frame
        if self._enteredFrame and self._frame is None:  # 添加帧
            # _ 为变量名（我们不需要），retrieve返回两个值，该函数只需要第二个；用于忽略该值
            _, self._frame = self._capture.retrieve(self._frame, self._channel)  # 返回VideoCapture.grab 捕获的帧
        return self._frame  # 将VideoCapture.grab 捕获的帧添加至对象属性self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None  # 返回拍摄图片保存地址不为无

    @property
    def isWritingVideo(self):  # 返回拍摄视频保存地址不为无
        return self._videoFilename is not None

    def modeChange(self):
        if self._mode:
            self._mode = False
            print("开始扫描")
        elif not self._mode:
            self._mode = True
            print("关闭扫描")

    def upSFactor(self):
        if self._num1 < 5:
            self._num1 += 0.03
            print(f"ScaleFactor is {self._num1}")
        else:
            print("ScaleFactor is out of range.")

    def dwSFactor(self):
        if self._num1 > 1:
            self._num1 -= 0.03
            print(f"ScaleFactor is {self._num1}")
        else:
            print("ScaleFactor is out of range.")

    def dwMinNeighbor(self):
        if self._num2 > 0:
            self._num2 -= 1
            print(f"MinNeighbor is {self._num2}")
        else:
            print("MinNeighbor is out of range.")

    def upMinNeighbor(self):
        if self._num2 < 7:
            self._num2 += 1
            print(f"MinNeighbor is {self._num2}")
        else:
            print("MinNeighbor is out of range.")

    def resetDetectScale(self):
        self._num1 = 1.2
        self._num2 = 3

    def enterFrame(self):  # enteredFrame的实现值（同步的）抓取一帧，而来自通道的实际检索被推迟到frame变量的后续读取
        """捕获下一帧,如果有的话"""
        # 但首先,检查前一个帧是否被退出
        assert not self._enteredFrame, "precious enterFrame() had no matching exitFrame()"
        if self._channel is not None:
            self._enteredFrame = self._capture.grab()  # 将视频引向下一帧，若成功则返回True

    def exitFrame(self):
        """Draw to the windows. Write to file. Release the frame."""
        # Check whether any grabbed frame is retrievable
        # The getter may retrieve and cache the frame
        if self.frame is None:
            self._enteredFrame = False
            return

        # Update the FPS estimate ad related variables.
        if self._framesElapsed == 0:  # 帧数
            self._startTime = time.time()  # 开始计时
        else:
            timeElapsed = time.time() - self._startTime  # 拍摄时间
            # 帧率实时更新， 越到后面越精确
            self._fpsEstimate = self._framesElapsed / timeElapsed  # 帧率 = 帧数 / 拍摄时间
        self._framesElapsed += 1  # 帧数加一

        # Draw to the window, if any.
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:  # 拍摄图片是镜像
                mirroredFrame = np.fliplr(self._frame)  # 当前帧水平方向反转（既再次镜像化）
                if self._mode:
                    self.previewWindowManager.show(mirroredFrame)
                elif not self._mode:
                    frame = mirroredFrame.copy()
                    frame1 = satellite_detect(frame, self._num1, self._num2)
                    self.previewWindowManager.show(frame1)
            else:
                if self._mode:
                    self.previewWindowManager.show(self._frame)
                elif not self._mode:
                    frame1 = self._frame.copy()
                    frame1 = satellite_detect(frame1, self._num1, self._num2)
                    self.previewWindowManager.show(frame1)
        # Write to the image file, if any.
        if self.isWritingImage:  # 判断是否按下space
            # 判断是否需要标识出对应天体
            if self._mode:
                cv2.imwrite(self._imageFilename, self._frame)
            elif not self._mode:
                img = self._frame.copy()
                img = satellite_detect(img, self._num1, self._num2)
                cv2.imwrite(self._imageFilename, img)
            self._imageFilename = None  # 当前帧只截屏一次

        # Write to the video file, if any.
        self._writeVideoFrame()  # 判断是否按下TAB
        # release the frame
        self._frame = None
        self._enteredFrame = False

    def writeImage(self, filename):
        """Write the next exited frame to an image file"""
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding=cv2.VideoWriter_fourcc("X", "V", "I", "D")):
        """Start writing exited frames to a video file."""
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """Stop writing exited frame to a video file"""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):
        if not self.isWritingVideo:
            return
        if self._videoWriter is None:
            fps = self._capture.get(5)
            if fps <= 0.0:
                # THe capture's FPS is unknown so use an estimate/
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that the estimate is more stable.
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)
        if self._mode:
            self._videoWriter.write(self._frame)
        elif not self._mode:
            img = self._frame.copy()
            img = satellite_detect(img, self._num1, self._num2)
            self._videoWriter.write(img)


class WindowManager(object):
    def __init__(self, windowName, keypressCall=None):
        self.keypressCallback = keypressCall
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            self.keypressCallback(keycode)