import cv2
import numpy as np
import time


def celestial_detect(img_processed, scores_threshold, nms_threshold):
    weightsPath = ""  # 识别的物体权重文件
    configPath = ""  # 模型配置文件
    labelsPath = ""  # 模型特征文件
    labels = open(labelsPath).read().strip().split("\n")  # 格式化标签参数
    boxes = []
    confidences = []
    classIDs = []
    net = cv2.dnn.readNetFromCaffe(configPath, weightsPath)  # 构建识别神经网络（输入模型配置及权值）
    img = cv2.imread("a")  # 测试用图片
    (H, W) = img.shape[:2]  # 得到图片大小（高，宽）
    In = net.getLayerNames()  # 得到各层的名称
    out = net.getUnconnectedOutLayers()  # 得到未连接层的序号（前一个既是需要的输出层的序号）
    x = []
    for i in out:
        x.append(In[i[0] - 1])
    # 对In（输入层）的进行筛选，得到yolo需要的输出层
    In = x
    # 利用blob算法提取特征，对应训练的模型
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (100, 100), swapRB=True, crop=False)  # 对img进行归一化操作，便于后期提取特征
    net.setInput(blob)  # 将blob设置为输入值（进行检测）
    layerOutputs = net.forward(In)  # 将前一层输出转化为下一层输入，直至达到输出层（原理看不太懂）
    # 遍历多个输出层
    for output in layerOutputs:
        # 每个输出层有多个检测框
        for detection in output:
            scores = detection[5:]  # 取最后两位
            classID = np.argmax(scores)  # 反馈最大值的索引（class1，class2）
            confidence = scores[classID]  # 置信度高的
            if confidence > 0.5:  # 排除置信度低的检测框
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # 起始点
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # 检测框坐标, 导入检测框参数值
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, scores_threshold, nms_threshold)
    box_seq = idxs.flatten()  # 将数据一维化，卷积层到全连接层的过渡
    if len(idxs) > 0:
        for seq in box_seq:
            (x, y) = (boxes[seq][0], boxes[seq][1])  # 起始点
            (w, h) = (boxes[seq][2], boxes[seq][3])  # 框的大小
            if classIDs[seq] == 0:
                color = [0, 0, 255]
            else:
                color = [255, 0, 0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[classIDs[seq]]} {confidences[classIDs[seq]]}"
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.3, color, 1)

    return img_processed


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
        self._num1 = 0.2
        self._num2 = 0.3

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

    def upScope_threshold(self):
        if self._num1 < 1:
            self._num1 += 0.1
            print(f"Scope_threshold is {self._num1}")
        else:
            print("Scope_threshold is out of range.")

    def dwScope_threshold(self):
        if self._num1 > 0:
            self._num1 -= 0.1
            print(f"Scope_threshold is {self._num1}")
        else:
            print("Scope_threshold is out of range.")

    def dwNms_threshold(self):
        if self._num2 > 0:
            self._num2 -= 0.1
            print(f"Nms_threshold is {self._num2}")
        else:
            print("Nms_threshold is out of range.")

    def upNms_threshold(self):
        if self._num2 < 1:
            self._num2 += 0.1
            print(f"Nms_threshold is {self._num2}")
        else:
            print("Nms_threshold is out of range.")

    def reset(self):
        self._num1 = 0.2
        self._num2 = 0.3

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
                    frame1 = celestial_detect(frame, self._num1, self._num2)
                    self.previewWindowManager.show(frame1)
            else:
                if self._mode:
                    self.previewWindowManager.show(self._frame)
                elif not self._mode:
                    frame1 = self._frame.copy()
                    frame1 = celestial_detect(frame1, self._num1, self._num2)
                    self.previewWindowManager.show(frame1)
        # Write to the image file, if any.
        if self.isWritingImage:  # 判断是否按下space
            # 判断是否需要标识出对应天体
            if self._mode:
                cv2.imwrite(self._imageFilename, self._frame)
            elif not self._mode:
                img = self._frame.copy()
                img = celestial_detect(img, self._num1, self._num2)
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
            img = celestial_detect(img, self._num1, self._num2)
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
