import cv2 as cv
import numpy as np

WINDOW_NAME = "Test"
MARKER_NUMB = 11
MARKER_QUANTITY = MARKER_NUMB * MARKER_NUMB
MARKER_SIZE = 3
MARKER_COLOR = (0, 0, 255)


def imgScale(src, scale):
    width = int(src.shape[1] * scale)
    height = int(src.shape[0] * scale)
    return cv.resize(src, (width, height), interpolation=cv.INTER_AREA)


def getPoints():
    p = []
    startW = img.shape[1] * 0.1
    startH = img.shape[0] * 0.1
    w = img.shape[1] / (MARKER_NUMB - 1)
    h = img.shape[0] / (MARKER_NUMB - 1)
    for i in range(0, MARKER_NUMB):
        for j in range(0, MARKER_NUMB):
            p.append((int(startW + w * i), int(startH + h * j)))
    return p


def getSquares():
    squares = []

    for i in range(0, MARKER_NUMB - 1):
        for j in range(0, MARKER_NUMB - 1):
            squares.append([i * MARKER_NUMB + j, i * MARKER_NUMB + j + 1, (i + 1) * MARKER_NUMB + j,
                            (i + 1) * MARKER_NUMB + j + 1])

    return squares


def drawPoints(src, p):
    for i in range(0, len(p)):
        cv.circle(img=src, center=p[i], radius=MARKER_SIZE, color=MARKER_COLOR, thickness=cv.FILLED,
                  lineType=cv.LINE_AA)


def drawSquares(src, p, s):
    for i in range(0, len(s)):
        cv.line(src, p[s[i][0]], p[s[i][1]], (255, 255, 0), 1, cv.LINE_AA)
        cv.line(src, p[s[i][1]], p[s[i][3]], (255, 255, 0), 1, cv.LINE_AA)
        cv.line(src, p[s[i][3]], p[s[i][2]], (255, 255, 0), 1, cv.LINE_AA)
        cv.line(src, p[s[i][2]], p[s[i][0]], (255, 255, 0), 1, cv.LINE_AA)


def affineTransform(src):
    result = []
    center = int(MARKER_NUMB / 2)
    for i in range(0, MARKER_NUMB):
        for j in range(0, MARKER_NUMB):
            iDist = (i - center)
            jDist = (j - center)

            iCoef = jCoef = abs(iDist) ** 2 + abs(jDist) ** 2\

            if iDist < 0:
                iCoef *= -1

            if iDist == 0:
                iCoef = 0

            if jDist < 0:
                jCoef *= -1

            if jDist == 0:
                jCoef = 0

            # print(iCoef, sep=' ', end=' ', flush=True)
            result.append((src[i * MARKER_NUMB + j][0] + iCoef, src[i * MARKER_NUMB + j][1] + jCoef))

        # print(sep="\n")
    return result


def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv.INTER_LINEAR,
                        borderMode=cv.BORDER_REFLECT_101)

    return dst


def distortion(src, res, pSrc, pRes, s):
    for i in range(0, len(s)):
        distortionTriangle(src, res, (pSrc[s[i][0]], pSrc[s[i][1]], pSrc[s[i][2]]), (pRes[s[i][0]], pRes[s[i][1]], pRes[s[i][2]]))
        distortionTriangle(src, res, (pSrc[s[i][1]], pSrc[s[i][2]], pSrc[s[i][3]]), (pRes[s[i][1]], pRes[s[i][2]], pRes[s[i][3]]))


def distortionTriangle(src, res, t1, t):
    r1 = cv.boundingRect(np.float32([t1]))
    r = cv.boundingRect(np.float32([t]))

    t1Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    img1Rect = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r[2], r[3])
    warpImage = applyAffineTransform(img1Rect, t1Rect, tRect, size)

    res[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = res[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + warpImage * mask


if __name__ == "__main__":
    img = cv.imread("chess.png")
    img = imgScale(img, 0.5)
    scaledImg = np.zeros((int(img.shape[0] * 1.2), int(img.shape[1] * 1.2), 3), dtype=np.uint8)
    scaledImg[int(img.shape[0] * 0.1):scaledImg.shape[0] - int(img.shape[0] * 0.1),
            int(img.shape[1] * 0.1):scaledImg.shape[1] - int(img.shape[1] * 0.1)] = img

    points = getPoints()
    sq = getSquares()
    resPoints = affineTransform(points)

    resImg = np.zeros(scaledImg.shape, dtype=scaledImg.dtype)
    distortion(scaledImg, resImg, points, resPoints, sq)

    drawSquares(scaledImg, resPoints, sq)
    cv.imshow("scaled img", scaledImg)
    cv.imshow("result", resImg)
    cv.waitKey(0)
    cv.destroyAllWindows()
