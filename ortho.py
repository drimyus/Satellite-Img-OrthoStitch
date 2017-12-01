import numpy as np
import os
import cv2

print(cv2.__version__)


class Match:
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
        search_params = dict(checks=5)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.warp_size = None

    def _get_SURF_features(self, gray):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}

    def match(self, path1, path2):

        img1 = cv2.imread(path1, 0)
        img2 = cv2.imread(path2, 0)
        print(str(path_1) + ":" + str(path_2))

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        print(str(h1) + "," + str(w1))
        print(str(h2) + "," + str(w2))

        feature1 = self._get_SURF_features(img1)
        feature2 = self._get_SURF_features(img2)

        matches = self.flann.knnMatch(feature1['des'], feature2['des'], k=2)
        kps1 = feature1['kp']
        kps2 = feature2['kp']

        goods = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.6 * n.distance and np.fabs(kps1[m.queryIdx].pt[1] - kps2[m.trainIdx].pt[1]) < 0.1 * h:
                goods.append((m.trainIdx, m.queryIdx))

        print(len(goods))

        if len(goods) > 10:
            matched_kps1 = np.float32([kps1[i].pt for (_, i) in goods])
            matched_kps2 = np.float32([kps2[i].pt for (i, _) in goods])

            H, _ = cv2.findHomography(matched_kps1, matched_kps2, cv2.RANSAC, 5.0)

            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            new_corners2 = cv2.perspectiveTransform(corners2, H)

            pts = np.concatenate((corners1, new_corners2), axis=0)
            [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

            t = [-xmin, -ymin]
            newH = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

            im = cv2.warpPerspective(img2, newH.dot(H), (xmax - xmin, ymax - ymin))

            im[t[1]:h1 + t[1], t[0]:w1 + t[0]] = img1

            return im
        else:
            return None


if __name__ == '__main__':
    folder = "data/photos/Maks_P"

    fname1 = "1H8A4744_geotag.JPG"
    fname2 = "1H8A4745_geotag.JPG"

    path_1 = os.path.join(folder, fname1)
    path_2 = os.path.join(folder, fname2)

    res_img = Match().match(path_1, path_2)

    result_path = 'ortho_rectified.jpeg'
    cv2.imwrite(result_path, img)

    print("success!")