import unittest
import cv2
from niqe import niqe


class NIQE(unittest.TestCase):

    def test(self):

        img3 = cv2.imread("hdr.jpg")

        n3 = niqe(img3)

        print(n3)

        # self.assertGreater(n1, n2, "clean image NIQE is higher than degraded")

if __name__ == '__main__':
    unittest.main()
