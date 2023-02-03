from utils.generate import Single_Anchor, Anchors
import numpy as np
base_size = np.array(8)
ratios = np.array([0.25, 1, 4])
scales = np.array([4, 6, 8])

anchors = Single_Anchor(base_size, scales, ratios)
input_shape = np.array([2, 3])
final_anchors = Anchors(input_shape, base_size, scales, ratios)
print(final_anchors)