import numpy as np

def Single_Anchor(base_size, scales, ratios, center_offset=0.):
   """
   在(0,0)坐标点根据scales,ratios
   生成anchors
   base_size: base size of anchors
   scales: numpy(n)
   ratios: numpy(m)
   首先计算出每一个anchor的高度和宽度;
   然后计算出每一个anchor的[-0.5w, -0.5h, 0.5w, 0.5h]

   """
   anchor_num = len(scales) * len(ratios)
   anchors = np.empty(shape=(anchor_num, 4))
   w = base_size
   h = base_size
   x_center = w * center_offset
   y_center = h * center_offset
   h_ratios = np.sqrt(ratios)
   w_ratios = 1./h_ratios
   anchor_h = (h*scales[None, :]*h_ratios[:, None])[:]
   anchor_w = (w*scales[None, :]*w_ratios[:, None])[:]
   anchors = np.stack([x_center-0.5*anchor_w, y_center-0.5*anchor_h, x_center + 0.5*anchor_w, y_center+0.5*anchor_h], axis=-1).reshape(-1, 4)
   return anchors



def Anchors(input_shape, base_size, scales, ratios):
   """
   input_shape: (W, H)
   base_size:
   scales:
   ratios:
   """
   W = input_shape[0]
   H = input_shape[1]
   W_range = np.array(range(W))*base_size
   H_range = np.array(range(H))*base_size
   anchors = Single_Anchor(base_size, scales, ratios)
   xx, yy = np.meshgrid(W_range, H_range)
   xx = xx.reshape(-1)
   yy = yy.reshape(-1)
   xyxy = np.stack([xx, yy, xx, yy], axis=-1)
   return (xyxy[:, None, :] + anchors[None, :, :]).reshape(-1, 4)


def Assign():
   """
   """
   

