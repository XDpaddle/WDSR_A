from x2paddle import torch2paddle
import paddle
from paddle import nn


# class ShiftMean(nn.Layer):

#     def __init__(self, rgb_mean):
#         super(ShiftMean, self).__init__()
#         self.rgb_mean = paddle.to_tensor(rgb_mean).reshape([1, 3, 1, 1])
    
#     def forward(self, x, mode):
#         if mode == 'sub':
#             return (x - self.rgb_mean.to(x.device) * 255.0) / 127.5
#         elif mode == 'add':
#             return x * 127.5 + self.rgb_mean.to(x.device) * 255.0
#         else:
#             raise NotImplementedError
        
class ShiftMean(nn.Layer):
    def __init__(self, rgb_mean):
        super(ShiftMean, self).__init__()
        self.rgb_mean = paddle.to_tensor(rgb_mean).reshape([1, 3, 1, 1])

    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.rgb_mean.to(x.place) * 255.0) / 127.5
        elif mode == 'add':
            return x * 127.5 + self.rgb_mean.to(x.place) * 255.0
        else:
            raise NotImplementedError 

