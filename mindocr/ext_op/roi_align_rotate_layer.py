import math
from mindspore.nn import Cell
import mindspore.ops as ops
from .bilinear_interp import bilinear_interpolate, bilinear_interpolate_gradient


class RoiAlignRotatedLayer(Cell):
    def __init__(self,
                 pooled_height,
                 pooled_width,
                 spatial_scale,
                 sampling_ratio=0,
                 aligned=True,
                 clockwise=False):
        super(RoiAlignRotatedLayer, self).__init__()
        self.spatial_scale = spatial_scale
        self.aligned = aligned
        self.clockwise = clockwise
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.sampling_ratio = sampling_ratio
        self.rois = None

    def construct(self, input, rois):
        # input: 输入特征图，shape为 (N, C, H, W)
        # rois: ROI框，shape为 (R, 6)，每行格式为 [batch_ind, xc, yc, w, h, theta]
        # pooled_height: 池化后的高度
        # pooled_width: 池化后的宽度
        # spatial_scale: ROI的缩放比例
        _, channels, height, width = input.shape
        n_rois = rois.shape[0]
        offset = 0.5 if self.aligned else 0
        output = ops.zeros((n_rois, channels, self.pooled_height, self.pooled_width), dtype=input.dtype)

        for idx, curr_rois in enumerate(rois):
            batch_ind, xc, yc, roi_width, roi_height, theta = curr_rois
            batch_ind = int(batch_ind)  # TODO: confirm this or use idx
            # 将ROI坐标缩放到输入特征图尺寸
            xc = xc * self.spatial_scale - offset
            yc = yc * self.spatial_scale - offset
            roi_width *= self.spatial_scale
            roi_height *= self.spatial_scale

            if self.clockwise:  # if clockwise, the angle needs to be reversed
                theta = -theta
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            if self.aligned:
                assert roi_width >= 0 and roi_height >= 0, "roi_width < 0 or roi_height < 0"
            else:
                roi_width = max(roi_width, 1.0)
                roi_height = max(roi_height, 1.0)

            # 计算ROI在池化后特征图中的尺寸和每个池化格子的大小
            bin_size_h = roi_height / self.pooled_height
            bin_size_w = roi_width / self.pooled_width

            # We use roi_bin_grid to sample the grid and mimic integral
            roi_bin_grid_h = self.sampling_ratio if self.sampling_ratio > 0 else math.ceil(
                roi_height / self.pooled_height)
            roi_bin_grid_w = self.sampling_ratio if self.sampling_ratio > 0 else math.ceil(
                roi_width / self.pooled_width)

            # do average pooling inside a bin
            count = max(roi_bin_grid_h * roi_bin_grid_w, 1)

            x1 = -roi_width / 2.0
            y1 = -roi_height / 2.0
            # 遍历池化后特征图中的每个格子
            for ph in range(self.pooled_height):
                for pw in range(self.pooled_width):
                    # 采样点计算
                    curr_sum = ops.zeros(channels, dtype=input.dtype)
                    for iy in range(roi_bin_grid_h):
                        # 计算格子在ROI坐标系下的中心点坐标
                        yy = y1 + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                        for ix in range(roi_bin_grid_w):
                            xx = x1 + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w

                            # 将中心点坐标转换到输入特征图坐标系下
                            y = yy * cos_theta - xx * sin_theta + yc
                            x = yy * sin_theta + xx * cos_theta + xc

                            curr_sum += bilinear_interpolate(input[batch_ind], y, x)

                    # do average pooling
                    output[idx, :, ph, pw] = curr_sum / count
        self.rois = rois
        return output

    def bprop(self, input, out, grad_output):
        offset = 0.5 if self.aligned else 0.0
        grad_input = ops.zeros_like(input)

        for idx, curr_rois in enumerate(self.rois):
            batch_ind, xc, yc, roi_width, roi_height, theta = curr_rois
            batch_ind = int(batch_ind)  # TODO: confirm this or use idx
            # 将ROI坐标缩放到输入特征图尺寸
            xc = xc * self.spatial_scale - offset
            yc = yc * self.spatial_scale - offset
            roi_width *= self.spatial_scale
            roi_height *= self.spatial_scale

            theta = -theta if self.clockwise else theta
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            if self.aligned:
                assert roi_width >= 0 and roi_height >= 0, "roi_width < 0 or roi_height < 0"
            else:
                roi_width = max(roi_width, 1.0)
                roi_height = max(roi_height, 1.0)

            # 计算ROI在输入特征图中的四个角点坐标
            x1 = -roi_width / 2.0
            y1 = -roi_height / 2.0

            # 计算ROI在池化后特征图中的尺寸和每个池化格子的大小
            bin_size_h = roi_height / self.pooled_height
            bin_size_w = roi_width / self.pooled_width

            # We use roi_bin_grid to sample the grid and mimic integral
            roi_bin_grid_h = self.sampling_ratio if self.sampling_ratio > 0 else math.ceil(
                roi_height / self.pooled_height)
            roi_bin_grid_w = self.sampling_ratio if self.sampling_ratio > 0 else math.ceil(
                roi_width / self.pooled_width)

            # do average pooling inside a bin
            count = max(roi_bin_grid_h * roi_bin_grid_w, 1)

            # 遍历池化后特征图中的每个格子
            for ph in range(self.pooled_height):
                for pw in range(self.pooled_width):
                    # 采样点计算
                    curr_sum = ops.zeros_like(input[batch_ind])
                    for iy in range(roi_bin_grid_h):
                        # 计算格子在ROI坐标系下的中心点坐标
                        yy = y1 + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                        for ix in range(roi_bin_grid_w):
                            xx = x1 + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w

                            # 将中心点坐标转换到输入特征图坐标系下
                            y = yy * cos_theta - xx * sin_theta + yc
                            x = yy * sin_theta + xx * cos_theta + xc

                            curr_sum += bilinear_interpolate_gradient(grad_output[idx, :, ph, pw],
                                                                      input[batch_ind],
                                                                      y,
                                                                      x)

                    # 计算反向传播梯度
                    grad_input[batch_ind] += curr_sum / count

        return grad_input


if __name__ == "__main__":
    from mindspore import Tensor
    import numpy as np
    import mindspore as ms

    ms.set_context(device_id=6)
    roi_pooling = RoiAlignRotatedLayer(2, 2, 0.5, sampling_ratio=4)
    # rois: [batch_ind, xc, yc, w, h, theta]
    rois = Tensor(np.array([[0, 320, 320, 90, 90, 0.785]]), ms.float32)
    features = Tensor(np.random.randn(1, 3, 640, 640), ms.float32)
    roi_align_rotated = roi_pooling(features, rois)
    grad_output = Tensor(np.random.randn(1, 3, 640, 640), ms.float32)
    grad = roi_pooling.bprop(features, roi_align_rotated, grad_output)
    print("roi_align_rotated result: ", roi_align_rotated)
    print("grad = ", grad)
