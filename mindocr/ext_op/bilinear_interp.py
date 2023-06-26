import mindspore.ops as ops


def bilinear_interpolate(feature_map, y, x):
    channels, height, width = feature_map.shape

    if y < -1.0 or y > height or x < -1.0 or x > width:
        return ops.zeros(channels, dtype=feature_map.dtype)

    y = max(y, 0)
    x = max(x, 0)

    # 找出最接近(x,y)的四个整数坐标点
    y_low = int(y)
    x_low = int(x)
    y_high = y_low + 1 if y_low < height - 1 else y_low
    x_high = x_low + 1 if x_low < width - 1 else x_low

    wyh = y - y_low
    wxh = x - x_low
    wyl = 1 - wyh
    wxl = 1 - wxh

    value1 = feature_map[:, y_low, x_low] * wyl * wxl
    value2 = feature_map[:, y_low, x_high] * wyl * wxh
    value3 = feature_map[:, y_high, x_low] * wyh * wxl
    value4 = feature_map[:, y_high, x_high] * wyh * wxh

    return value1 + value2 + value3 + value4


def bilinear_interpolate_gradient(grad_output, feature_map, y, x):
    C, H, W = feature_map.shape

    if y < -1.0 or y > H or x < -1.0 or x > W:
        return ops.zeros_like(feature_map)

    y = max(y, 0)
    x = max(x, 0)

    y_low = int(y)
    x_low = int(x)
    y_high = y_low + 1 if y_low < H - 1 else y_low
    x_high = x_low + 1 if x_low < W - 1 else x_low

    wyh = y - y_low
    wxh = x - x_low
    wyl = 1 - wyh
    wxl = 1 - wxh

    grad_input = ops.zeros_like(feature_map)

    grad_input[:, y_low, x_low] += grad_output * wyl * wxl
    grad_input[:, y_low, x_high] += grad_output * wyl * wxh
    grad_input[:, y_high, x_low] += grad_output * wyh * wxl
    grad_input[:, y_high, x_high] += grad_output * wyh * wxh

    return grad_input
