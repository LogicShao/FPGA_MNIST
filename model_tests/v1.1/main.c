#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// 包含模型参数和测试图片
#include "tinylenet_weights.h"
#include "test_image.h"

// 网络结构常量
#define INPUT_H 28
#define INPUT_W 28
#define CONV1_OUT_C 6
#define CONV1_OUT_H 24
#define CONV1_OUT_W 24
#define POOL1_OUT_H 12
#define POOL1_OUT_W 12
#define CONV2_OUT_C 16
#define CONV2_OUT_H 8
#define CONV2_OUT_W 8
#define POOL2_OUT_H 4
#define POOL2_OUT_W 4
#define FC1_IN 256
#define FC1_OUT 32
#define FC2_OUT 10

// Input normalization (match training)
#define INPUT_MEAN 0.1307f
#define INPUT_STD 0.3081f

// 量化推理配置 - 层间缩放策略
#define CONV1_SHIFT 8   // Conv1输出右移位数
#define CONV2_SHIFT 10  // Conv2输出右移位数
#define FC1_SHIFT 8     // FC1输出右移位数

// ReLU 宏定义
#define RELU(x) ((x) > 0 ? (x) : 0)

// 裁剪宏定义 (保持在 int8 范围内 -128 ~ 127)
#define CLAMP_INT8(x) (((x) > 127) ? 127 : (((x) < -128) ? -128 : (int8_t)(x)))

// 最大值宏
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// 2D卷积函数 (5x5卷积核, stride=1, 无padding)
// input: [in_h][in_w], weights: [out_c][in_c][5][5], output: [out_c][out_h][out_w]
void conv2d_5x5(const int8_t *input, int in_h, int in_w, int in_c,
                const int8_t weights[][150], const int8_t *biases, int out_c,
                int8_t *output, int shift) {
    int out_h = in_h - 4;  // 5x5卷积，valid模式
    int out_w = in_w - 4;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int32_t sum = 0;

                // 卷积计算
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < 5; kh++) {
                        for (int kw = 0; kw < 5; kw++) {
                            int in_idx = (ic * in_h * in_w) + ((oh + kh) * in_w) + (ow + kw);
                            int w_idx = ic * 25 + kh * 5 + kw;
                            sum += (int32_t)input[in_idx] * (int32_t)weights[oc][w_idx];
                        }
                    }
                }

                // 加偏置
                sum += (int32_t)biases[oc];

                // ReLU激活
                sum = RELU(sum);

                // 缩放到INT8范围
                sum = sum >> shift;

                // 截断并保存
                int out_idx = oc * out_h * out_w + oh * out_w + ow;
                output[out_idx] = CLAMP_INT8(sum);
            }
        }
    }
}

// 首层卷积（输入通道为1）
void conv2d_5x5_first(const int8_t *input, int in_h, int in_w,
                      const int8_t weights[][25], const int8_t *biases, int out_c,
                      int8_t *output, int shift) {
    int out_h = in_h - 4;  // 5x5卷积，valid模式
    int out_w = in_w - 4;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int32_t sum = 0;

                // 卷积计算
                for (int kh = 0; kh < 5; kh++) {
                    for (int kw = 0; kw < 5; kw++) {
                        int in_idx = (oh + kh) * in_w + (ow + kw);
                        int w_idx = kh * 5 + kw;
                        sum += (int32_t)input[in_idx] * (int32_t)weights[oc][w_idx];
                    }
                }

                // 加偏置
                sum += (int32_t)biases[oc];

                // ReLU激活
                sum = RELU(sum);

                // 缩放到INT8范围
                sum = sum >> shift;

                // 截断并保存
                int out_idx = oc * out_h * out_w + oh * out_w + ow;
                output[out_idx] = CLAMP_INT8(sum);
            }
        }
    }
}

// 2x2最大池化 (stride=2)
void maxpool_2x2(const int8_t *input, int in_h, int in_w, int channels,
                 int8_t *output) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int in_base = c * in_h * in_w;
                int ih = oh * 2;
                int iw = ow * 2;

                int8_t val0 = input[in_base + ih * in_w + iw];
                int8_t val1 = input[in_base + ih * in_w + iw + 1];
                int8_t val2 = input[in_base + (ih + 1) * in_w + iw];
                int8_t val3 = input[in_base + (ih + 1) * in_w + iw + 1];

                int8_t max_val = MAX(MAX(val0, val1), MAX(val2, val3));

                int out_idx = c * out_h * out_w + oh * out_w + ow;
                output[out_idx] = max_val;
            }
        }
    }
}

// 全连接层
void fully_connected(const int8_t *input, const int8_t weights[][FC1_IN],
                    const int8_t *biases, int in_size, int out_size,
                    int8_t *output, int shift, int use_relu) {
    for (int i = 0; i < out_size; i++) {
        int32_t sum = 0;

        // 矩阵乘法
        for (int j = 0; j < in_size; j++) {
            sum += (int32_t)weights[i][j] * (int32_t)input[j];
        }

        // 加偏置
        sum += (int32_t)biases[i];

        // 可选的ReLU激活
        if (use_relu) {
            sum = RELU(sum);
        }

        // 缩放到INT8范围
        if (shift > 0) {
            sum = sum >> shift;
        }

        // 截断并保存
        output[i] = CLAMP_INT8(sum);
    }
}

// Float inference helpers
static inline float dequantize_int8(int8_t v, float scale) {
    return (float)v / scale;
}

// 2D convolution (5x5 kernel, stride=1, valid), float path
void conv2d_5x5_float(const float *input, int in_h, int in_w, int in_c,
                      const int8_t weights[][150], const int8_t *biases,
                      float scale_w, float scale_b, int out_c,
                      float *output, int use_relu) {
    int out_h = in_h - 4;
    int out_w = in_w - 4;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_c; ic++) {
                    for (int kh = 0; kh < 5; kh++) {
                        for (int kw = 0; kw < 5; kw++) {
                            int in_idx = (ic * in_h * in_w) + ((oh + kh) * in_w) + (ow + kw);
                            int w_idx = ic * 25 + kh * 5 + kw;
                            float w = dequantize_int8(weights[oc][w_idx], scale_w);
                            sum += input[in_idx] * w;
                        }
                    }
                }
                sum += dequantize_int8(biases[oc], scale_b);
                if (use_relu) {
                    sum = RELU(sum);
                }
                output[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
}

// First conv (single input channel), float path
void conv2d_5x5_first_float(const float *input, int in_h, int in_w,
                            const int8_t weights[][25], const int8_t *biases,
                            float scale_w, float scale_b, int out_c,
                            float *output, int use_relu) {
    int out_h = in_h - 4;
    int out_w = in_w - 4;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;
                for (int kh = 0; kh < 5; kh++) {
                    for (int kw = 0; kw < 5; kw++) {
                        int in_idx = (oh + kh) * in_w + (ow + kw);
                        int w_idx = kh * 5 + kw;
                        float w = dequantize_int8(weights[oc][w_idx], scale_w);
                        sum += input[in_idx] * w;
                    }
                }
                sum += dequantize_int8(biases[oc], scale_b);
                if (use_relu) {
                    sum = RELU(sum);
                }
                output[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
}

// 2x2 max pool (stride=2), float path
void maxpool_2x2_float(const float *input, int in_h, int in_w, int channels,
                       float *output) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int in_base = c * in_h * in_w;
                int ih = oh * 2;
                int iw = ow * 2;

                float val0 = input[in_base + ih * in_w + iw];
                float val1 = input[in_base + ih * in_w + iw + 1];
                float val2 = input[in_base + (ih + 1) * in_w + iw];
                float val3 = input[in_base + (ih + 1) * in_w + iw + 1];

                float max_val = MAX(MAX(val0, val1), MAX(val2, val3));
                output[c * out_h * out_w + oh * out_w + ow] = max_val;
            }
        }
    }
}

// Fully connected layer, float path
void fully_connected_float(const float *input, const int8_t *weights,
                           const int8_t *biases, float scale_w, float scale_b,
                           int in_size, int out_size, float *output, int use_relu) {
    for (int i = 0; i < out_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_size; j++) {
            float w = dequantize_int8(weights[i * in_size + j], scale_w);
            sum += w * input[j];
        }
        sum += dequantize_int8(biases[i], scale_b);
        if (use_relu) {
            sum = RELU(sum);
        }
        output[i] = sum;
    }
}


static int load_image_from_file(const char *path, int8_t *buffer, size_t len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open image file: %s\n", path);
        return 0;
    }
    size_t n = fread(buffer, 1, len, f);
    fclose(f);
    if (n != len) {
        printf("Image file size %zu != %zu\n", n, len);
        return 0;
    }
    return 1;
}

// TinyLeNet inference (float path using dequantized weights)
int inference(const int8_t *input_pixels) {
    static float input_f[INPUT_H * INPUT_W];
    static float conv1_out[CONV1_OUT_C * CONV1_OUT_H * CONV1_OUT_W];
    static float pool1_out[CONV1_OUT_C * POOL1_OUT_H * POOL1_OUT_W];
    static float conv2_out[CONV2_OUT_C * CONV2_OUT_H * CONV2_OUT_W];
    static float pool2_out[CONV2_OUT_C * POOL2_OUT_H * POOL2_OUT_W];
    static float fc1_out[FC1_OUT];
    static float fc2_out[FC2_OUT];

    printf("Start inference...\n");

    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
        float x = (float)input_pixels[i] / 127.0f;
        input_f[i] = (x - INPUT_MEAN) / INPUT_STD;
    }

    printf("Conv1: 1@28x28 -> 6@24x24...\n");
    conv2d_5x5_first_float(input_f, INPUT_H, INPUT_W,
                           CONV1_WEIGHTS, CONV1_BIASES,
                           SCALE_CONV1_W, SCALE_CONV1_B,
                           CONV1_OUT_C, conv1_out, 1);

    printf("Pool1: 6@24x24 -> 6@12x12...\n");
    maxpool_2x2_float(conv1_out, CONV1_OUT_H, CONV1_OUT_W, CONV1_OUT_C, pool1_out);

    printf("Conv2: 6@12x12 -> 16@8x8...\n");
    conv2d_5x5_float(pool1_out, POOL1_OUT_H, POOL1_OUT_W, CONV1_OUT_C,
                     CONV2_WEIGHTS, CONV2_BIASES,
                     SCALE_CONV2_W, SCALE_CONV2_B,
                     CONV2_OUT_C, conv2_out, 1);

    printf("Pool2: 16@8x8 -> 16@4x4...\n");
    maxpool_2x2_float(conv2_out, CONV2_OUT_H, CONV2_OUT_W, CONV2_OUT_C, pool2_out);

    printf("FC1: 256 -> 32...\n");
    fully_connected_float(pool2_out, (const int8_t *)FC1_WEIGHTS, FC1_BIASES,
                          SCALE_FC1_W, SCALE_FC1_B, FC1_IN, FC1_OUT, fc1_out, 1);

    printf("FC2: 32 -> 10...\n");
    fully_connected_float(fc1_out, (const int8_t *)FC2_WEIGHTS, FC2_BIASES,
                          SCALE_FC2_W, SCALE_FC2_B, FC1_OUT, FC2_OUT, fc2_out, 0);

    int max_index = 0;
    float max_val = fc2_out[0];

    printf("Final scores: [");
    for (int i = 0; i < FC2_OUT; i++) {
        printf("%.3f ", fc2_out[i]);
        if (fc2_out[i] > max_val) {
            max_val = fc2_out[i];
            max_index = i;
        }
    }
    printf("]\n");

    return max_index;
}

int main(int argc, char **argv) {
#if defined(_WIN32) || defined(_WIN64)
    // Windows 系统（包括 32 位和 64 位）
    system("chcp 65001 > nul");
#endif


    const int8_t *input_ptr = test_image;
    int8_t input_buf[INPUT_H * INPUT_W];
    int label = TEST_IMAGE_LABEL;

    if (argc >= 2) {
        if (!load_image_from_file(argv[1], input_buf, INPUT_H * INPUT_W)) {
            return 1;
        }
        input_ptr = input_buf;
    }
    if (argc >= 3) {
        label = atoi(argv[2]);
    }

    printf("=== TinyLeNet MNIST 推理测试 ===\n");
    printf("网络结构:\n");
    printf("  Conv1: 1@28x28 -> 6@24x24 (5x5 kernel)\n");
    printf("  Pool1: 6@24x24 -> 6@12x12 (2x2 max pool)\n");
    printf("  Conv2: 6@12x12 -> 16@8x8 (5x5 kernel)\n");
    printf("  Pool2: 16@8x8 -> 16@4x4 (2x2 max pool)\n");
    printf("  FC1: 256 -> 32\n");
    printf("  FC2: 32 -> 10\n");
    printf("===================================\n\n");

    printf("测试图片真实标签: %d\n\n", label);

    int prediction = inference(input_ptr);

    printf("\n预测结果: %d\n", prediction);
    printf("Prediction: %d\n", prediction);

    if (prediction >= 0 && prediction <= 9 && prediction == label) {
        printf("✓ 测试通过！预测正确。\n");
    } else {
        printf("✗ 测试失败！预测结果与真实标签不符。\n");
    }

    return 0;
}
