#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// 包含模型参数和测试图片
#include "model_weights.h"
#include "test_image.h"

// 量化推理配置
// 注意：model_weights.h 中包含 SCALE_W1/B1/W2/B2，这是训练时的缩放因子
// 但在INT8定点推理中，我们使用简化的定点运算策略

// 层间缩放策略：
// - Layer1输出右移位数（根据实际测试调整，范围6-10）
#define LAYER1_SHIFT 8
// - Layer2输出不需要缩放，直接用于argmax

// ReLU 宏定义
#define RELU(x) ((x) > 0 ? (x) : 0)

// 裁剪宏定义 (保持在 int8 范围内 -128 ~ 127)
#define CLAMP_INT8(x) (((x) > 127) ? 127 : (((x) < -128) ? -128 : (int8_t)(x)))

// 推理函数
int inference(const int8_t *input_pixels) {
    int8_t  hidden_output[HIDDEN_SIZE];
    int32_t final_accum[OUTPUT_SIZE];

    int i, j;

    // --- 第一层: FC (784 -> HIDDEN_SIZE) + ReLU ---
    for (i = 0; i < HIDDEN_SIZE; i++) {
        int32_t sum = 0;

        // 矩阵乘法：W1 * input
        for (j = 0; j < INPUT_SIZE; j++) {
            sum += (int32_t)W1[i][j] * (int32_t)input_pixels[j];
        }

        // 加偏置（bias缩放因子与weight接近，可以直接加）
        sum += (int32_t)B1[i];

        // ReLU激活
        sum = RELU(sum);

        // 缩放到INT8范围（右移以避免溢出）
        sum = sum >> LAYER1_SHIFT;

        // 截断并保存
        hidden_output[i] = CLAMP_INT8(sum);
    }

    // --- 第二层: FC (HIDDEN_SIZE -> OUTPUT_SIZE) ---
    for (i = 0; i < OUTPUT_SIZE; i++) {
        int32_t sum = 0;

        // 矩阵乘法：W2 * hidden
        for (j = 0; j < HIDDEN_SIZE; j++) {
            sum += (int32_t)W2[i][j] * (int32_t)hidden_output[j];
        }

        // 加偏置
        sum += (int32_t)B2[i];

        // 输出层不需要激活函数，直接用于argmax
        final_accum[i] = sum;
    }

    // --- Argmax: 找最大值 ---
    int max_index = 0;
    int32_t max_val = final_accum[0];

    printf("详细得分: [");
    for (i = 0; i < OUTPUT_SIZE; i++) {
        printf("%d ", final_accum[i]);
        if (final_accum[i] > max_val) {
            max_val = final_accum[i];
            max_index = i;
        }
    }
    printf("]\n");

    return max_index;
}

int main() {
#if defined(_WIN32) || defined(_WIN64)
    // Windows 系统（包括 32 位和 64 位）
    system("chcp 65001 > nul");
#endif

    printf("=== PC 端 MNIST 推理测试 ===\n");
    printf("测试图片真实标签: %d\n", TEST_IMAGE_LABEL);
    
    int prediction = inference(test_image);
    
    printf("\n预测结果: %d\n", prediction);
    
    if (prediction >= 0 && prediction <= 9 && prediction == TEST_IMAGE_LABEL) {
        printf("测试通过！逻辑正常。\n");
    } else {
        printf("结果异常！\n");
    }

    return 0;
}