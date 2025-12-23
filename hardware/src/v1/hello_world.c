#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include "system.h"
#include "io.h"
#include "altera_avalon_uart_regs.h"
#include "altera_avalon_pio_regs.h"
#include "altera_avalon_timer_regs.h" // [新增] 必须包含定时器寄存器定义

// 引入模型权重和推理逻辑
#include "model_weights.h"

// --- 硬件配置区 ---
#define LAYER1_SHIFT 8

// 检查 Timer 基地址 (如果 Qsys 里叫 timer_0，system.h 里就是 TIMER_0_BASE)
//#ifndef TIMER_0_BASE
//#error "请在 Qsys 中添加 Interval Timer 组件并命名为 timer_0，或者在 system.h 中查找正确的定时器基地址宏名"
//#endif

// --- 全局变量 ---
int8_t input_buffer[784]; 

// 宏定义
#define RELU(x) ((x) > 0 ? (x) : 0)
#define CLAMP_INT8(x) (((x) > 127) ? 127 : (((x) < -128) ? -128 : (int8_t)(x)))

// ==========================================
// [新增] 计时器驱动函数
// Nios II 的 Timer 通常是倒计数的
// ==========================================

// 开始计时：重置定时器并开始倒数
void timer_start() {
    // 1. 停止定时器，清除状态
    IOWR_ALTERA_AVALON_TIMER_CONTROL(TIMER_0_BASE, 0);
    IOWR_ALTERA_AVALON_TIMER_STATUS(TIMER_0_BASE, 0);

    // 2. 设置周期为最大值 (0xFFFFFFFF)
    // 这样可以保证很长时间不溢出
    IOWR_ALTERA_AVALON_TIMER_PERIODL(TIMER_0_BASE, 0xFFFF);
    IOWR_ALTERA_AVALON_TIMER_PERIODH(TIMER_0_BASE, 0xFFFF);

    // 3. 启动定时器 (START位 + CONT位连续运行)
    IOWR_ALTERA_AVALON_TIMER_CONTROL(TIMER_0_BASE,
        ALTERA_AVALON_TIMER_CONTROL_START_MSK |
        ALTERA_AVALON_TIMER_CONTROL_CONT_MSK);
}

// 停止计时并返回消耗的时钟周期数 (Ticks)
uint32_t timer_stop() {
    // 1. 触发快照 (写任意值到 SNAPL)
    // 这一步会将当前的计数值锁存到 SNAPL 和 SNAPH 中
    IOWR_ALTERA_AVALON_TIMER_SNAPL(TIMER_0_BASE, 0);

    // 2. 读取快照值
    uint32_t snap_lo = IORD_ALTERA_AVALON_TIMER_SNAPL(TIMER_0_BASE);
    uint32_t snap_hi = IORD_ALTERA_AVALON_TIMER_SNAPH(TIMER_0_BASE);
    uint32_t current_value = (snap_hi << 16) | snap_lo;

    // 3. 计算经过的 Ticks
    // 因为是倒计数，所以消耗的时间 = 最大值 - 当前值
    return 0xFFFFFFFF - current_value;
}

// 将 Ticks 转换为毫秒 (根据系统时钟频率)
float ticks_to_ms(uint32_t ticks) {
    // ALT_CPU_FREQ 是 system.h 中定义的 CPU 频率 (例如 50000000 代表 50MHz)
    return (float)ticks * 1000.0f / (float)ALT_CPU_FREQ;
}

// ==========================================
// 1. 数码管驱动函数
// ==========================================
void seg_display(int number) {
    int point = 0;       // 不显示小数点
    int seg_en = 1;      // 开启显示
    int sign = 0;        // 不显示负号

    uint32_t pio_val = 0;
    pio_val |= (number & 0xFFFFF);
    pio_val |= (point << 20);
    pio_val |= (seg_en << 26);
    pio_val |= (sign << 27);

    IOWR_ALTERA_AVALON_PIO_DATA(OUT_PIO_BASE, pio_val);
}

// ==========================================
// 2. 串口与推理
// ==========================================

void uart_send_byte(uint8_t data) {
    while (!(IORD_ALTERA_AVALON_UART_STATUS(UART_0_BASE) & ALTERA_AVALON_UART_STATUS_TRDY_MSK));
    IOWR_ALTERA_AVALON_UART_TXDATA(UART_0_BASE, data);
}

uint8_t uart_receive_byte() {
    while (!(IORD_ALTERA_AVALON_UART_STATUS(UART_0_BASE) & ALTERA_AVALON_UART_STATUS_RRDY_MSK));
    return (uint8_t)IORD_ALTERA_AVALON_UART_RXDATA(UART_0_BASE);
}

void wait_for_image() {
    int i;
    printf("等待 PC 发送图片...\n");
    while (1) {
        if (uart_receive_byte() == 0xAA) break;
    }
    for (i = 0; i < 784; i++) {
        input_buffer[i] = (int8_t)uart_receive_byte();
    }
    if (uart_receive_byte() == 0x55) {
        printf("接收成功!\n");
        uart_send_byte('K');
    }
}

int inference(const int8_t *input_pixels) {
    int8_t  hidden_output[HIDDEN_SIZE];
    int32_t final_accum[OUTPUT_SIZE];
    int i, j;

    // Layer 1
    for (i = 0; i < HIDDEN_SIZE; i++) {
        int32_t sum = 0;
        for (j = 0; j < INPUT_SIZE; j++) {
            sum += (int32_t)W1[i][j] * (int32_t)input_pixels[j];
        }
        sum += (int32_t)B1[i];
        sum = RELU(sum);
        sum = sum >> LAYER1_SHIFT; 
        hidden_output[i] = CLAMP_INT8(sum);
    }

    // Layer 2
    for (i = 0; i < OUTPUT_SIZE; i++) {
        int32_t sum = 0;
        for (j = 0; j < HIDDEN_SIZE; j++) {
            sum += (int32_t)W2[i][j] * (int32_t)hidden_output[j];
        }
        sum += (int32_t)B2[i];
        final_accum[i] = sum;
    }

    // Argmax
    int max_index = 0;
    int32_t max_val = final_accum[0];
    for (i = 0; i < OUTPUT_SIZE; i++) {
        if (final_accum[i] > max_val) {
            max_val = final_accum[i];
            max_index = i;
        }
    }
    return max_index;
}

// ==========================================
// 3. 主函数
// ==========================================
int main() {
    printf("=== FPGA MNIST + Timer Stats ===\n");
    printf("CPU Freq: %d MHz\n", ALT_CPU_FREQ / 1000000); // 打印检测到的频率

    seg_display(0);

    while (1) {
        // 1. 等待图片
        wait_for_image();

        // 2. 开始计时
        timer_start();

        // 3. 推理
        int prediction = inference(input_buffer);

        // 4. 停止计时并计算
        uint32_t ticks = timer_stop();
        float time_ms = ticks_to_ms(ticks);

        // 5. 打印结果
        // 为了兼容 Small C Library (不支持 %f), 我们把浮点拆成整数和小数打印
        // 或者直接打印 Ticks
        int ms_int = (int)time_ms;
        int ms_frac = (int)((time_ms - ms_int) * 1000);

        printf("\n-----------------------------\n");
        printf(">>> 预测结果: %d <<<\n", prediction);
        printf(">>> 推理耗时: %d.%03d ms (%lu Ticks)\n", ms_int, ms_frac, ticks);
        printf("-----------------------------\n\n");

        // 6. 数码管显示
        seg_display(prediction);
    }

    return 0;
}
