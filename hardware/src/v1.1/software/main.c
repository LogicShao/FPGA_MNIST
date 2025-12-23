#include "system.h"
#include "altera_avalon_pio_regs.h"
#include <unistd.h> // for usleep if needed

// 定义控制位
#define CMD_VALID_MASK  0x100  // Bit 8
#define CMD_RST_MASK    0x200  // Bit 9 (假设你把复位接在 Bit 9)

void hardware_accelerated_inference(unsigned char *image_data) {
    
    // 1. 先复位一下加速器 (可选，确保状态机清零)
    // 拉低复位 (假设低电平复位有效，具体看你的 Verilog 逻辑)
    // 这里假设 Bit 9 是 1 为正常，0 为复位
    IOWR_ALTERA_AVALON_PIO_DATA(PIO_IMG_DATA_BASE, 0x000); 
    usleep(1);
    // 释放复位
    IOWR_ALTERA_AVALON_PIO_DATA(PIO_IMG_DATA_BASE, CMD_RST_MASK); 

    // 2. 循环发送 784 个像素
    for(int i = 0; i < 784; i++) {
        unsigned int pixel = image_data[i];
        
        // 组合数据：复位保持高(Bit9=1) + Valid置高(Bit8=1) + 像素值
        unsigned int data_valid = CMD_RST_MASK | CMD_VALID_MASK | pixel;
        
        // 组合数据：复位保持高(Bit9=1) + Valid拉低(Bit8=0) + 像素值
        unsigned int data_idle  = CMD_RST_MASK | pixel;
        
        // 发送脉冲
        IOWR_ALTERA_AVALON_PIO_DATA(PIO_IMG_DATA_BASE, data_valid);
        IOWR_ALTERA_AVALON_PIO_DATA(PIO_IMG_DATA_BASE, data_idle);
    }

    // 3. 读取结果
    int fpga_result = IORD_ALTERA_AVALON_PIO_DATA(PIO_RESULT_BASE);
    
    printf("FPGA Logic Result: %d\n", fpga_result);
}