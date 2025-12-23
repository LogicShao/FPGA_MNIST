`timescale 1ns/1ps

module layer1_window_gen #(
    parameter IMG_WIDTH = 28,  // 图片宽度
    parameter DATA_WIDTH = 8   // 数据位宽
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,             // 输入数据有效
    input wire [DATA_WIDTH-1:0] din, // 实时输入的一个像素 (Newest Pixel)

    // ============================================
    // 输出：5x5 矩阵窗口 (共 25 个像素)
    // 命名约定：w_行_列 (w_0_0 是窗口左上角/最旧，w_4_4 是窗口右下角/最新)
    // ============================================
    output wire [DATA_WIDTH-1:0] w00, w01, w02, w03, w04, // Row 0 (Top)
    output wire [DATA_WIDTH-1:0] w10, w11, w12, w13, w14, // Row 1
    output wire [DATA_WIDTH-1:0] w20, w21, w22, w23, w24, // Row 2
    output wire [DATA_WIDTH-1:0] w30, w31, w32, w33, w34, // Row 3
    output wire [DATA_WIDTH-1:0] w40, w41, w42, w43, w44, // Row 4 (Bottom - current)
    
    output reg window_valid // 只有当窗口完全填满有效数据时置 1
);

    // =========================================================
    // Part 1: Line Buffers (行缓存) - 负责纵向延迟
    // =========================================================
    // 我们需要 4 条长延迟线，每条延迟 IMG_WIDTH 个周期
    // 数据流向：din -> LB3 -> LB2 -> LB1 -> LB0 -> (丢弃)
    // 这样，din 是第 4 行，LB3出的是第 3 行... LB0出的是第 0 行
    
    reg [DATA_WIDTH-1:0] lb0 [0:IMG_WIDTH-1];
    reg [DATA_WIDTH-1:0] lb1 [0:IMG_WIDTH-1];
    reg [DATA_WIDTH-1:0] lb2 [0:IMG_WIDTH-1];
    reg [DATA_WIDTH-1:0] lb3 [0:IMG_WIDTH-1];
    
    // 定义 Line Buffer 的输出端 (Taps)
    wire [DATA_WIDTH-1:0] row0_out = lb0[IMG_WIDTH-1];
    wire [DATA_WIDTH-1:0] row1_out = lb1[IMG_WIDTH-1];
    wire [DATA_WIDTH-1:0] row2_out = lb2[IMG_WIDTH-1];
    wire [DATA_WIDTH-1:0] row3_out = lb3[IMG_WIDTH-1];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 复位清零 (在FPGA中这部分逻辑其实可以省略，依赖初始值)
            for(i=0; i<IMG_WIDTH; i=i+1) begin
                lb0[i] <= 0; lb1[i] <= 0; lb2[i] <= 0; lb3[i] <= 0;
            end
        end else if (valid_in) begin
            // 移位逻辑：将数据推入移位寄存器
            // 这是一个简单的 Shift Register 链
            
            // 1. 内部移位 (除了第一个元素)
            for(i=IMG_WIDTH-1; i>0; i=i-1) begin
                lb3[i] <= lb3[i-1];
                lb2[i] <= lb2[i-1];
                lb1[i] <= lb1[i-1];
                lb0[i] <= lb0[i-1];
            end
            
            // 2. 级联输入 (Chain connection)
            lb3[0] <= din;        // 当前数据进 LB3
            lb2[0] <= row3_out;   // LB3 的尾巴进 LB2
            lb1[0] <= row2_out;   // LB2 的尾巴进 LB1
            lb0[0] <= row1_out;   // LB1 的尾巴进 LB0
        end
    end

    // =========================================================
    // Part 2: Window Registers (窗口寄存器) - 负责横向延迟
    // =========================================================
    // 我们需要 5 个小的移位寄存器（长度为 5），分别对应窗口的 5 行
    
    reg [DATA_WIDTH-1:0] win_row0 [0:4];
    reg [DATA_WIDTH-1:0] win_row1 [0:4];
    reg [DATA_WIDTH-1:0] win_row2 [0:4];
    reg [DATA_WIDTH-1:0] win_row3 [0:4];
    reg [DATA_WIDTH-1:0] win_row4 [0:4];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 清零逻辑省略
        end else if (valid_in) begin
            // 每个窗口行都在做移位: [0]<=[1], [1]<=[2]... 
            // 注意：这里我们定义 index 4 为最新，0 为最老
            
            // Row 4 (Bottom) - 来自实时输入
            win_row4[4] <= din;
            win_row4[3] <= win_row4[4]; win_row4[2] <= win_row4[3]; win_row4[1] <= win_row4[2]; win_row4[0] <= win_row4[1];

            // Row 3 - 来自 LB3 输出 (注意要用 row3_out 之前的数据，即 lb3[IMG_WIDTH-1] 是刚吐出来的)
            // 修正：我们要接的是 LineBuffer 刚刚吐出来的那个数
            win_row3[4] <= row3_out; 
            win_row3[3] <= win_row3[4]; win_row3[2] <= win_row3[3]; win_row3[1] <= win_row3[2]; win_row3[0] <= win_row3[1];

            // Row 2
            win_row2[4] <= row2_out;
            win_row2[3] <= win_row2[4]; win_row2[2] <= win_row2[3]; win_row2[1] <= win_row2[2]; win_row2[0] <= win_row2[1];

            // Row 1
            win_row1[4] <= row1_out;
            win_row1[3] <= win_row1[4]; win_row1[2] <= win_row1[3]; win_row1[1] <= win_row1[2]; win_row1[0] <= win_row1[1];

            // Row 0 (Top)
            win_row0[4] <= row0_out;
            win_row0[3] <= win_row0[4]; win_row0[2] <= win_row0[3]; win_row0[1] <= win_row0[2]; win_row0[0] <= win_row0[1];
        end
    end

    // =========================================================
    // Part 3: 计数器与 Valid 控制 logic
    // =========================================================
    reg [9:0] x_cnt; // 列计数
    reg [9:0] y_cnt; // 行计数

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_cnt <= 0;
            y_cnt <= 0;
            window_valid <= 0;
        end else if (valid_in) begin
            // 坐标更新逻辑
            if (x_cnt == IMG_WIDTH - 1) begin
                x_cnt <= 0;
                y_cnt <= y_cnt + 1;
            end else begin
                x_cnt <= x_cnt + 1;
            end

            // Valid 判断逻辑：无 Padding 卷积 (Valid Convolution)
            // 只有当我们在第 4 行 (y_cnt >= 4) 且第 4 列 (x_cnt >= 4) 之后，窗口才充满了来自图像内部的数据
            if (y_cnt >= 4 && x_cnt >= 4) begin
                window_valid <= 1;
            end else begin
                window_valid <= 0;
            end
        end else begin
            window_valid <= 0; // 如果输入暂停，输出有效性也暂停
        end
    end

    // =========================================================
    // Part 4: 输出连线
    // =========================================================
    // 将寄存器映射到输出端口
    
    assign w00 = win_row0[0]; assign w01 = win_row0[1]; assign w02 = win_row0[2]; assign w03 = win_row0[3]; assign w04 = win_row0[4];
    assign w10 = win_row1[0]; assign w11 = win_row1[1]; assign w12 = win_row1[2]; assign w13 = win_row1[3]; assign w14 = win_row1[4];
    assign w20 = win_row2[0]; assign w21 = win_row2[1]; assign w22 = win_row2[2]; assign w23 = win_row2[3]; assign w24 = win_row2[4];
    assign w30 = win_row3[0]; assign w31 = win_row3[1]; assign w32 = win_row3[2]; assign w33 = win_row3[3]; assign w34 = win_row3[4];
    assign w40 = win_row4[0]; assign w41 = win_row4[1]; assign w42 = win_row4[2]; assign w43 = win_row4[3]; assign w44 = win_row4[4];

endmodule