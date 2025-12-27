`timescale 1ns/1ps

module conv_pe_5x5 #(
    parameter PIXEL_WIDTH = 9,
    parameter WEIGHT_WIDTH = 8,
    parameter BIAS_WIDTH = 32,
    parameter OUT_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,

    // 25 个输入像素 (注意：位宽现在是 PIXEL_WIDTH)
    input wire signed [PIXEL_WIDTH-1:0] p00, p01, p02, p03, p04,
    input wire signed [PIXEL_WIDTH-1:0] p10, p11, p12, p13, p14,
    input wire signed [PIXEL_WIDTH-1:0] p20, p21, p22, p23, p24,
    input wire signed [PIXEL_WIDTH-1:0] p30, p31, p32, p33, p34,
    input wire signed [PIXEL_WIDTH-1:0] p40, p41, p42, p43, p44,

    // 25 个权重 (位宽是 WEIGHT_WIDTH)
    input wire signed [WEIGHT_WIDTH-1:0] w00, w01, w02, w03, w04,
    input wire signed [WEIGHT_WIDTH-1:0] w10, w11, w12, w13, w14,
    input wire signed [WEIGHT_WIDTH-1:0] w20, w21, w22, w23, w24,
    input wire signed [WEIGHT_WIDTH-1:0] w30, w31, w32, w33, w34,
    input wire signed [WEIGHT_WIDTH-1:0] w40, w41, w42, w43, w44,
    
    input wire signed [BIAS_WIDTH-1:0] bias,

    output reg signed [OUT_WIDTH-1:0] result,
    output reg result_valid
);

    // ==========================================
    // Stage 1: 并行乘法
    // ==========================================
    // 9bit * 8bit 结果需要 17bit
    reg signed [16:0] m00, m01, m02, m03, m04;
    reg signed [16:0] m10, m11, m12, m13, m14;
    reg signed [16:0] m20, m21, m22, m23, m24;
    reg signed [16:0] m30, m31, m32, m33, m34;
    reg signed [16:0] m40, m41, m42, m43, m44;
    
    reg valid_s1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1 <= 0;
            m00 <= 0; // 简写清零，实际建议全写
        end else begin
            valid_s1 <= valid_in;
            if (valid_in) begin
                // Verilog 会自动处理不同位宽的有符号乘法，结果扩展到 LHS 位宽
                m00 <= p00 * w00; m01 <= p01 * w01; m02 <= p02 * w02; m03 <= p03 * w03; m04 <= p04 * w04;
                m10 <= p10 * w10; m11 <= p11 * w11; m12 <= p12 * w12; m13 <= p13 * w13; m14 <= p14 * w14;
                m20 <= p20 * w20; m21 <= p21 * w21; m22 <= p22 * w22; m23 <= p23 * w23; m24 <= p24 * w24;
                m30 <= p30 * w30; m31 <= p31 * w31; m32 <= p32 * w32; m33 <= p33 * w33; m34 <= p34 * w34;
                m40 <= p40 * w40; m41 <= p41 * w41; m42 <= p42 * w42; m43 <= p43 * w43; m44 <= p44 * w44;
            end
        end
    end

    // ==========================================
    // Stage 2: 累加求和
    // ==========================================
    reg signed [31:0] sum;
    reg valid_s2;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sum <= 0;
            valid_s2 <= 0;
        end else begin
            valid_s2 <= valid_s1;
            if (valid_s1) begin
                sum <= m00 + m01 + m02 + m03 + m04 +
                       m10 + m11 + m12 + m13 + m14 +
                       m20 + m21 + m22 + m23 + m24 +
                       m30 + m31 + m32 + m33 + m34 +
                       m40 + m41 + m42 + m43 + m44 +
                       bias; // Verilog 会自动符号扩展 bias
            end
        end
    end

    // ==========================================
    // Stage 3: ReLU 激活
    // ==========================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 0;
            result_valid <= 0;
        end else begin
            result_valid <= valid_s2;
            if (valid_s2) begin
                // ReLU: 负数变0
                if (sum[31] == 1'b1) 
                    result <= 0;
                else
                    result <= sum;
            end
        end
    end

endmodule
