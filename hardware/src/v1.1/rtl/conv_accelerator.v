`timescale 1ns/1ps

module conv_accelerator #(
    parameter IMG_WIDTH = 28,
    parameter DATA_WIDTH = 8,
    parameter WEIGHT_WIDTH = 8,
    parameter OUT_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [DATA_WIDTH-1:0] pixel_in,

    input wire signed [WEIGHT_WIDTH-1:0] w00, w01, w02, w03, w04,
    input wire signed [WEIGHT_WIDTH-1:0] w10, w11, w12, w13, w14,
    input wire signed [WEIGHT_WIDTH-1:0] w20, w21, w22, w23, w24,
    input wire signed [WEIGHT_WIDTH-1:0] w30, w31, w32, w33, w34,
    input wire signed [WEIGHT_WIDTH-1:0] w40, w41, w42, w43, w44,
    input wire signed [31:0] bias,

    output wire signed [OUT_WIDTH-1:0] result_ch0,
    output wire result_valid
);

    // ===============================================
    // 1. Window Generator
    // ===============================================
    wire [DATA_WIDTH-1:0] w00_px, w01_px, w02_px, w03_px, w04_px;
    wire [DATA_WIDTH-1:0] w10_px, w11_px, w12_px, w13_px, w14_px;
    wire [DATA_WIDTH-1:0] w20_px, w21_px, w22_px, w23_px, w24_px;
    wire [DATA_WIDTH-1:0] w30_px, w31_px, w32_px, w33_px, w34_px;
    wire [DATA_WIDTH-1:0] w40_px, w41_px, w42_px, w43_px, w44_px;
    wire window_valid;

    layer1_window_gen #(
        .IMG_WIDTH  (IMG_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_window_gen (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in),
        .din          (pixel_in),
        .w00(w00_px), .w01(w01_px), .w02(w02_px), .w03(w03_px), .w04(w04_px),
        .w10(w10_px), .w11(w11_px), .w12(w12_px), .w13(w13_px), .w14(w14_px),
        .w20(w20_px), .w21(w21_px), .w22(w22_px), .w23(w23_px), .w24(w24_px),
        .w30(w30_px), .w31(w31_px), .w32(w32_px), .w33(w33_px), .w34(w34_px),
        .w40(w40_px), .w41(w41_px), .w42(w42_px), .w43(w43_px), .w44(w44_px),
        .window_valid (window_valid)
    );

    // ===============================================
    // 2. Zero-extend pixels to keep unsigned range
    // ===============================================
    localparam PE_PIXEL_WIDTH = DATA_WIDTH + 1;

    wire signed [PE_PIXEL_WIDTH-1:0] p00 = {{1{w00_px[DATA_WIDTH-1]}}, w00_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p01 = {{1{w01_px[DATA_WIDTH-1]}}, w01_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p02 = {{1{w02_px[DATA_WIDTH-1]}}, w02_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p03 = {{1{w03_px[DATA_WIDTH-1]}}, w03_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p04 = {{1{w04_px[DATA_WIDTH-1]}}, w04_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p10 = {{1{w10_px[DATA_WIDTH-1]}}, w10_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p11 = {{1{w11_px[DATA_WIDTH-1]}}, w11_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p12 = {{1{w12_px[DATA_WIDTH-1]}}, w12_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p13 = {{1{w13_px[DATA_WIDTH-1]}}, w13_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p14 = {{1{w14_px[DATA_WIDTH-1]}}, w14_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p20 = {{1{w20_px[DATA_WIDTH-1]}}, w20_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p21 = {{1{w21_px[DATA_WIDTH-1]}}, w21_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p22 = {{1{w22_px[DATA_WIDTH-1]}}, w22_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p23 = {{1{w23_px[DATA_WIDTH-1]}}, w23_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p24 = {{1{w24_px[DATA_WIDTH-1]}}, w24_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p30 = {{1{w30_px[DATA_WIDTH-1]}}, w30_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p31 = {{1{w31_px[DATA_WIDTH-1]}}, w31_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p32 = {{1{w32_px[DATA_WIDTH-1]}}, w32_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p33 = {{1{w33_px[DATA_WIDTH-1]}}, w33_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p34 = {{1{w34_px[DATA_WIDTH-1]}}, w34_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p40 = {{1{w40_px[DATA_WIDTH-1]}}, w40_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p41 = {{1{w41_px[DATA_WIDTH-1]}}, w41_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p42 = {{1{w42_px[DATA_WIDTH-1]}}, w42_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p43 = {{1{w43_px[DATA_WIDTH-1]}}, w43_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p44 = {{1{w44_px[DATA_WIDTH-1]}}, w44_px};

    // ===============================================
    // 3. PE instance
    // ===============================================
    conv_pe_5x5 #(
        .PIXEL_WIDTH  (PE_PIXEL_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .BIAS_WIDTH   (32),
        .OUT_WIDTH    (OUT_WIDTH)
    ) u_pe (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (window_valid),
        .p00(p00), .p01(p01), .p02(p02), .p03(p03), .p04(p04),
        .p10(p10), .p11(p11), .p12(p12), .p13(p13), .p14(p14),
        .p20(p20), .p21(p21), .p22(p22), .p23(p23), .p24(p24),
        .p30(p30), .p31(p31), .p32(p32), .p33(p33), .p34(p34),
        .p40(p40), .p41(p41), .p42(p42), .p43(p43), .p44(p44),
        .w00(w00), .w01(w01), .w02(w02), .w03(w03), .w04(w04),
        .w10(w10), .w11(w11), .w12(w12), .w13(w13), .w14(w14),
        .w20(w20), .w21(w21), .w22(w22), .w23(w23), .w24(w24),
        .w30(w30), .w31(w31), .w32(w32), .w33(w33), .w34(w34),
        .w40(w40), .w41(w41), .w42(w42), .w43(w43), .w44(w44),
        .bias         (bias),
        .result       (result_ch0),
        .result_valid (result_valid)
    );

endmodule
