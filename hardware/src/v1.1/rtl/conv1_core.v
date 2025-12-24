`timescale 1ns/1ps

module conv1_core #(
    parameter IMG_WIDTH = 28,
    parameter DATA_WIDTH = 8,
    parameter WEIGHT_WIDTH = 8,
    parameter OUT_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [DATA_WIDTH-1:0] pixel_in,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch0, w01_ch0, w02_ch0, w03_ch0, w04_ch0,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch0, w11_ch0, w12_ch0, w13_ch0, w14_ch0,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch0, w21_ch0, w22_ch0, w23_ch0, w24_ch0,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch0, w31_ch0, w32_ch0, w33_ch0, w34_ch0,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch0, w41_ch0, w42_ch0, w43_ch0, w44_ch0,
    input wire signed [WEIGHT_WIDTH-1:0] bias_ch0,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch1, w01_ch1, w02_ch1, w03_ch1, w04_ch1,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch1, w11_ch1, w12_ch1, w13_ch1, w14_ch1,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch1, w21_ch1, w22_ch1, w23_ch1, w24_ch1,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch1, w31_ch1, w32_ch1, w33_ch1, w34_ch1,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch1, w41_ch1, w42_ch1, w43_ch1, w44_ch1,
    input wire signed [WEIGHT_WIDTH-1:0] bias_ch1,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch2, w01_ch2, w02_ch2, w03_ch2, w04_ch2,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch2, w11_ch2, w12_ch2, w13_ch2, w14_ch2,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch2, w21_ch2, w22_ch2, w23_ch2, w24_ch2,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch2, w31_ch2, w32_ch2, w33_ch2, w34_ch2,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch2, w41_ch2, w42_ch2, w43_ch2, w44_ch2,
    input wire signed [WEIGHT_WIDTH-1:0] bias_ch2,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch3, w01_ch3, w02_ch3, w03_ch3, w04_ch3,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch3, w11_ch3, w12_ch3, w13_ch3, w14_ch3,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch3, w21_ch3, w22_ch3, w23_ch3, w24_ch3,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch3, w31_ch3, w32_ch3, w33_ch3, w34_ch3,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch3, w41_ch3, w42_ch3, w43_ch3, w44_ch3,
    input wire signed [WEIGHT_WIDTH-1:0] bias_ch3,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch4, w01_ch4, w02_ch4, w03_ch4, w04_ch4,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch4, w11_ch4, w12_ch4, w13_ch4, w14_ch4,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch4, w21_ch4, w22_ch4, w23_ch4, w24_ch4,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch4, w31_ch4, w32_ch4, w33_ch4, w34_ch4,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch4, w41_ch4, w42_ch4, w43_ch4, w44_ch4,
    input wire signed [WEIGHT_WIDTH-1:0] bias_ch4,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch5, w01_ch5, w02_ch5, w03_ch5, w04_ch5,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch5, w11_ch5, w12_ch5, w13_ch5, w14_ch5,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch5, w21_ch5, w22_ch5, w23_ch5, w24_ch5,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch5, w31_ch5, w32_ch5, w33_ch5, w34_ch5,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch5, w41_ch5, w42_ch5, w43_ch5, w44_ch5,
    input wire signed [WEIGHT_WIDTH-1:0] bias_ch5,

    output wire signed [OUT_WIDTH-1:0] result_ch0,
    output wire signed [OUT_WIDTH-1:0] result_ch1,
    output wire signed [OUT_WIDTH-1:0] result_ch2,
    output wire signed [OUT_WIDTH-1:0] result_ch3,
    output wire signed [OUT_WIDTH-1:0] result_ch4,
    output wire signed [OUT_WIDTH-1:0] result_ch5,
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
    // 2. Zero-extend pixels
    // ===============================================
    localparam PE_PIXEL_WIDTH = DATA_WIDTH + 1;

    wire signed [PE_PIXEL_WIDTH-1:0] p00 = {1'b0, w00_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p01 = {1'b0, w01_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p02 = {1'b0, w02_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p03 = {1'b0, w03_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p04 = {1'b0, w04_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p10 = {1'b0, w10_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p11 = {1'b0, w11_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p12 = {1'b0, w12_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p13 = {1'b0, w13_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p14 = {1'b0, w14_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p20 = {1'b0, w20_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p21 = {1'b0, w21_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p22 = {1'b0, w22_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p23 = {1'b0, w23_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p24 = {1'b0, w24_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p30 = {1'b0, w30_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p31 = {1'b0, w31_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p32 = {1'b0, w32_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p33 = {1'b0, w33_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p34 = {1'b0, w34_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p40 = {1'b0, w40_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p41 = {1'b0, w41_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p42 = {1'b0, w42_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p43 = {1'b0, w43_px};
    wire signed [PE_PIXEL_WIDTH-1:0] p44 = {1'b0, w44_px};

    // ===============================================
    // 3. PE group (6 channels)
    // ===============================================
    conv_pe_group #(
        .PIXEL_WIDTH  (PE_PIXEL_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .OUT_WIDTH    (OUT_WIDTH)
    ) u_pe_group (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (window_valid),
        .p00(p00), .p01(p01), .p02(p02), .p03(p03), .p04(p04),
        .p10(p10), .p11(p11), .p12(p12), .p13(p13), .p14(p14),
        .p20(p20), .p21(p21), .p22(p22), .p23(p23), .p24(p24),
        .p30(p30), .p31(p31), .p32(p32), .p33(p33), .p34(p34),
        .p40(p40), .p41(p41), .p42(p42), .p43(p43), .p44(p44),
        .w00_ch0(w00_ch0), .w01_ch0(w01_ch0), .w02_ch0(w02_ch0), .w03_ch0(w03_ch0), .w04_ch0(w04_ch0),
        .w10_ch0(w10_ch0), .w11_ch0(w11_ch0), .w12_ch0(w12_ch0), .w13_ch0(w13_ch0), .w14_ch0(w14_ch0),
        .w20_ch0(w20_ch0), .w21_ch0(w21_ch0), .w22_ch0(w22_ch0), .w23_ch0(w23_ch0), .w24_ch0(w24_ch0),
        .w30_ch0(w30_ch0), .w31_ch0(w31_ch0), .w32_ch0(w32_ch0), .w33_ch0(w33_ch0), .w34_ch0(w34_ch0),
        .w40_ch0(w40_ch0), .w41_ch0(w41_ch0), .w42_ch0(w42_ch0), .w43_ch0(w43_ch0), .w44_ch0(w44_ch0),
        .bias_ch0(bias_ch0),
        .w00_ch1(w00_ch1), .w01_ch1(w01_ch1), .w02_ch1(w02_ch1), .w03_ch1(w03_ch1), .w04_ch1(w04_ch1),
        .w10_ch1(w10_ch1), .w11_ch1(w11_ch1), .w12_ch1(w12_ch1), .w13_ch1(w13_ch1), .w14_ch1(w14_ch1),
        .w20_ch1(w20_ch1), .w21_ch1(w21_ch1), .w22_ch1(w22_ch1), .w23_ch1(w23_ch1), .w24_ch1(w24_ch1),
        .w30_ch1(w30_ch1), .w31_ch1(w31_ch1), .w32_ch1(w32_ch1), .w33_ch1(w33_ch1), .w34_ch1(w34_ch1),
        .w40_ch1(w40_ch1), .w41_ch1(w41_ch1), .w42_ch1(w42_ch1), .w43_ch1(w43_ch1), .w44_ch1(w44_ch1),
        .bias_ch1(bias_ch1),
        .w00_ch2(w00_ch2), .w01_ch2(w01_ch2), .w02_ch2(w02_ch2), .w03_ch2(w03_ch2), .w04_ch2(w04_ch2),
        .w10_ch2(w10_ch2), .w11_ch2(w11_ch2), .w12_ch2(w12_ch2), .w13_ch2(w13_ch2), .w14_ch2(w14_ch2),
        .w20_ch2(w20_ch2), .w21_ch2(w21_ch2), .w22_ch2(w22_ch2), .w23_ch2(w23_ch2), .w24_ch2(w24_ch2),
        .w30_ch2(w30_ch2), .w31_ch2(w31_ch2), .w32_ch2(w32_ch2), .w33_ch2(w33_ch2), .w34_ch2(w34_ch2),
        .w40_ch2(w40_ch2), .w41_ch2(w41_ch2), .w42_ch2(w42_ch2), .w43_ch2(w43_ch2), .w44_ch2(w44_ch2),
        .bias_ch2(bias_ch2),
        .w00_ch3(w00_ch3), .w01_ch3(w01_ch3), .w02_ch3(w02_ch3), .w03_ch3(w03_ch3), .w04_ch3(w04_ch3),
        .w10_ch3(w10_ch3), .w11_ch3(w11_ch3), .w12_ch3(w12_ch3), .w13_ch3(w13_ch3), .w14_ch3(w14_ch3),
        .w20_ch3(w20_ch3), .w21_ch3(w21_ch3), .w22_ch3(w22_ch3), .w23_ch3(w23_ch3), .w24_ch3(w24_ch3),
        .w30_ch3(w30_ch3), .w31_ch3(w31_ch3), .w32_ch3(w32_ch3), .w33_ch3(w33_ch3), .w34_ch3(w34_ch3),
        .w40_ch3(w40_ch3), .w41_ch3(w41_ch3), .w42_ch3(w42_ch3), .w43_ch3(w43_ch3), .w44_ch3(w44_ch3),
        .bias_ch3(bias_ch3),
        .w00_ch4(w00_ch4), .w01_ch4(w01_ch4), .w02_ch4(w02_ch4), .w03_ch4(w03_ch4), .w04_ch4(w04_ch4),
        .w10_ch4(w10_ch4), .w11_ch4(w11_ch4), .w12_ch4(w12_ch4), .w13_ch4(w13_ch4), .w14_ch4(w14_ch4),
        .w20_ch4(w20_ch4), .w21_ch4(w21_ch4), .w22_ch4(w22_ch4), .w23_ch4(w23_ch4), .w24_ch4(w24_ch4),
        .w30_ch4(w30_ch4), .w31_ch4(w31_ch4), .w32_ch4(w32_ch4), .w33_ch4(w33_ch4), .w34_ch4(w34_ch4),
        .w40_ch4(w40_ch4), .w41_ch4(w41_ch4), .w42_ch4(w42_ch4), .w43_ch4(w43_ch4), .w44_ch4(w44_ch4),
        .bias_ch4(bias_ch4),
        .w00_ch5(w00_ch5), .w01_ch5(w01_ch5), .w02_ch5(w02_ch5), .w03_ch5(w03_ch5), .w04_ch5(w04_ch5),
        .w10_ch5(w10_ch5), .w11_ch5(w11_ch5), .w12_ch5(w12_ch5), .w13_ch5(w13_ch5), .w14_ch5(w14_ch5),
        .w20_ch5(w20_ch5), .w21_ch5(w21_ch5), .w22_ch5(w22_ch5), .w23_ch5(w23_ch5), .w24_ch5(w24_ch5),
        .w30_ch5(w30_ch5), .w31_ch5(w31_ch5), .w32_ch5(w32_ch5), .w33_ch5(w33_ch5), .w34_ch5(w34_ch5),
        .w40_ch5(w40_ch5), .w41_ch5(w41_ch5), .w42_ch5(w42_ch5), .w43_ch5(w43_ch5), .w44_ch5(w44_ch5),
        .bias_ch5(bias_ch5),
        .result_ch0(result_ch0),
        .result_ch1(result_ch1),
        .result_ch2(result_ch2),
        .result_ch3(result_ch3),
        .result_ch4(result_ch4),
        .result_ch5(result_ch5),
        .result_valid(result_valid)
    );

endmodule
