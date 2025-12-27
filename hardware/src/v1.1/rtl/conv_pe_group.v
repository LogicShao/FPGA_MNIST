`timescale 1ns/1ps

module conv_pe_group #(
    parameter PIXEL_WIDTH = 9,
    parameter WEIGHT_WIDTH = 8,
    parameter BIAS_WIDTH = 32,
    parameter OUT_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,

    input wire signed [PIXEL_WIDTH-1:0] p00, p01, p02, p03, p04,
    input wire signed [PIXEL_WIDTH-1:0] p10, p11, p12, p13, p14,
    input wire signed [PIXEL_WIDTH-1:0] p20, p21, p22, p23, p24,
    input wire signed [PIXEL_WIDTH-1:0] p30, p31, p32, p33, p34,
    input wire signed [PIXEL_WIDTH-1:0] p40, p41, p42, p43, p44,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch0, w01_ch0, w02_ch0, w03_ch0, w04_ch0,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch0, w11_ch0, w12_ch0, w13_ch0, w14_ch0,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch0, w21_ch0, w22_ch0, w23_ch0, w24_ch0,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch0, w31_ch0, w32_ch0, w33_ch0, w34_ch0,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch0, w41_ch0, w42_ch0, w43_ch0, w44_ch0,
    input wire signed [BIAS_WIDTH-1:0] bias_ch0,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch1, w01_ch1, w02_ch1, w03_ch1, w04_ch1,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch1, w11_ch1, w12_ch1, w13_ch1, w14_ch1,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch1, w21_ch1, w22_ch1, w23_ch1, w24_ch1,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch1, w31_ch1, w32_ch1, w33_ch1, w34_ch1,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch1, w41_ch1, w42_ch1, w43_ch1, w44_ch1,
    input wire signed [BIAS_WIDTH-1:0] bias_ch1,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch2, w01_ch2, w02_ch2, w03_ch2, w04_ch2,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch2, w11_ch2, w12_ch2, w13_ch2, w14_ch2,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch2, w21_ch2, w22_ch2, w23_ch2, w24_ch2,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch2, w31_ch2, w32_ch2, w33_ch2, w34_ch2,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch2, w41_ch2, w42_ch2, w43_ch2, w44_ch2,
    input wire signed [BIAS_WIDTH-1:0] bias_ch2,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch3, w01_ch3, w02_ch3, w03_ch3, w04_ch3,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch3, w11_ch3, w12_ch3, w13_ch3, w14_ch3,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch3, w21_ch3, w22_ch3, w23_ch3, w24_ch3,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch3, w31_ch3, w32_ch3, w33_ch3, w34_ch3,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch3, w41_ch3, w42_ch3, w43_ch3, w44_ch3,
    input wire signed [BIAS_WIDTH-1:0] bias_ch3,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch4, w01_ch4, w02_ch4, w03_ch4, w04_ch4,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch4, w11_ch4, w12_ch4, w13_ch4, w14_ch4,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch4, w21_ch4, w22_ch4, w23_ch4, w24_ch4,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch4, w31_ch4, w32_ch4, w33_ch4, w34_ch4,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch4, w41_ch4, w42_ch4, w43_ch4, w44_ch4,
    input wire signed [BIAS_WIDTH-1:0] bias_ch4,

    input wire signed [WEIGHT_WIDTH-1:0] w00_ch5, w01_ch5, w02_ch5, w03_ch5, w04_ch5,
    input wire signed [WEIGHT_WIDTH-1:0] w10_ch5, w11_ch5, w12_ch5, w13_ch5, w14_ch5,
    input wire signed [WEIGHT_WIDTH-1:0] w20_ch5, w21_ch5, w22_ch5, w23_ch5, w24_ch5,
    input wire signed [WEIGHT_WIDTH-1:0] w30_ch5, w31_ch5, w32_ch5, w33_ch5, w34_ch5,
    input wire signed [WEIGHT_WIDTH-1:0] w40_ch5, w41_ch5, w42_ch5, w43_ch5, w44_ch5,
    input wire signed [BIAS_WIDTH-1:0] bias_ch5,

    output wire signed [OUT_WIDTH-1:0] result_ch0,
    output wire signed [OUT_WIDTH-1:0] result_ch1,
    output wire signed [OUT_WIDTH-1:0] result_ch2,
    output wire signed [OUT_WIDTH-1:0] result_ch3,
    output wire signed [OUT_WIDTH-1:0] result_ch4,
    output wire signed [OUT_WIDTH-1:0] result_ch5,
    output wire result_valid
);

    wire valid0;
    wire valid1;
    wire valid2;
    wire valid3;
    wire valid4;
    wire valid5;

    assign result_valid = valid0;

    conv_pe_5x5 #(
        .PIXEL_WIDTH  (PIXEL_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .BIAS_WIDTH   (BIAS_WIDTH),
        .OUT_WIDTH    (OUT_WIDTH)
    ) u_pe0 (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (valid_in),
        .p00(p00), .p01(p01), .p02(p02), .p03(p03), .p04(p04),
        .p10(p10), .p11(p11), .p12(p12), .p13(p13), .p14(p14),
        .p20(p20), .p21(p21), .p22(p22), .p23(p23), .p24(p24),
        .p30(p30), .p31(p31), .p32(p32), .p33(p33), .p34(p34),
        .p40(p40), .p41(p41), .p42(p42), .p43(p43), .p44(p44),
        .w00(w00_ch0), .w01(w01_ch0), .w02(w02_ch0), .w03(w03_ch0), .w04(w04_ch0),
        .w10(w10_ch0), .w11(w11_ch0), .w12(w12_ch0), .w13(w13_ch0), .w14(w14_ch0),
        .w20(w20_ch0), .w21(w21_ch0), .w22(w22_ch0), .w23(w23_ch0), .w24(w24_ch0),
        .w30(w30_ch0), .w31(w31_ch0), .w32(w32_ch0), .w33(w33_ch0), .w34(w34_ch0),
        .w40(w40_ch0), .w41(w41_ch0), .w42(w42_ch0), .w43(w43_ch0), .w44(w44_ch0),
        .bias         (bias_ch0),
        .result       (result_ch0),
        .result_valid (valid0)
    );

    conv_pe_5x5 #(
        .PIXEL_WIDTH  (PIXEL_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .BIAS_WIDTH   (BIAS_WIDTH),
        .OUT_WIDTH    (OUT_WIDTH)
    ) u_pe1 (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (valid_in),
        .p00(p00), .p01(p01), .p02(p02), .p03(p03), .p04(p04),
        .p10(p10), .p11(p11), .p12(p12), .p13(p13), .p14(p14),
        .p20(p20), .p21(p21), .p22(p22), .p23(p23), .p24(p24),
        .p30(p30), .p31(p31), .p32(p32), .p33(p33), .p34(p34),
        .p40(p40), .p41(p41), .p42(p42), .p43(p43), .p44(p44),
        .w00(w00_ch1), .w01(w01_ch1), .w02(w02_ch1), .w03(w03_ch1), .w04(w04_ch1),
        .w10(w10_ch1), .w11(w11_ch1), .w12(w12_ch1), .w13(w13_ch1), .w14(w14_ch1),
        .w20(w20_ch1), .w21(w21_ch1), .w22(w22_ch1), .w23(w23_ch1), .w24(w24_ch1),
        .w30(w30_ch1), .w31(w31_ch1), .w32(w32_ch1), .w33(w33_ch1), .w34(w34_ch1),
        .w40(w40_ch1), .w41(w41_ch1), .w42(w42_ch1), .w43(w43_ch1), .w44(w44_ch1),
        .bias         (bias_ch1),
        .result       (result_ch1),
        .result_valid (valid1)
    );

    conv_pe_5x5 #(
        .PIXEL_WIDTH  (PIXEL_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .BIAS_WIDTH   (BIAS_WIDTH),
        .OUT_WIDTH    (OUT_WIDTH)
    ) u_pe2 (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (valid_in),
        .p00(p00), .p01(p01), .p02(p02), .p03(p03), .p04(p04),
        .p10(p10), .p11(p11), .p12(p12), .p13(p13), .p14(p14),
        .p20(p20), .p21(p21), .p22(p22), .p23(p23), .p24(p24),
        .p30(p30), .p31(p31), .p32(p32), .p33(p33), .p34(p34),
        .p40(p40), .p41(p41), .p42(p42), .p43(p43), .p44(p44),
        .w00(w00_ch2), .w01(w01_ch2), .w02(w02_ch2), .w03(w03_ch2), .w04(w04_ch2),
        .w10(w10_ch2), .w11(w11_ch2), .w12(w12_ch2), .w13(w13_ch2), .w14(w14_ch2),
        .w20(w20_ch2), .w21(w21_ch2), .w22(w22_ch2), .w23(w23_ch2), .w24(w24_ch2),
        .w30(w30_ch2), .w31(w31_ch2), .w32(w32_ch2), .w33(w33_ch2), .w34(w34_ch2),
        .w40(w40_ch2), .w41(w41_ch2), .w42(w42_ch2), .w43(w43_ch2), .w44(w44_ch2),
        .bias         (bias_ch2),
        .result       (result_ch2),
        .result_valid (valid2)
    );

    conv_pe_5x5 #(
        .PIXEL_WIDTH  (PIXEL_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .BIAS_WIDTH   (BIAS_WIDTH),
        .OUT_WIDTH    (OUT_WIDTH)
    ) u_pe3 (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (valid_in),
        .p00(p00), .p01(p01), .p02(p02), .p03(p03), .p04(p04),
        .p10(p10), .p11(p11), .p12(p12), .p13(p13), .p14(p14),
        .p20(p20), .p21(p21), .p22(p22), .p23(p23), .p24(p24),
        .p30(p30), .p31(p31), .p32(p32), .p33(p33), .p34(p34),
        .p40(p40), .p41(p41), .p42(p42), .p43(p43), .p44(p44),
        .w00(w00_ch3), .w01(w01_ch3), .w02(w02_ch3), .w03(w03_ch3), .w04(w04_ch3),
        .w10(w10_ch3), .w11(w11_ch3), .w12(w12_ch3), .w13(w13_ch3), .w14(w14_ch3),
        .w20(w20_ch3), .w21(w21_ch3), .w22(w22_ch3), .w23(w23_ch3), .w24(w24_ch3),
        .w30(w30_ch3), .w31(w31_ch3), .w32(w32_ch3), .w33(w33_ch3), .w34(w34_ch3),
        .w40(w40_ch3), .w41(w41_ch3), .w42(w42_ch3), .w43(w43_ch3), .w44(w44_ch3),
        .bias         (bias_ch3),
        .result       (result_ch3),
        .result_valid (valid3)
    );

    conv_pe_5x5 #(
        .PIXEL_WIDTH  (PIXEL_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .BIAS_WIDTH   (BIAS_WIDTH),
        .OUT_WIDTH    (OUT_WIDTH)
    ) u_pe4 (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (valid_in),
        .p00(p00), .p01(p01), .p02(p02), .p03(p03), .p04(p04),
        .p10(p10), .p11(p11), .p12(p12), .p13(p13), .p14(p14),
        .p20(p20), .p21(p21), .p22(p22), .p23(p23), .p24(p24),
        .p30(p30), .p31(p31), .p32(p32), .p33(p33), .p34(p34),
        .p40(p40), .p41(p41), .p42(p42), .p43(p43), .p44(p44),
        .w00(w00_ch4), .w01(w01_ch4), .w02(w02_ch4), .w03(w03_ch4), .w04(w04_ch4),
        .w10(w10_ch4), .w11(w11_ch4), .w12(w12_ch4), .w13(w13_ch4), .w14(w14_ch4),
        .w20(w20_ch4), .w21(w21_ch4), .w22(w22_ch4), .w23(w23_ch4), .w24(w24_ch4),
        .w30(w30_ch4), .w31(w31_ch4), .w32(w32_ch4), .w33(w33_ch4), .w34(w34_ch4),
        .w40(w40_ch4), .w41(w41_ch4), .w42(w42_ch4), .w43(w43_ch4), .w44(w44_ch4),
        .bias         (bias_ch4),
        .result       (result_ch4),
        .result_valid (valid4)
    );

    conv_pe_5x5 #(
        .PIXEL_WIDTH  (PIXEL_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .BIAS_WIDTH   (BIAS_WIDTH),
        .OUT_WIDTH    (OUT_WIDTH)
    ) u_pe5 (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (valid_in),
        .p00(p00), .p01(p01), .p02(p02), .p03(p03), .p04(p04),
        .p10(p10), .p11(p11), .p12(p12), .p13(p13), .p14(p14),
        .p20(p20), .p21(p21), .p22(p22), .p23(p23), .p24(p24),
        .p30(p30), .p31(p31), .p32(p32), .p33(p33), .p34(p34),
        .p40(p40), .p41(p41), .p42(p42), .p43(p43), .p44(p44),
        .w00(w00_ch5), .w01(w01_ch5), .w02(w02_ch5), .w03(w03_ch5), .w04(w04_ch5),
        .w10(w10_ch5), .w11(w11_ch5), .w12(w12_ch5), .w13(w13_ch5), .w14(w14_ch5),
        .w20(w20_ch5), .w21(w21_ch5), .w22(w22_ch5), .w23(w23_ch5), .w24(w24_ch5),
        .w30(w30_ch5), .w31(w31_ch5), .w32(w32_ch5), .w33(w33_ch5), .w34(w34_ch5),
        .w40(w40_ch5), .w41(w41_ch5), .w42(w42_ch5), .w43(w43_ch5), .w44(w44_ch5),
        .bias         (bias_ch5),
        .result       (result_ch5),
        .result_valid (valid5)
    );

endmodule
