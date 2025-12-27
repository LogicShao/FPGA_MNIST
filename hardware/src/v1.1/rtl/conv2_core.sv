`timescale 1ns/1ps

module conv2_core #(
    parameter IMG_WIDTH = 12,
    parameter DATA_WIDTH = 32,
    parameter WEIGHT_WIDTH = 8,
    parameter OUT_WIDTH = 32
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,

    input wire signed [DATA_WIDTH-1:0] in_ch0,
    input wire signed [DATA_WIDTH-1:0] in_ch1,
    input wire signed [DATA_WIDTH-1:0] in_ch2,
    input wire signed [DATA_WIDTH-1:0] in_ch3,
    input wire signed [DATA_WIDTH-1:0] in_ch4,
    input wire signed [DATA_WIDTH-1:0] in_ch5,

    input wire signed [WEIGHT_WIDTH-1:0] weights [0:15][0:149],
    input wire signed [31:0] biases [0:15],

    output reg signed [OUT_WIDTH-1:0] result_ch0,
    output reg signed [OUT_WIDTH-1:0] result_ch1,
    output reg signed [OUT_WIDTH-1:0] result_ch2,
    output reg signed [OUT_WIDTH-1:0] result_ch3,
    output reg signed [OUT_WIDTH-1:0] result_ch4,
    output reg signed [OUT_WIDTH-1:0] result_ch5,
    output reg signed [OUT_WIDTH-1:0] result_ch6,
    output reg signed [OUT_WIDTH-1:0] result_ch7,
    output reg signed [OUT_WIDTH-1:0] result_ch8,
    output reg signed [OUT_WIDTH-1:0] result_ch9,
    output reg signed [OUT_WIDTH-1:0] result_ch10,
    output reg signed [OUT_WIDTH-1:0] result_ch11,
    output reg signed [OUT_WIDTH-1:0] result_ch12,
    output reg signed [OUT_WIDTH-1:0] result_ch13,
    output reg signed [OUT_WIDTH-1:0] result_ch14,
    output reg signed [OUT_WIDTH-1:0] result_ch15,
    output reg result_valid
);

    // Window generators per input channel (12x12 stream)
    wire signed [DATA_WIDTH-1:0] w00_c0, w01_c0, w02_c0, w03_c0, w04_c0;
    wire signed [DATA_WIDTH-1:0] w10_c0, w11_c0, w12_c0, w13_c0, w14_c0;
    wire signed [DATA_WIDTH-1:0] w20_c0, w21_c0, w22_c0, w23_c0, w24_c0;
    wire signed [DATA_WIDTH-1:0] w30_c0, w31_c0, w32_c0, w33_c0, w34_c0;
    wire signed [DATA_WIDTH-1:0] w40_c0, w41_c0, w42_c0, w43_c0, w44_c0;
    wire window_valid_c0;

    wire signed [DATA_WIDTH-1:0] w00_c1, w01_c1, w02_c1, w03_c1, w04_c1;
    wire signed [DATA_WIDTH-1:0] w10_c1, w11_c1, w12_c1, w13_c1, w14_c1;
    wire signed [DATA_WIDTH-1:0] w20_c1, w21_c1, w22_c1, w23_c1, w24_c1;
    wire signed [DATA_WIDTH-1:0] w30_c1, w31_c1, w32_c1, w33_c1, w34_c1;
    wire signed [DATA_WIDTH-1:0] w40_c1, w41_c1, w42_c1, w43_c1, w44_c1;
    wire window_valid_c1;

    wire signed [DATA_WIDTH-1:0] w00_c2, w01_c2, w02_c2, w03_c2, w04_c2;
    wire signed [DATA_WIDTH-1:0] w10_c2, w11_c2, w12_c2, w13_c2, w14_c2;
    wire signed [DATA_WIDTH-1:0] w20_c2, w21_c2, w22_c2, w23_c2, w24_c2;
    wire signed [DATA_WIDTH-1:0] w30_c2, w31_c2, w32_c2, w33_c2, w34_c2;
    wire signed [DATA_WIDTH-1:0] w40_c2, w41_c2, w42_c2, w43_c2, w44_c2;
    wire window_valid_c2;

    wire signed [DATA_WIDTH-1:0] w00_c3, w01_c3, w02_c3, w03_c3, w04_c3;
    wire signed [DATA_WIDTH-1:0] w10_c3, w11_c3, w12_c3, w13_c3, w14_c3;
    wire signed [DATA_WIDTH-1:0] w20_c3, w21_c3, w22_c3, w23_c3, w24_c3;
    wire signed [DATA_WIDTH-1:0] w30_c3, w31_c3, w32_c3, w33_c3, w34_c3;
    wire signed [DATA_WIDTH-1:0] w40_c3, w41_c3, w42_c3, w43_c3, w44_c3;
    wire window_valid_c3;

    wire signed [DATA_WIDTH-1:0] w00_c4, w01_c4, w02_c4, w03_c4, w04_c4;
    wire signed [DATA_WIDTH-1:0] w10_c4, w11_c4, w12_c4, w13_c4, w14_c4;
    wire signed [DATA_WIDTH-1:0] w20_c4, w21_c4, w22_c4, w23_c4, w24_c4;
    wire signed [DATA_WIDTH-1:0] w30_c4, w31_c4, w32_c4, w33_c4, w34_c4;
    wire signed [DATA_WIDTH-1:0] w40_c4, w41_c4, w42_c4, w43_c4, w44_c4;
    wire window_valid_c4;

    wire signed [DATA_WIDTH-1:0] w00_c5, w01_c5, w02_c5, w03_c5, w04_c5;
    wire signed [DATA_WIDTH-1:0] w10_c5, w11_c5, w12_c5, w13_c5, w14_c5;
    wire signed [DATA_WIDTH-1:0] w20_c5, w21_c5, w22_c5, w23_c5, w24_c5;
    wire signed [DATA_WIDTH-1:0] w30_c5, w31_c5, w32_c5, w33_c5, w34_c5;
    wire signed [DATA_WIDTH-1:0] w40_c5, w41_c5, w42_c5, w43_c5, w44_c5;
    wire window_valid_c5;

    wire window_valid = window_valid_c0;

    layer1_window_gen #(
        .IMG_WIDTH  (IMG_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_win0 (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in),
        .din          (in_ch0),
        .w00(w00_c0), .w01(w01_c0), .w02(w02_c0), .w03(w03_c0), .w04(w04_c0),
        .w10(w10_c0), .w11(w11_c0), .w12(w12_c0), .w13(w13_c0), .w14(w14_c0),
        .w20(w20_c0), .w21(w21_c0), .w22(w22_c0), .w23(w23_c0), .w24(w24_c0),
        .w30(w30_c0), .w31(w31_c0), .w32(w32_c0), .w33(w33_c0), .w34(w34_c0),
        .w40(w40_c0), .w41(w41_c0), .w42(w42_c0), .w43(w43_c0), .w44(w44_c0),
        .window_valid (window_valid_c0)
    );

    layer1_window_gen #(
        .IMG_WIDTH  (IMG_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_win1 (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in),
        .din          (in_ch1),
        .w00(w00_c1), .w01(w01_c1), .w02(w02_c1), .w03(w03_c1), .w04(w04_c1),
        .w10(w10_c1), .w11(w11_c1), .w12(w12_c1), .w13(w13_c1), .w14(w14_c1),
        .w20(w20_c1), .w21(w21_c1), .w22(w22_c1), .w23(w23_c1), .w24(w24_c1),
        .w30(w30_c1), .w31(w31_c1), .w32(w32_c1), .w33(w33_c1), .w34(w34_c1),
        .w40(w40_c1), .w41(w41_c1), .w42(w42_c1), .w43(w43_c1), .w44(w44_c1),
        .window_valid (window_valid_c1)
    );

    layer1_window_gen #(
        .IMG_WIDTH  (IMG_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_win2 (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in),
        .din          (in_ch2),
        .w00(w00_c2), .w01(w01_c2), .w02(w02_c2), .w03(w03_c2), .w04(w04_c2),
        .w10(w10_c2), .w11(w11_c2), .w12(w12_c2), .w13(w13_c2), .w14(w14_c2),
        .w20(w20_c2), .w21(w21_c2), .w22(w22_c2), .w23(w23_c2), .w24(w24_c2),
        .w30(w30_c2), .w31(w31_c2), .w32(w32_c2), .w33(w33_c2), .w34(w34_c2),
        .w40(w40_c2), .w41(w41_c2), .w42(w42_c2), .w43(w43_c2), .w44(w44_c2),
        .window_valid (window_valid_c2)
    );

    layer1_window_gen #(
        .IMG_WIDTH  (IMG_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_win3 (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in),
        .din          (in_ch3),
        .w00(w00_c3), .w01(w01_c3), .w02(w02_c3), .w03(w03_c3), .w04(w04_c3),
        .w10(w10_c3), .w11(w11_c3), .w12(w12_c3), .w13(w13_c3), .w14(w14_c3),
        .w20(w20_c3), .w21(w21_c3), .w22(w22_c3), .w23(w23_c3), .w24(w24_c3),
        .w30(w30_c3), .w31(w31_c3), .w32(w32_c3), .w33(w33_c3), .w34(w34_c3),
        .w40(w40_c3), .w41(w41_c3), .w42(w42_c3), .w43(w43_c3), .w44(w44_c3),
        .window_valid (window_valid_c3)
    );

    layer1_window_gen #(
        .IMG_WIDTH  (IMG_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_win4 (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in),
        .din          (in_ch4),
        .w00(w00_c4), .w01(w01_c4), .w02(w02_c4), .w03(w03_c4), .w04(w04_c4),
        .w10(w10_c4), .w11(w11_c4), .w12(w12_c4), .w13(w13_c4), .w14(w14_c4),
        .w20(w20_c4), .w21(w21_c4), .w22(w22_c4), .w23(w23_c4), .w24(w24_c4),
        .w30(w30_c4), .w31(w31_c4), .w32(w32_c4), .w33(w33_c4), .w34(w34_c4),
        .w40(w40_c4), .w41(w41_c4), .w42(w42_c4), .w43(w43_c4), .w44(w44_c4),
        .window_valid (window_valid_c4)
    );

    layer1_window_gen #(
        .IMG_WIDTH  (IMG_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_win5 (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in),
        .din          (in_ch5),
        .w00(w00_c5), .w01(w01_c5), .w02(w02_c5), .w03(w03_c5), .w04(w04_c5),
        .w10(w10_c5), .w11(w11_c5), .w12(w12_c5), .w13(w13_c5), .w14(w14_c5),
        .w20(w20_c5), .w21(w21_c5), .w22(w22_c5), .w23(w23_c5), .w24(w24_c5),
        .w30(w30_c5), .w31(w31_c5), .w32(w32_c5), .w33(w33_c5), .w34(w34_c5),
        .w40(w40_c5), .w41(w41_c5), .w42(w42_c5), .w43(w43_c5), .w44(w44_c5),
        .window_valid (window_valid_c5)
    );

    wire signed [DATA_WIDTH-1:0] pixels [0:149];

    assign pixels[0] = w00_c0; assign pixels[1] = w01_c0; assign pixels[2] = w02_c0; assign pixels[3] = w03_c0; assign pixels[4] = w04_c0;
    assign pixels[5] = w10_c0; assign pixels[6] = w11_c0; assign pixels[7] = w12_c0; assign pixels[8] = w13_c0; assign pixels[9] = w14_c0;
    assign pixels[10] = w20_c0; assign pixels[11] = w21_c0; assign pixels[12] = w22_c0; assign pixels[13] = w23_c0; assign pixels[14] = w24_c0;
    assign pixels[15] = w30_c0; assign pixels[16] = w31_c0; assign pixels[17] = w32_c0; assign pixels[18] = w33_c0; assign pixels[19] = w34_c0;
    assign pixels[20] = w40_c0; assign pixels[21] = w41_c0; assign pixels[22] = w42_c0; assign pixels[23] = w43_c0; assign pixels[24] = w44_c0;

    assign pixels[25] = w00_c1; assign pixels[26] = w01_c1; assign pixels[27] = w02_c1; assign pixels[28] = w03_c1; assign pixels[29] = w04_c1;
    assign pixels[30] = w10_c1; assign pixels[31] = w11_c1; assign pixels[32] = w12_c1; assign pixels[33] = w13_c1; assign pixels[34] = w14_c1;
    assign pixels[35] = w20_c1; assign pixels[36] = w21_c1; assign pixels[37] = w22_c1; assign pixels[38] = w23_c1; assign pixels[39] = w24_c1;
    assign pixels[40] = w30_c1; assign pixels[41] = w31_c1; assign pixels[42] = w32_c1; assign pixels[43] = w33_c1; assign pixels[44] = w34_c1;
    assign pixels[45] = w40_c1; assign pixels[46] = w41_c1; assign pixels[47] = w42_c1; assign pixels[48] = w43_c1; assign pixels[49] = w44_c1;

    assign pixels[50] = w00_c2; assign pixels[51] = w01_c2; assign pixels[52] = w02_c2; assign pixels[53] = w03_c2; assign pixels[54] = w04_c2;
    assign pixels[55] = w10_c2; assign pixels[56] = w11_c2; assign pixels[57] = w12_c2; assign pixels[58] = w13_c2; assign pixels[59] = w14_c2;
    assign pixels[60] = w20_c2; assign pixels[61] = w21_c2; assign pixels[62] = w22_c2; assign pixels[63] = w23_c2; assign pixels[64] = w24_c2;
    assign pixels[65] = w30_c2; assign pixels[66] = w31_c2; assign pixels[67] = w32_c2; assign pixels[68] = w33_c2; assign pixels[69] = w34_c2;
    assign pixels[70] = w40_c2; assign pixels[71] = w41_c2; assign pixels[72] = w42_c2; assign pixels[73] = w43_c2; assign pixels[74] = w44_c2;

    assign pixels[75] = w00_c3; assign pixels[76] = w01_c3; assign pixels[77] = w02_c3; assign pixels[78] = w03_c3; assign pixels[79] = w04_c3;
    assign pixels[80] = w10_c3; assign pixels[81] = w11_c3; assign pixels[82] = w12_c3; assign pixels[83] = w13_c3; assign pixels[84] = w14_c3;
    assign pixels[85] = w20_c3; assign pixels[86] = w21_c3; assign pixels[87] = w22_c3; assign pixels[88] = w23_c3; assign pixels[89] = w24_c3;
    assign pixels[90] = w30_c3; assign pixels[91] = w31_c3; assign pixels[92] = w32_c3; assign pixels[93] = w33_c3; assign pixels[94] = w34_c3;
    assign pixels[95] = w40_c3; assign pixels[96] = w41_c3; assign pixels[97] = w42_c3; assign pixels[98] = w43_c3; assign pixels[99] = w44_c3;

    assign pixels[100] = w00_c4; assign pixels[101] = w01_c4; assign pixels[102] = w02_c4; assign pixels[103] = w03_c4; assign pixels[104] = w04_c4;
    assign pixels[105] = w10_c4; assign pixels[106] = w11_c4; assign pixels[107] = w12_c4; assign pixels[108] = w13_c4; assign pixels[109] = w14_c4;
    assign pixels[110] = w20_c4; assign pixels[111] = w21_c4; assign pixels[112] = w22_c4; assign pixels[113] = w23_c4; assign pixels[114] = w24_c4;
    assign pixels[115] = w30_c4; assign pixels[116] = w31_c4; assign pixels[117] = w32_c4; assign pixels[118] = w33_c4; assign pixels[119] = w34_c4;
    assign pixels[120] = w40_c4; assign pixels[121] = w41_c4; assign pixels[122] = w42_c4; assign pixels[123] = w43_c4; assign pixels[124] = w44_c4;

    assign pixels[125] = w00_c5; assign pixels[126] = w01_c5; assign pixels[127] = w02_c5; assign pixels[128] = w03_c5; assign pixels[129] = w04_c5;
    assign pixels[130] = w10_c5; assign pixels[131] = w11_c5; assign pixels[132] = w12_c5; assign pixels[133] = w13_c5; assign pixels[134] = w14_c5;
    assign pixels[135] = w20_c5; assign pixels[136] = w21_c5; assign pixels[137] = w22_c5; assign pixels[138] = w23_c5; assign pixels[139] = w24_c5;
    assign pixels[140] = w30_c5; assign pixels[141] = w31_c5; assign pixels[142] = w32_c5; assign pixels[143] = w33_c5; assign pixels[144] = w34_c5;
    assign pixels[145] = w40_c5; assign pixels[146] = w41_c5; assign pixels[147] = w42_c5; assign pixels[148] = w43_c5; assign pixels[149] = w44_c5;

    integer oc;
    integer i;
    reg signed [63:0] acc;

`ifdef FAST_SIM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_ch0 <= 0; result_ch1 <= 0; result_ch2 <= 0; result_ch3 <= 0;
            result_ch4 <= 0; result_ch5 <= 0; result_ch6 <= 0; result_ch7 <= 0;
            result_ch8 <= 0; result_ch9 <= 0; result_ch10 <= 0; result_ch11 <= 0;
            result_ch12 <= 0; result_ch13 <= 0; result_ch14 <= 0; result_ch15 <= 0;
            result_valid <= 1'b0;
        end else begin
            result_valid <= window_valid;
            if (window_valid) begin
                result_ch0 <= w00_c0;
                result_ch1 <= w00_c1;
                result_ch2 <= w00_c2;
                result_ch3 <= w00_c3;
                result_ch4 <= w00_c4;
                result_ch5 <= w00_c5;
                result_ch6 <= w01_c0;
                result_ch7 <= w01_c1;
                result_ch8 <= w01_c2;
                result_ch9 <= w01_c3;
                result_ch10 <= w01_c4;
                result_ch11 <= w01_c5;
                result_ch12 <= w02_c0;
                result_ch13 <= w02_c1;
                result_ch14 <= w02_c2;
                result_ch15 <= w02_c3;
            end
        end
    end
`else
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result_ch0 <= 0; result_ch1 <= 0; result_ch2 <= 0; result_ch3 <= 0;
            result_ch4 <= 0; result_ch5 <= 0; result_ch6 <= 0; result_ch7 <= 0;
            result_ch8 <= 0; result_ch9 <= 0; result_ch10 <= 0; result_ch11 <= 0;
            result_ch12 <= 0; result_ch13 <= 0; result_ch14 <= 0; result_ch15 <= 0;
            result_valid <= 1'b0;
        end else begin
            result_valid <= window_valid;
            if (window_valid) begin
                for (oc = 0; oc < 16; oc = oc + 1) begin
                    acc = 0;
                    for (i = 0; i < 150; i = i + 1) begin
                        acc = acc + $signed(pixels[i]) * $signed(weights[oc][i]);
                    end
                    acc = acc + $signed(biases[oc]);
                    if (acc[63]) begin
                        case (oc)
                            0: result_ch0 <= 0;
                            1: result_ch1 <= 0;
                            2: result_ch2 <= 0;
                            3: result_ch3 <= 0;
                            4: result_ch4 <= 0;
                            5: result_ch5 <= 0;
                            6: result_ch6 <= 0;
                            7: result_ch7 <= 0;
                            8: result_ch8 <= 0;
                            9: result_ch9 <= 0;
                            10: result_ch10 <= 0;
                            11: result_ch11 <= 0;
                            12: result_ch12 <= 0;
                            13: result_ch13 <= 0;
                            14: result_ch14 <= 0;
                            default: result_ch15 <= 0;
                        endcase
                    end else begin
                        case (oc)
                            0: result_ch0 <= acc[31:0];
                            1: result_ch1 <= acc[31:0];
                            2: result_ch2 <= acc[31:0];
                            3: result_ch3 <= acc[31:0];
                            4: result_ch4 <= acc[31:0];
                            5: result_ch5 <= acc[31:0];
                            6: result_ch6 <= acc[31:0];
                            7: result_ch7 <= acc[31:0];
                            8: result_ch8 <= acc[31:0];
                            9: result_ch9 <= acc[31:0];
                            10: result_ch10 <= acc[31:0];
                            11: result_ch11 <= acc[31:0];
                            12: result_ch12 <= acc[31:0];
                            13: result_ch13 <= acc[31:0];
                            14: result_ch14 <= acc[31:0];
                            default: result_ch15 <= acc[31:0];
                        endcase
                    end
                end
            end
        end
    end
`endif

endmodule

