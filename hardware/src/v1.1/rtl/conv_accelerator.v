module conv_accelerator #(
    parameter IMG_WIDTH = 28,
    parameter DATA_WIDTH = 8,
    parameter OUT_WIDTH = 32,
    parameter signed [DATA_WIDTH-1:0] K00 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K01 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K02 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K03 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K04 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K10 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K11 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K12 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K13 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K14 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K20 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K21 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K22 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K23 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K24 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K30 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K31 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K32 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K33 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K34 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K40 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K41 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K42 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K43 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] K44 = 8'sd0,
    parameter signed [DATA_WIDTH-1:0] BIAS = 8'sd0
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [DATA_WIDTH-1:0] pixel_in,

    output wire signed [OUT_WIDTH-1:0] result_ch0,
    output wire result_valid
);

    wire [DATA_WIDTH-1:0] w00, w01, w02, w03, w04;
    wire [DATA_WIDTH-1:0] w10, w11, w12, w13, w14;
    wire [DATA_WIDTH-1:0] w20, w21, w22, w23, w24;
    wire [DATA_WIDTH-1:0] w30, w31, w32, w33, w34;
    wire [DATA_WIDTH-1:0] w40, w41, w42, w43, w44;
    wire window_valid;

    layer1_window_gen #(
        .IMG_WIDTH  (IMG_WIDTH),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_window_gen (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in),
        .din          (pixel_in),
        .w00          (w00), .w01(w01), .w02(w02), .w03(w03), .w04(w04),
        .w10          (w10), .w11(w11), .w12(w12), .w13(w13), .w14(w14),
        .w20          (w20), .w21(w21), .w22(w22), .w23(w23), .w24(w24),
        .w30          (w30), .w31(w31), .w32(w32), .w33(w33), .w34(w34),
        .w40          (w40), .w41(w41), .w42(w42), .w43(w43), .w44(w44),
        .window_valid (window_valid)
    );

    wire signed [DATA_WIDTH-1:0] p00 = $signed(w00);
    wire signed [DATA_WIDTH-1:0] p01 = $signed(w01);
    wire signed [DATA_WIDTH-1:0] p02 = $signed(w02);
    wire signed [DATA_WIDTH-1:0] p03 = $signed(w03);
    wire signed [DATA_WIDTH-1:0] p04 = $signed(w04);
    wire signed [DATA_WIDTH-1:0] p10 = $signed(w10);
    wire signed [DATA_WIDTH-1:0] p11 = $signed(w11);
    wire signed [DATA_WIDTH-1:0] p12 = $signed(w12);
    wire signed [DATA_WIDTH-1:0] p13 = $signed(w13);
    wire signed [DATA_WIDTH-1:0] p14 = $signed(w14);
    wire signed [DATA_WIDTH-1:0] p20 = $signed(w20);
    wire signed [DATA_WIDTH-1:0] p21 = $signed(w21);
    wire signed [DATA_WIDTH-1:0] p22 = $signed(w22);
    wire signed [DATA_WIDTH-1:0] p23 = $signed(w23);
    wire signed [DATA_WIDTH-1:0] p24 = $signed(w24);
    wire signed [DATA_WIDTH-1:0] p30 = $signed(w30);
    wire signed [DATA_WIDTH-1:0] p31 = $signed(w31);
    wire signed [DATA_WIDTH-1:0] p32 = $signed(w32);
    wire signed [DATA_WIDTH-1:0] p33 = $signed(w33);
    wire signed [DATA_WIDTH-1:0] p34 = $signed(w34);
    wire signed [DATA_WIDTH-1:0] p40 = $signed(w40);
    wire signed [DATA_WIDTH-1:0] p41 = $signed(w41);
    wire signed [DATA_WIDTH-1:0] p42 = $signed(w42);
    wire signed [DATA_WIDTH-1:0] p43 = $signed(w43);
    wire signed [DATA_WIDTH-1:0] p44 = $signed(w44);

    conv_pe_5x5 #(
        .DATA_WIDTH (DATA_WIDTH),
        .OUT_WIDTH  (OUT_WIDTH)
    ) u_pe (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (window_valid),
        .p00        (p00), .p01(p01), .p02(p02), .p03(p03), .p04(p04),
        .p10        (p10), .p11(p11), .p12(p12), .p13(p13), .p14(p14),
        .p20        (p20), .p21(p21), .p22(p22), .p23(p23), .p24(p24),
        .p30        (p30), .p31(p31), .p32(p32), .p33(p33), .p34(p34),
        .p40        (p40), .p41(p41), .p42(p42), .p43(p43), .p44(p44),
        .w00        (K00), .w01(K01), .w02(K02), .w03(K03), .w04(K04),
        .w10        (K10), .w11(K11), .w12(K12), .w13(K13), .w14(K14),
        .w20        (K20), .w21(K21), .w22(K22), .w23(K23), .w24(K24),
        .w30        (K30), .w31(K31), .w32(K32), .w33(K33), .w34(K34),
        .w40        (K40), .w41(K41), .w42(K42), .w43(K43), .w44(K44),
        .bias       (BIAS),
        .result     (result_ch0),
        .result_valid (result_valid)
    );

endmodule
