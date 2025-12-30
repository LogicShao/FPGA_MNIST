`timescale 1ns/1ps

module mnist_network_core(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [7:0] pixel_in,
    output wire [31:0] result,
    output wire result_valid
);

    wire signed [7:0] l1_ch0;
    wire signed [7:0] l1_ch1;
    wire signed [7:0] l1_ch2;
    wire signed [7:0] l1_ch3;
    wire signed [7:0] l1_ch4;
    wire signed [7:0] l1_ch5;
    wire l1_valid;

    wire signed [7:0] l2_ch0;
    wire signed [7:0] l2_ch1;
    wire signed [7:0] l2_ch2;
    wire signed [7:0] l2_ch3;
    wire signed [7:0] l2_ch4;
    wire signed [7:0] l2_ch5;
    wire signed [7:0] l2_ch6;
    wire signed [7:0] l2_ch7;
    wire signed [7:0] l2_ch8;
    wire signed [7:0] l2_ch9;
    wire signed [7:0] l2_ch10;
    wire signed [7:0] l2_ch11;
    wire signed [7:0] l2_ch12;
    wire signed [7:0] l2_ch13;
    wire signed [7:0] l2_ch14;
    wire signed [7:0] l2_ch15;
    wire l2_valid;

    wire signed [7:0] l3_data;
    wire l3_valid;

    wire [31:0] l4_data;
    wire l4_valid;

    layer1_block u_layer1 (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in),
        .pixel_in     (pixel_in),
        .result_ch0   (l1_ch0),
        .result_ch1   (l1_ch1),
        .result_ch2   (l1_ch2),
        .result_ch3   (l1_ch3),
        .result_ch4   (l1_ch4),
        .result_ch5   (l1_ch5),
        .result_valid (l1_valid)
    );

    layer2_block u_layer2 (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (l1_valid),
        .in_ch0   (l1_ch0),
        .in_ch1   (l1_ch1),
        .in_ch2   (l1_ch2),
        .in_ch3   (l1_ch3),
        .in_ch4   (l1_ch4),
        .in_ch5   (l1_ch5),
        .out_ch0  (l2_ch0),
        .out_ch1  (l2_ch1),
        .out_ch2  (l2_ch2),
        .out_ch3  (l2_ch3),
        .out_ch4  (l2_ch4),
        .out_ch5  (l2_ch5),
        .out_ch6  (l2_ch6),
        .out_ch7  (l2_ch7),
        .out_ch8  (l2_ch8),
        .out_ch9  (l2_ch9),
        .out_ch10 (l2_ch10),
        .out_ch11 (l2_ch11),
        .out_ch12 (l2_ch12),
        .out_ch13 (l2_ch13),
        .out_ch14 (l2_ch14),
        .out_ch15 (l2_ch15),
        .out_valid(l2_valid)
    );

    layer3_fc1 u_layer3 (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (l2_valid),
        .in_ch0   (l2_ch0),
        .in_ch1   (l2_ch1),
        .in_ch2   (l2_ch2),
        .in_ch3   (l2_ch3),
        .in_ch4   (l2_ch4),
        .in_ch5   (l2_ch5),
        .in_ch6   (l2_ch6),
        .in_ch7   (l2_ch7),
        .in_ch8   (l2_ch8),
        .in_ch9   (l2_ch9),
        .in_ch10  (l2_ch10),
        .in_ch11  (l2_ch11),
        .in_ch12  (l2_ch12),
        .in_ch13  (l2_ch13),
        .in_ch14  (l2_ch14),
        .in_ch15  (l2_ch15),
        .out_data (l3_data),
        .out_valid(l3_valid)
    );

    layer4_fc2 u_layer4 (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (l3_valid),
        .in_data  (l3_data),
        .out_data (l4_data),
        .out_valid(l4_valid)
    );

    assign result = l4_data;
    assign result_valid = l4_valid;

endmodule
