`timescale 1ns/1ps
`include "quant_params.vh"

module layer2_block(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [31:0] in_ch0,
    input wire [31:0] in_ch1,
    input wire [31:0] in_ch2,
    input wire [31:0] in_ch3,
    input wire [31:0] in_ch4,
    input wire [31:0] in_ch5,
    output wire [31:0] out_ch0,
    output wire [31:0] out_ch1,
    output wire [31:0] out_ch2,
    output wire [31:0] out_ch3,
    output wire [31:0] out_ch4,
    output wire [31:0] out_ch5,
    output wire [31:0] out_ch6,
    output wire [31:0] out_ch7,
    output wire [31:0] out_ch8,
    output wire [31:0] out_ch9,
    output wire [31:0] out_ch10,
    output wire [31:0] out_ch11,
    output wire [31:0] out_ch12,
    output wire [31:0] out_ch13,
    output wire [31:0] out_ch14,
    output wire [31:0] out_ch15,
    output wire out_valid
);

    // ==========================================
    // 1. Weight ROM signals and loader
    // ==========================================
    wire signed [7:0] conv2_weight_q;
    wire signed [31:0] conv2_bias_q;

    reg [11:0] conv2_weight_addr;
    reg [3:0] conv2_bias_addr;

    reg signed [7:0] conv2_weights [0:15][0:149];
    reg signed [31:0] conv2_biases [0:15];

    localparam LOAD_WEIGHTS = 2'd0;
    localparam LOAD_BIAS    = 2'd1;
    localparam LOAD_DONE    = 2'd2;

    reg [1:0] load_state;
    reg [11:0] load_idx;
    reg        load_capture;
    reg [11:0] load_idx_d1;
    reg        load_capture_d1;

    reg [4:0] bias_idx;
    reg       bias_capture;
    reg [4:0] bias_idx_d1;
    reg       bias_capture_d1;

    wire weights_ready = (load_state == LOAD_DONE);
    wire conv_valid_in = valid_in & weights_ready;

    integer i;

    rom_CONV2_WEIGHTS #(
        .ADDR_WIDTH(12),
        .DATA_WIDTH(8),
        .DEPTH(2400),
        .MEM_FILE("rtl/weights/CONV2_WEIGHTS.mem")
    ) u_conv2_wrom (
        .clk  (clk),
        .addr (conv2_weight_addr),
        .q    (conv2_weight_q)
    );

    rom_CONV2_BIASES_INT32 #(
        .ADDR_WIDTH(4),
        .DATA_WIDTH(32),
        .DEPTH(16),
        .MEM_FILE("rtl/weights/CONV2_BIASES_INT32.mem")
    ) u_conv2_brom (
        .clk  (clk),
        .addr (conv2_bias_addr),
        .q    (conv2_bias_q)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv2_weight_addr <= 12'd0;
            conv2_bias_addr <= 4'd0;
            load_state <= LOAD_WEIGHTS;
            load_idx <= 12'd0;
            load_capture <= 1'b0;
            load_idx_d1 <= 12'd0;
            load_capture_d1 <= 1'b0;
            bias_idx <= 5'd0;
            bias_capture <= 1'b0;
            bias_idx_d1 <= 5'd0;
            bias_capture_d1 <= 1'b0;
        end else begin
            case (load_state)
                LOAD_WEIGHTS: begin
                    load_idx_d1 <= load_idx;
                    load_capture_d1 <= load_capture;

                    if (load_capture_d1 && load_idx_d1 != 0 && load_idx_d1 <= 12'd2400) begin
                        conv2_weights[(load_idx_d1 - 1) / 150][(load_idx_d1 - 1) % 150] <= conv2_weight_q;
                    end
                    if (load_idx < 2400) begin
                        conv2_weight_addr <= load_idx;
                        load_idx <= load_idx + 1'b1;
                        load_capture <= 1'b1;
                    end else begin
                        load_capture <= 1'b0;
                        if (!load_capture_d1) begin
                            load_state <= LOAD_BIAS;
                            bias_idx <= 5'd0;
                            bias_capture <= 1'b0;
                            bias_idx_d1 <= 5'd0;
                            bias_capture_d1 <= 1'b0;
                        end
                    end
                end
                LOAD_BIAS: begin
                    bias_idx_d1 <= bias_idx;
                    bias_capture_d1 <= bias_capture;

                    if (bias_capture_d1 && bias_idx_d1 != 0 && bias_idx_d1 <= 5'd16) begin
                        conv2_biases[bias_idx_d1 - 1] <= conv2_bias_q;
                    end
                    if (bias_idx < 5'd16) begin
                        conv2_bias_addr <= bias_idx[3:0];
                        bias_idx <= bias_idx + 1'b1;
                        bias_capture <= 1'b1;
                    end else begin
                        bias_capture <= 1'b0;
                        if (!bias_capture_d1)
                            load_state <= LOAD_DONE;
                    end
                end
                default: begin
                end
            endcase
        end
    end

    // ==========================================
    // 2. CONV2 core (16 channels)
    // ==========================================
    wire [31:0] conv2_ch0;
    wire [31:0] conv2_ch1;
    wire [31:0] conv2_ch2;
    wire [31:0] conv2_ch3;
    wire [31:0] conv2_ch4;
    wire [31:0] conv2_ch5;
    wire [31:0] conv2_ch6;
    wire [31:0] conv2_ch7;
    wire [31:0] conv2_ch8;
    wire [31:0] conv2_ch9;
    wire [31:0] conv2_ch10;
    wire [31:0] conv2_ch11;
    wire [31:0] conv2_ch12;
    wire [31:0] conv2_ch13;
    wire [31:0] conv2_ch14;
    wire [31:0] conv2_ch15;
    wire conv2_valid;
    reg [31:0] conv2_ch0_r;
    reg [31:0] conv2_ch1_r;
    reg [31:0] conv2_ch2_r;
    reg [31:0] conv2_ch3_r;
    reg [31:0] conv2_ch4_r;
    reg [31:0] conv2_ch5_r;
    reg [31:0] conv2_ch6_r;
    reg [31:0] conv2_ch7_r;
    reg [31:0] conv2_ch8_r;
    reg [31:0] conv2_ch9_r;
    reg [31:0] conv2_ch10_r;
    reg [31:0] conv2_ch11_r;
    reg [31:0] conv2_ch12_r;
    reg [31:0] conv2_ch13_r;
    reg [31:0] conv2_ch14_r;
    reg [31:0] conv2_ch15_r;
    reg conv2_valid_r;

    conv2_core #(
        .IMG_WIDTH(12),
        .DATA_WIDTH(32),
        .WEIGHT_WIDTH(8),
        .OUT_WIDTH(32)
    ) u_conv2 (
        .clk        (clk),
        .rst_n      (rst_n),
        .valid_in   (conv_valid_in),
        .in_ch0     (in_ch0),
        .in_ch1     (in_ch1),
        .in_ch2     (in_ch2),
        .in_ch3     (in_ch3),
        .in_ch4     (in_ch4),
        .in_ch5     (in_ch5),
        .weights    (conv2_weights),
        .biases     (conv2_biases),
        .result_ch0 (conv2_ch0),
        .result_ch1 (conv2_ch1),
        .result_ch2 (conv2_ch2),
        .result_ch3 (conv2_ch3),
        .result_ch4 (conv2_ch4),
        .result_ch5 (conv2_ch5),
        .result_ch6 (conv2_ch6),
        .result_ch7 (conv2_ch7),
        .result_ch8 (conv2_ch8),
        .result_ch9 (conv2_ch9),
        .result_ch10(conv2_ch10),
        .result_ch11(conv2_ch11),
        .result_ch12(conv2_ch12),
        .result_ch13(conv2_ch13),
        .result_ch14(conv2_ch14),
        .result_ch15(conv2_ch15),
        .result_valid(conv2_valid)
    );

    // ==========================================
    // 3. Max pool 2x2 (16 channels)
    // ==========================================

    wire signed [7:0] conv2_q0;
    wire signed [7:0] conv2_q1;
    wire signed [7:0] conv2_q2;
    wire signed [7:0] conv2_q3;
    wire signed [7:0] conv2_q4;
    wire signed [7:0] conv2_q5;
    wire signed [7:0] conv2_q6;
    wire signed [7:0] conv2_q7;
    wire signed [7:0] conv2_q8;
    wire signed [7:0] conv2_q9;
    wire signed [7:0] conv2_q10;
    wire signed [7:0] conv2_q11;
    wire signed [7:0] conv2_q12;
    wire signed [7:0] conv2_q13;
    wire signed [7:0] conv2_q14;
    wire signed [7:0] conv2_q15;

    wire signed [31:0] conv2_q0_32 = {{24{conv2_q0[7]}}, conv2_q0};
    wire signed [31:0] conv2_q1_32 = {{24{conv2_q1[7]}}, conv2_q1};
    wire signed [31:0] conv2_q2_32 = {{24{conv2_q2[7]}}, conv2_q2};
    wire signed [31:0] conv2_q3_32 = {{24{conv2_q3[7]}}, conv2_q3};
    wire signed [31:0] conv2_q4_32 = {{24{conv2_q4[7]}}, conv2_q4};
    wire signed [31:0] conv2_q5_32 = {{24{conv2_q5[7]}}, conv2_q5};
    wire signed [31:0] conv2_q6_32 = {{24{conv2_q6[7]}}, conv2_q6};
    wire signed [31:0] conv2_q7_32 = {{24{conv2_q7[7]}}, conv2_q7};
    wire signed [31:0] conv2_q8_32 = {{24{conv2_q8[7]}}, conv2_q8};
    wire signed [31:0] conv2_q9_32 = {{24{conv2_q9[7]}}, conv2_q9};
    wire signed [31:0] conv2_q10_32 = {{24{conv2_q10[7]}}, conv2_q10};
    wire signed [31:0] conv2_q11_32 = {{24{conv2_q11[7]}}, conv2_q11};
    wire signed [31:0] conv2_q12_32 = {{24{conv2_q12[7]}}, conv2_q12};
    wire signed [31:0] conv2_q13_32 = {{24{conv2_q13[7]}}, conv2_q13};
    wire signed [31:0] conv2_q14_32 = {{24{conv2_q14[7]}}, conv2_q14};
    wire signed [31:0] conv2_q15_32 = {{24{conv2_q15[7]}}, conv2_q15};

    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch0 (.acc_in(conv2_ch0_r), .q_out(conv2_q0));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch1 (.acc_in(conv2_ch1_r), .q_out(conv2_q1));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch2 (.acc_in(conv2_ch2_r), .q_out(conv2_q2));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch3 (.acc_in(conv2_ch3_r), .q_out(conv2_q3));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch4 (.acc_in(conv2_ch4_r), .q_out(conv2_q4));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch5 (.acc_in(conv2_ch5_r), .q_out(conv2_q5));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch6 (.acc_in(conv2_ch6_r), .q_out(conv2_q6));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch7 (.acc_in(conv2_ch7_r), .q_out(conv2_q7));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch8 (.acc_in(conv2_ch8_r), .q_out(conv2_q8));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch9 (.acc_in(conv2_ch9_r), .q_out(conv2_q9));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch10 (.acc_in(conv2_ch10_r), .q_out(conv2_q10));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch11 (.acc_in(conv2_ch11_r), .q_out(conv2_q11));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch12 (.acc_in(conv2_ch12_r), .q_out(conv2_q12));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch13 (.acc_in(conv2_ch13_r), .q_out(conv2_q13));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch14 (.acc_in(conv2_ch14_r), .q_out(conv2_q14));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2_ch15 (.acc_in(conv2_ch15_r), .q_out(conv2_q15));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv2_ch0_r <= 0;
            conv2_ch1_r <= 0;
            conv2_ch2_r <= 0;
            conv2_ch3_r <= 0;
            conv2_ch4_r <= 0;
            conv2_ch5_r <= 0;
            conv2_ch6_r <= 0;
            conv2_ch7_r <= 0;
            conv2_ch8_r <= 0;
            conv2_ch9_r <= 0;
            conv2_ch10_r <= 0;
            conv2_ch11_r <= 0;
            conv2_ch12_r <= 0;
            conv2_ch13_r <= 0;
            conv2_ch14_r <= 0;
            conv2_ch15_r <= 0;
            conv2_valid_r <= 1'b0;
        end else begin
            conv2_ch0_r <= conv2_ch0;
            conv2_ch1_r <= conv2_ch1;
            conv2_ch2_r <= conv2_ch2;
            conv2_ch3_r <= conv2_ch3;
            conv2_ch4_r <= conv2_ch4;
            conv2_ch5_r <= conv2_ch5;
            conv2_ch6_r <= conv2_ch6;
            conv2_ch7_r <= conv2_ch7;
            conv2_ch8_r <= conv2_ch8;
            conv2_ch9_r <= conv2_ch9;
            conv2_ch10_r <= conv2_ch10;
            conv2_ch11_r <= conv2_ch11;
            conv2_ch12_r <= conv2_ch12;
            conv2_ch13_r <= conv2_ch13;
            conv2_ch14_r <= conv2_ch14;
            conv2_ch15_r <= conv2_ch15;
            conv2_valid_r <= conv2_valid;
        end
    end

    max_pool_2x2_16ch #(
        .IN_WIDTH(8),
        .IN_HEIGHT(8)
    ) u_pool2 (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (conv2_valid_r),
        .in_ch0   (conv2_q0_32),
        .in_ch1   (conv2_q1_32),
        .in_ch2   (conv2_q2_32),
        .in_ch3   (conv2_q3_32),
        .in_ch4   (conv2_q4_32),
        .in_ch5   (conv2_q5_32),
        .in_ch6   (conv2_q6_32),
        .in_ch7   (conv2_q7_32),
        .in_ch8   (conv2_q8_32),
        .in_ch9   (conv2_q9_32),
        .in_ch10  (conv2_q10_32),
        .in_ch11  (conv2_q11_32),
        .in_ch12  (conv2_q12_32),
        .in_ch13  (conv2_q13_32),
        .in_ch14  (conv2_q14_32),
        .in_ch15  (conv2_q15_32),
        .out_ch0  (out_ch0),
        .out_ch1  (out_ch1),
        .out_ch2  (out_ch2),
        .out_ch3  (out_ch3),
        .out_ch4  (out_ch4),
        .out_ch5  (out_ch5),
        .out_ch6  (out_ch6),
        .out_ch7  (out_ch7),
        .out_ch8  (out_ch8),
        .out_ch9  (out_ch9),
        .out_ch10 (out_ch10),
        .out_ch11 (out_ch11),
        .out_ch12 (out_ch12),
        .out_ch13 (out_ch13),
        .out_ch14 (out_ch14),
        .out_ch15 (out_ch15),
        .out_valid(out_valid)
    );

endmodule
