`timescale 1ns/1ps
`include "quant_params.vh"

module layer1_block(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [7:0] pixel_in,

    output wire [31:0] result_ch0,
    output wire [31:0] result_ch1,
    output wire [31:0] result_ch2,
    output wire [31:0] result_ch3,
    output wire [31:0] result_ch4,
    output wire [31:0] result_ch5,
    output wire result_valid
);

    // ==========================================
    // 1. Weight ROM signals and loader
    // ==========================================
    wire signed [7:0] conv1_weight_q;
    wire signed [31:0] conv1_bias_q;

    reg [7:0] conv1_weight_addr;
    reg [2:0] conv1_bias_addr;

    reg signed [7:0] conv1_weights_ch0 [0:24];
    reg signed [7:0] conv1_weights_ch1 [0:24];
    reg signed [7:0] conv1_weights_ch2 [0:24];
    reg signed [7:0] conv1_weights_ch3 [0:24];
    reg signed [7:0] conv1_weights_ch4 [0:24];
    reg signed [7:0] conv1_weights_ch5 [0:24];

    reg signed [31:0] conv1_bias_ch0;
    reg signed [31:0] conv1_bias_ch1;
    reg signed [31:0] conv1_bias_ch2;
    reg signed [31:0] conv1_bias_ch3;
    reg signed [31:0] conv1_bias_ch4;
    reg signed [31:0] conv1_bias_ch5;

    localparam LOAD_WEIGHTS = 2'd0;
    localparam LOAD_BIAS    = 2'd1;
    localparam LOAD_DONE    = 2'd2;

    reg [1:0] load_state;
    reg [7:0] load_idx;
    reg       load_capture;
    reg [7:0] load_idx_d1;
    reg       load_capture_d1;

    reg [2:0] bias_idx;
    reg       bias_capture;
    reg [2:0] bias_idx_d1;
    reg       bias_capture_d1;

    wire weights_ready = (load_state == LOAD_DONE);
    wire conv_valid_in = valid_in & weights_ready;

    integer i;
    wire signed [7:0] pixel_in_s = pixel_in;

    rom_CONV1_WEIGHTS #(
        .ADDR_WIDTH(8),
        .DATA_WIDTH(8),
        .DEPTH(150),
        .MEM_FILE("rtl/weights/CONV1_WEIGHTS.mem")
    ) u_conv1_wrom (
        .clk  (clk),
        .addr (conv1_weight_addr),
        .q    (conv1_weight_q)
    );

    rom_CONV1_BIASES_INT32 #(
        .ADDR_WIDTH(3),
        .DATA_WIDTH(32),
        .DEPTH(6),
        .MEM_FILE("rtl/weights/CONV1_BIASES_INT32.mem")
    ) u_conv1_brom (
        .clk  (clk),
        .addr (conv1_bias_addr),
        .q    (conv1_bias_q)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv1_weight_addr <= 8'd0;
            conv1_bias_addr <= 3'd0;
            load_state <= LOAD_WEIGHTS;
            load_idx <= 8'd0;
            load_capture <= 1'b0;
            load_idx_d1 <= 8'd0;
            load_capture_d1 <= 1'b0;
            bias_idx <= 3'd0;
            bias_capture <= 1'b0;
            bias_idx_d1 <= 3'd0;
            bias_capture_d1 <= 1'b0;
            conv1_bias_ch0 <= 32'sd0;
            conv1_bias_ch1 <= 32'sd0;
            conv1_bias_ch2 <= 32'sd0;
            conv1_bias_ch3 <= 32'sd0;
            conv1_bias_ch4 <= 32'sd0;
            conv1_bias_ch5 <= 32'sd0;
            for (i = 0; i < 25; i = i + 1) begin
                conv1_weights_ch0[i] <= 8'sd0;
                conv1_weights_ch1[i] <= 8'sd0;
                conv1_weights_ch2[i] <= 8'sd0;
                conv1_weights_ch3[i] <= 8'sd0;
                conv1_weights_ch4[i] <= 8'sd0;
                conv1_weights_ch5[i] <= 8'sd0;
            end
        end else begin
            case (load_state)
                LOAD_WEIGHTS: begin
                    load_idx_d1 <= load_idx;
                    load_capture_d1 <= load_capture;

                    if (load_capture_d1 && load_idx_d1 != 0 && load_idx_d1 <= 8'd150) begin
                        if (load_idx_d1 - 1 < 25)
                            conv1_weights_ch0[load_idx_d1 - 1] <= conv1_weight_q;
                        else if (load_idx_d1 - 1 < 50)
                            conv1_weights_ch1[load_idx_d1 - 1 - 25] <= conv1_weight_q;
                        else if (load_idx_d1 - 1 < 75)
                            conv1_weights_ch2[load_idx_d1 - 1 - 50] <= conv1_weight_q;
                        else if (load_idx_d1 - 1 < 100)
                            conv1_weights_ch3[load_idx_d1 - 1 - 75] <= conv1_weight_q;
                        else if (load_idx_d1 - 1 < 125)
                            conv1_weights_ch4[load_idx_d1 - 1 - 100] <= conv1_weight_q;
                        else
                            conv1_weights_ch5[load_idx_d1 - 1 - 125] <= conv1_weight_q;
                    end

                    if (load_idx < 150) begin
                        conv1_weight_addr <= load_idx;
                        load_idx <= load_idx + 1'b1;
                        load_capture <= 1'b1;
                    end else begin
                        load_capture <= 1'b0;
                        if (!load_capture_d1) begin
                            load_state <= LOAD_BIAS;
                            bias_idx <= 3'd0;
                            bias_capture <= 1'b0;
                            bias_idx_d1 <= 3'd0;
                            bias_capture_d1 <= 1'b0;
                        end
                    end
                end
                LOAD_BIAS: begin
                    bias_idx_d1 <= bias_idx;
                    bias_capture_d1 <= bias_capture;

                    if (bias_capture_d1 && bias_idx_d1 != 0 && bias_idx_d1 <= 3'd6) begin
                        case (bias_idx_d1 - 1)
                            3'd0: conv1_bias_ch0 <= conv1_bias_q;
                            3'd1: conv1_bias_ch1 <= conv1_bias_q;
                            3'd2: conv1_bias_ch2 <= conv1_bias_q;
                            3'd3: conv1_bias_ch3 <= conv1_bias_q;
                            3'd4: conv1_bias_ch4 <= conv1_bias_q;
                            default: conv1_bias_ch5 <= conv1_bias_q;
                        endcase
                    end

                    if (bias_idx < 6) begin
                        conv1_bias_addr <= bias_idx;
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
    // 2. CONV1 core (6 channels)
    // ==========================================
    wire [31:0] conv_ch0;
    wire [31:0] conv_ch1;
    wire [31:0] conv_ch2;
    wire [31:0] conv_ch3;
    wire [31:0] conv_ch4;
    wire [31:0] conv_ch5;
    wire conv_valid;
    reg [31:0] conv_ch0_r;
    reg [31:0] conv_ch1_r;
    reg [31:0] conv_ch2_r;
    reg [31:0] conv_ch3_r;
    reg [31:0] conv_ch4_r;
    reg [31:0] conv_ch5_r;
    reg conv_valid_r;

    conv1_core #(
        .IMG_WIDTH(28),
        .DATA_WIDTH(8),
        .WEIGHT_WIDTH(8),
        .OUT_WIDTH(32)
    ) u_conv1 (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (conv_valid_in),
        .pixel_in (pixel_in_s),
        .w00_ch0 (conv1_weights_ch0[0]),  .w01_ch0 (conv1_weights_ch0[1]),  .w02_ch0 (conv1_weights_ch0[2]),  .w03_ch0 (conv1_weights_ch0[3]),  .w04_ch0 (conv1_weights_ch0[4]),
        .w10_ch0 (conv1_weights_ch0[5]),  .w11_ch0 (conv1_weights_ch0[6]),  .w12_ch0 (conv1_weights_ch0[7]),  .w13_ch0 (conv1_weights_ch0[8]),  .w14_ch0 (conv1_weights_ch0[9]),
        .w20_ch0 (conv1_weights_ch0[10]), .w21_ch0 (conv1_weights_ch0[11]), .w22_ch0 (conv1_weights_ch0[12]), .w23_ch0 (conv1_weights_ch0[13]), .w24_ch0 (conv1_weights_ch0[14]),
        .w30_ch0 (conv1_weights_ch0[15]), .w31_ch0 (conv1_weights_ch0[16]), .w32_ch0 (conv1_weights_ch0[17]), .w33_ch0 (conv1_weights_ch0[18]), .w34_ch0 (conv1_weights_ch0[19]),
        .w40_ch0 (conv1_weights_ch0[20]), .w41_ch0 (conv1_weights_ch0[21]), .w42_ch0 (conv1_weights_ch0[22]), .w43_ch0 (conv1_weights_ch0[23]), .w44_ch0 (conv1_weights_ch0[24]),
        .bias_ch0 (conv1_bias_ch0),
        .w00_ch1 (conv1_weights_ch1[0]),  .w01_ch1 (conv1_weights_ch1[1]),  .w02_ch1 (conv1_weights_ch1[2]),  .w03_ch1 (conv1_weights_ch1[3]),  .w04_ch1 (conv1_weights_ch1[4]),
        .w10_ch1 (conv1_weights_ch1[5]),  .w11_ch1 (conv1_weights_ch1[6]),  .w12_ch1 (conv1_weights_ch1[7]),  .w13_ch1 (conv1_weights_ch1[8]),  .w14_ch1 (conv1_weights_ch1[9]),
        .w20_ch1 (conv1_weights_ch1[10]), .w21_ch1 (conv1_weights_ch1[11]), .w22_ch1 (conv1_weights_ch1[12]), .w23_ch1 (conv1_weights_ch1[13]), .w24_ch1 (conv1_weights_ch1[14]),
        .w30_ch1 (conv1_weights_ch1[15]), .w31_ch1 (conv1_weights_ch1[16]), .w32_ch1 (conv1_weights_ch1[17]), .w33_ch1 (conv1_weights_ch1[18]), .w34_ch1 (conv1_weights_ch1[19]),
        .w40_ch1 (conv1_weights_ch1[20]), .w41_ch1 (conv1_weights_ch1[21]), .w42_ch1 (conv1_weights_ch1[22]), .w43_ch1 (conv1_weights_ch1[23]), .w44_ch1 (conv1_weights_ch1[24]),
        .bias_ch1 (conv1_bias_ch1),
        .w00_ch2 (conv1_weights_ch2[0]),  .w01_ch2 (conv1_weights_ch2[1]),  .w02_ch2 (conv1_weights_ch2[2]),  .w03_ch2 (conv1_weights_ch2[3]),  .w04_ch2 (conv1_weights_ch2[4]),
        .w10_ch2 (conv1_weights_ch2[5]),  .w11_ch2 (conv1_weights_ch2[6]),  .w12_ch2 (conv1_weights_ch2[7]),  .w13_ch2 (conv1_weights_ch2[8]),  .w14_ch2 (conv1_weights_ch2[9]),
        .w20_ch2 (conv1_weights_ch2[10]), .w21_ch2 (conv1_weights_ch2[11]), .w22_ch2 (conv1_weights_ch2[12]), .w23_ch2 (conv1_weights_ch2[13]), .w24_ch2 (conv1_weights_ch2[14]),
        .w30_ch2 (conv1_weights_ch2[15]), .w31_ch2 (conv1_weights_ch2[16]), .w32_ch2 (conv1_weights_ch2[17]), .w33_ch2 (conv1_weights_ch2[18]), .w34_ch2 (conv1_weights_ch2[19]),
        .w40_ch2 (conv1_weights_ch2[20]), .w41_ch2 (conv1_weights_ch2[21]), .w42_ch2 (conv1_weights_ch2[22]), .w43_ch2 (conv1_weights_ch2[23]), .w44_ch2 (conv1_weights_ch2[24]),
        .bias_ch2 (conv1_bias_ch2),
        .w00_ch3 (conv1_weights_ch3[0]),  .w01_ch3 (conv1_weights_ch3[1]),  .w02_ch3 (conv1_weights_ch3[2]),  .w03_ch3 (conv1_weights_ch3[3]),  .w04_ch3 (conv1_weights_ch3[4]),
        .w10_ch3 (conv1_weights_ch3[5]),  .w11_ch3 (conv1_weights_ch3[6]),  .w12_ch3 (conv1_weights_ch3[7]),  .w13_ch3 (conv1_weights_ch3[8]),  .w14_ch3 (conv1_weights_ch3[9]),
        .w20_ch3 (conv1_weights_ch3[10]), .w21_ch3 (conv1_weights_ch3[11]), .w22_ch3 (conv1_weights_ch3[12]), .w23_ch3 (conv1_weights_ch3[13]), .w24_ch3 (conv1_weights_ch3[14]),
        .w30_ch3 (conv1_weights_ch3[15]), .w31_ch3 (conv1_weights_ch3[16]), .w32_ch3 (conv1_weights_ch3[17]), .w33_ch3 (conv1_weights_ch3[18]), .w34_ch3 (conv1_weights_ch3[19]),
        .w40_ch3 (conv1_weights_ch3[20]), .w41_ch3 (conv1_weights_ch3[21]), .w42_ch3 (conv1_weights_ch3[22]), .w43_ch3 (conv1_weights_ch3[23]), .w44_ch3 (conv1_weights_ch3[24]),
        .bias_ch3 (conv1_bias_ch3),
        .w00_ch4 (conv1_weights_ch4[0]),  .w01_ch4 (conv1_weights_ch4[1]),  .w02_ch4 (conv1_weights_ch4[2]),  .w03_ch4 (conv1_weights_ch4[3]),  .w04_ch4 (conv1_weights_ch4[4]),
        .w10_ch4 (conv1_weights_ch4[5]),  .w11_ch4 (conv1_weights_ch4[6]),  .w12_ch4 (conv1_weights_ch4[7]),  .w13_ch4 (conv1_weights_ch4[8]),  .w14_ch4 (conv1_weights_ch4[9]),
        .w20_ch4 (conv1_weights_ch4[10]), .w21_ch4 (conv1_weights_ch4[11]), .w22_ch4 (conv1_weights_ch4[12]), .w23_ch4 (conv1_weights_ch4[13]), .w24_ch4 (conv1_weights_ch4[14]),
        .w30_ch4 (conv1_weights_ch4[15]), .w31_ch4 (conv1_weights_ch4[16]), .w32_ch4 (conv1_weights_ch4[17]), .w33_ch4 (conv1_weights_ch4[18]), .w34_ch4 (conv1_weights_ch4[19]),
        .w40_ch4 (conv1_weights_ch4[20]), .w41_ch4 (conv1_weights_ch4[21]), .w42_ch4 (conv1_weights_ch4[22]), .w43_ch4 (conv1_weights_ch4[23]), .w44_ch4 (conv1_weights_ch4[24]),
        .bias_ch4 (conv1_bias_ch4),
        .w00_ch5 (conv1_weights_ch5[0]),  .w01_ch5 (conv1_weights_ch5[1]),  .w02_ch5 (conv1_weights_ch5[2]),  .w03_ch5 (conv1_weights_ch5[3]),  .w04_ch5 (conv1_weights_ch5[4]),
        .w10_ch5 (conv1_weights_ch5[5]),  .w11_ch5 (conv1_weights_ch5[6]),  .w12_ch5 (conv1_weights_ch5[7]),  .w13_ch5 (conv1_weights_ch5[8]),  .w14_ch5 (conv1_weights_ch5[9]),
        .w20_ch5 (conv1_weights_ch5[10]), .w21_ch5 (conv1_weights_ch5[11]), .w22_ch5 (conv1_weights_ch5[12]), .w23_ch5 (conv1_weights_ch5[13]), .w24_ch5 (conv1_weights_ch5[14]),
        .w30_ch5 (conv1_weights_ch5[15]), .w31_ch5 (conv1_weights_ch5[16]), .w32_ch5 (conv1_weights_ch5[17]), .w33_ch5 (conv1_weights_ch5[18]), .w34_ch5 (conv1_weights_ch5[19]),
        .w40_ch5 (conv1_weights_ch5[20]), .w41_ch5 (conv1_weights_ch5[21]), .w42_ch5 (conv1_weights_ch5[22]), .w43_ch5 (conv1_weights_ch5[23]), .w44_ch5 (conv1_weights_ch5[24]),
        .bias_ch5 (conv1_bias_ch5),
        .result_ch0 (conv_ch0),
        .result_ch1 (conv_ch1),
        .result_ch2 (conv_ch2),
        .result_ch3 (conv_ch3),
        .result_ch4 (conv_ch4),
        .result_ch5 (conv_ch5),
        .result_valid (conv_valid)
    );

    // ==========================================
    // 3. Max pool 2x2 (6 channels)
    // ==========================================

    wire signed [7:0] conv1_q0;
    wire signed [7:0] conv1_q1;
    wire signed [7:0] conv1_q2;
    wire signed [7:0] conv1_q3;
    wire signed [7:0] conv1_q4;
    wire signed [7:0] conv1_q5;

    wire signed [31:0] conv1_q0_32 = {{24{conv1_q0[7]}}, conv1_q0};
    wire signed [31:0] conv1_q1_32 = {{24{conv1_q1[7]}}, conv1_q1};
    wire signed [31:0] conv1_q2_32 = {{24{conv1_q2[7]}}, conv1_q2};
    wire signed [31:0] conv1_q3_32 = {{24{conv1_q3[7]}}, conv1_q3};
    wire signed [31:0] conv1_q4_32 = {{24{conv1_q4[7]}}, conv1_q4};
    wire signed [31:0] conv1_q5_32 = {{24{conv1_q5[7]}}, conv1_q5};

    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV1), .SHIFT_VAL(`Q_SHIFT)) u_q1_ch0 (.acc_in(conv_ch0_r), .q_out(conv1_q0));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV1), .SHIFT_VAL(`Q_SHIFT)) u_q1_ch1 (.acc_in(conv_ch1_r), .q_out(conv1_q1));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV1), .SHIFT_VAL(`Q_SHIFT)) u_q1_ch2 (.acc_in(conv_ch2_r), .q_out(conv1_q2));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV1), .SHIFT_VAL(`Q_SHIFT)) u_q1_ch3 (.acc_in(conv_ch3_r), .q_out(conv1_q3));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV1), .SHIFT_VAL(`Q_SHIFT)) u_q1_ch4 (.acc_in(conv_ch4_r), .q_out(conv1_q4));
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV1), .SHIFT_VAL(`Q_SHIFT)) u_q1_ch5 (.acc_in(conv_ch5_r), .q_out(conv1_q5));
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_ch0_r <= 0;
            conv_ch1_r <= 0;
            conv_ch2_r <= 0;
            conv_ch3_r <= 0;
            conv_ch4_r <= 0;
            conv_ch5_r <= 0;
            conv_valid_r <= 1'b0;
        end else begin
            conv_ch0_r <= conv_ch0;
            conv_ch1_r <= conv_ch1;
            conv_ch2_r <= conv_ch2;
            conv_ch3_r <= conv_ch3;
            conv_ch4_r <= conv_ch4;
            conv_ch5_r <= conv_ch5;
            conv_valid_r <= conv_valid;
        end
    end

    max_pool_2x2 #(
        .IN_WIDTH(24),
        .IN_HEIGHT(24)
    ) u_pool (
        .clk      (clk),
        .rst_n    (rst_n),
        .valid_in (conv_valid_r),
        .in_ch0   (conv1_q0_32),
        .in_ch1   (conv1_q1_32),
        .in_ch2   (conv1_q2_32),
        .in_ch3   (conv1_q3_32),
        .in_ch4   (conv1_q4_32),
        .in_ch5   (conv1_q5_32),
        .out_ch0  (result_ch0),
        .out_ch1  (result_ch1),
        .out_ch2  (result_ch2),
        .out_ch3  (result_ch3),
        .out_ch4  (result_ch4),
        .out_ch5  (result_ch5),
        .out_valid(result_valid)
    );

endmodule
