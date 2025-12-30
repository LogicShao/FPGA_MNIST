`timescale 1ns/1ps
`include "quant_params.vh"

module layer3_fc1(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire signed [7:0] in_ch0,
    input wire signed [7:0] in_ch1,
    input wire signed [7:0] in_ch2,
    input wire signed [7:0] in_ch3,
    input wire signed [7:0] in_ch4,
    input wire signed [7:0] in_ch5,
    input wire signed [7:0] in_ch6,
    input wire signed [7:0] in_ch7,
    input wire signed [7:0] in_ch8,
    input wire signed [7:0] in_ch9,
    input wire signed [7:0] in_ch10,
    input wire signed [7:0] in_ch11,
    input wire signed [7:0] in_ch12,
    input wire signed [7:0] in_ch13,
    input wire signed [7:0] in_ch14,
    input wire signed [7:0] in_ch15,
    output reg signed [7:0] out_data,
    output reg out_valid
);

    // FC1: 256 -> 32

    wire signed [7:0] fc1_weight_q;
    wire signed [31:0] fc1_bias_q;

    wire [12:0] fc1_weight_addr;
    reg [4:0] fc1_bias_addr;

    reg signed [31:0] fc1_biases [0:31];

    localparam LOAD_BIAS    = 1'd0;
    localparam LOAD_DONE    = 1'd1;

    reg        load_state;

    reg [5:0] bias_idx;
    reg       bias_capture;
    reg [5:0] bias_idx_d1;
    reg       bias_capture_d1;

    wire weights_ready = (load_state == LOAD_DONE);

    rom_FC1_WEIGHTS #(
        .ADDR_WIDTH(13),
        .DATA_WIDTH(8),
        .DEPTH(8192),
        .MEM_FILE("rtl/weights/FC1_WEIGHTS.mem")
    ) u_fc1_wrom (
        .clk  (clk),
        .addr (fc1_weight_addr),
        .q    (fc1_weight_q)
    );

    rom_FC1_BIASES_INT32 #(
        .ADDR_WIDTH(5),
        .DATA_WIDTH(32),
        .DEPTH(32),
        .MEM_FILE("rtl/weights/FC1_BIASES_INT32.mem")
    ) u_fc1_brom (
        .clk  (clk),
        .addr (fc1_bias_addr),
        .q    (fc1_bias_q)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc1_bias_addr <= 5'd0;
            load_state <= LOAD_BIAS;
            bias_idx <= 6'd0;
            bias_capture <= 1'b0;
            bias_idx_d1 <= 6'd0;
            bias_capture_d1 <= 1'b0;
        end else begin
            case (load_state)
                LOAD_BIAS: begin
                    bias_idx_d1 <= bias_idx;
                    bias_capture_d1 <= bias_capture;

                    if (bias_capture_d1 && bias_idx_d1 != 0 && bias_idx_d1 <= 6'd32) begin
                        fc1_biases[bias_idx_d1 - 1] <= fc1_bias_q;
                    end
                    if (bias_idx < 6'd32) begin
                        fc1_bias_addr <= bias_idx[4:0];
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

    reg signed [7:0] feature_buf [0:255];
    reg signed [31:0] out_buf [0:31];

    localparam S_LOAD = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_OUTPUT = 2'd2;

    reg [1:0] state;
    reg [3:0] sample_idx;
    reg [5:0] out_idx;
    reg [5:0] oc_idx;
    reg [7:0] in_idx;

    assign fc1_weight_addr = {oc_idx[4:0], in_idx};

    
    function automatic signed [7:0] quantize32;
        input signed [31:0] val;
        reg signed [63:0] prod;
        reg signed [63:0] rounded;
        reg signed [63:0] shifted;
        begin
            prod = val * $signed(`Q_MULT_FC1);
            rounded = prod + $signed(1 << (`Q_SHIFT - 1));
            shifted = rounded >>> `Q_SHIFT;
            if (shifted > 127)
                quantize32 = 8'sd127;
            else if (shifted < -128)
                quantize32 = -8'sd128;
            else
                quantize32 = shifted[7:0];
        end
    endfunction
    wire signed [7:0] out_q = quantize32(out_buf[out_idx]);
    wire signed [63:0] fc1_prod = $signed(feature_buf[in_idx]) * $signed(fc1_weight_q);
    wire signed [63:0] fc1_acc_next = acc + fc1_prod;
    wire signed [63:0] fc1_acc_bias = fc1_acc_next + $signed(fc1_biases[oc_idx]);
    integer oc;
    reg signed [63:0] acc;
    reg mac_phase;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_LOAD;
            sample_idx <= 4'd0;
            out_idx <= 6'd0;
            oc_idx <= 6'd0;
            in_idx <= 8'd0;
            acc <= 64'd0;
            mac_phase <= 1'b0;
            out_data <= 0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            case (state)
                S_LOAD: begin
                    if (weights_ready && valid_in) begin
                        feature_buf[0 * 16 + sample_idx] <= in_ch0;
                        feature_buf[1 * 16 + sample_idx] <= in_ch1;
                        feature_buf[2 * 16 + sample_idx] <= in_ch2;
                        feature_buf[3 * 16 + sample_idx] <= in_ch3;
                        feature_buf[4 * 16 + sample_idx] <= in_ch4;
                        feature_buf[5 * 16 + sample_idx] <= in_ch5;
                        feature_buf[6 * 16 + sample_idx] <= in_ch6;
                        feature_buf[7 * 16 + sample_idx] <= in_ch7;
                        feature_buf[8 * 16 + sample_idx] <= in_ch8;
                        feature_buf[9 * 16 + sample_idx] <= in_ch9;
                        feature_buf[10 * 16 + sample_idx] <= in_ch10;
                        feature_buf[11 * 16 + sample_idx] <= in_ch11;
                        feature_buf[12 * 16 + sample_idx] <= in_ch12;
                        feature_buf[13 * 16 + sample_idx] <= in_ch13;
                        feature_buf[14 * 16 + sample_idx] <= in_ch14;
                        feature_buf[15 * 16 + sample_idx] <= in_ch15;
                        if (sample_idx == 4'd15) begin
                            sample_idx <= 4'd0;
                            state <= S_COMPUTE;
                            oc_idx <= 6'd0;
                            in_idx <= 8'd0;
                            acc <= 64'd0;
                            mac_phase <= 1'b0;
                        end else begin
                            sample_idx <= sample_idx + 1'b1;
                        end
                    end
                end
                S_COMPUTE: begin
                    if (weights_ready) begin
`ifdef FAST_SIM
                        for (oc = 0; oc < 32; oc = oc + 1) begin
                            out_buf[oc] <= feature_buf[oc];
                        end
                        out_idx <= 6'd0;
                        state <= S_OUTPUT;
`else
                        if (!mac_phase) begin
                            mac_phase <= 1'b1;
                        end else begin
                            if (in_idx == 8'd255) begin
                                if (fc1_acc_bias[63])
                                    out_buf[oc_idx] <= 0;
                                else
                                    out_buf[oc_idx] <= fc1_acc_bias[31:0];
                                if (oc_idx == 6'd31) begin
                                    oc_idx <= 6'd0;
                                    in_idx <= 8'd0;
                                    acc <= 64'd0;
                                    out_idx <= 6'd0;
                                    state <= S_OUTPUT;
                                end else begin
                                    oc_idx <= oc_idx + 1'b1;
                                    in_idx <= 8'd0;
                                    acc <= 64'd0;
                                end
                            end else begin
                                acc <= fc1_acc_next;
                                in_idx <= in_idx + 1'b1;
                            end
                            mac_phase <= 1'b0;
                        end
`endif
                    end
                end
                S_OUTPUT: begin
                    out_data <= out_q;
                    out_valid <= 1'b1;
                    if (out_idx == 6'd31) begin
                        out_idx <= 6'd0;
                        state <= S_LOAD;
                    end else begin
                        out_idx <= out_idx + 1'b1;
                    end
                end
                default: begin
                    state <= S_LOAD;
                end
            endcase
        end
    end

endmodule
