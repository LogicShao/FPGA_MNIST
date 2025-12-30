`timescale 1ns/1ps
`include "quant_params.vh"

module layer2_block(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire signed [7:0] in_ch0,
    input wire signed [7:0] in_ch1,
    input wire signed [7:0] in_ch2,
    input wire signed [7:0] in_ch3,
    input wire signed [7:0] in_ch4,
    input wire signed [7:0] in_ch5,
    output wire signed [7:0] out_ch0,
    output wire signed [7:0] out_ch1,
    output wire signed [7:0] out_ch2,
    output wire signed [7:0] out_ch3,
    output wire signed [7:0] out_ch4,
    output wire signed [7:0] out_ch5,
    output wire signed [7:0] out_ch6,
    output wire signed [7:0] out_ch7,
    output wire signed [7:0] out_ch8,
    output wire signed [7:0] out_ch9,
    output wire signed [7:0] out_ch10,
    output wire signed [7:0] out_ch11,
    output wire signed [7:0] out_ch12,
    output wire signed [7:0] out_ch13,
    output wire signed [7:0] out_ch14,
    output wire signed [7:0] out_ch15,
    output wire out_valid
);

    // ==========================================
    // 1. Weight ROM signals and loader
    // ==========================================
    wire signed [7:0] conv2_weight_q;
    wire signed [31:0] conv2_bias_q;

    wire [11:0] conv2_weight_addr;
    reg [3:0] conv2_bias_addr;

    reg signed [31:0] conv2_biases [0:15];

    localparam LOAD_BIAS    = 1'd0;
    localparam LOAD_DONE    = 1'd1;

    reg        load_state;

    reg [4:0] bias_idx;
    reg       bias_capture;
    reg [4:0] bias_idx_d1;
    reg       bias_capture_d1;

    wire weights_ready = (load_state == LOAD_DONE);

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
            conv2_bias_addr <= 4'd0;
            load_state <= LOAD_BIAS;
            bias_idx <= 5'd0;
            bias_capture <= 1'b0;
            bias_idx_d1 <= 5'd0;
            bias_capture_d1 <= 1'b0;
        end else begin
            case (load_state)
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
    // 2. CONV2 sequential core (16 channels, 1 MAC)
    // ==========================================
    reg [7:0] feat_idx;
    reg [2:0] pos_x;
    reg [2:0] pos_y;
    reg [3:0] oc_idx;
    reg [2:0] k_ch;
    reg [2:0] k_y;
    reg [2:0] k_x;
    reg signed [63:0] acc;
    reg mac_phase;

    reg signed [31:0] conv2_out_buf [0:15];
    reg conv2_valid;
    reg conv2_valid_r;
    reg signed [7:0] conv2_q_buf [0:15];
    reg [3:0] q_idx;

    (* ramstyle = "M9K" *) reg signed [7:0] feat_mem0 [0:143];
    (* ramstyle = "M9K" *) reg signed [7:0] feat_mem1 [0:143];
    (* ramstyle = "M9K" *) reg signed [7:0] feat_mem2 [0:143];
    (* ramstyle = "M9K" *) reg signed [7:0] feat_mem3 [0:143];
    (* ramstyle = "M9K" *) reg signed [7:0] feat_mem4 [0:143];
    (* ramstyle = "M9K" *) reg signed [7:0] feat_mem5 [0:143];
    reg [7:0] feat_wr_addr;
    reg signed [7:0] feat_wr_data0;
    reg signed [7:0] feat_wr_data1;
    reg signed [7:0] feat_wr_data2;
    reg signed [7:0] feat_wr_data3;
    reg signed [7:0] feat_wr_data4;
    reg signed [7:0] feat_wr_data5;
    reg feat_wr_en;
    reg [7:0] feat_rd_addr;
    reg signed [7:0] feat_rd_data0;
    reg signed [7:0] feat_rd_data1;
    reg signed [7:0] feat_rd_data2;
    reg signed [7:0] feat_rd_data3;
    reg signed [7:0] feat_rd_data4;
    reg signed [7:0] feat_rd_data5;

    localparam C_IDLE = 2'd0;
    localparam C_LOAD = 2'd1;
    localparam C_MAC  = 2'd2;
    localparam C_OUT  = 2'd3;
    reg [1:0] c_state;

    wire [3:0] base_row = pos_y + k_y;
    wire [3:0] base_col = pos_x + k_x;
    wire [7:0] row12 = {base_row, 3'b0} + {base_row, 2'b0};
    wire [7:0] in_index = row12 + base_col;
    wire signed [7:0] feat_val =
        (k_ch == 3'd0) ? feat_rd_data0 :
        (k_ch == 3'd1) ? feat_rd_data1 :
        (k_ch == 3'd2) ? feat_rd_data2 :
        (k_ch == 3'd3) ? feat_rd_data3 :
        (k_ch == 3'd4) ? feat_rd_data4 :
                         feat_rd_data5;

    wire [7:0] k_ch_25 = {k_ch, 4'b0} + {k_ch, 3'b0} + k_ch;
    wire [7:0] k_y_5 = {k_y, 2'b0} + k_y;
    wire [7:0] k_flat = k_ch_25 + k_y_5 + k_x;
    wire [11:0] oc_idx_ext = {8'd0, oc_idx};
    wire [11:0] oc150 = (oc_idx_ext << 7) + (oc_idx_ext << 4) + (oc_idx_ext << 2) + (oc_idx_ext << 1);
    assign conv2_weight_addr = oc150 + {4'd0, k_flat};
    wire signed [7:0] w_val = conv2_weight_q;

    wire signed [15:0] prod = $signed(feat_val) * $signed(w_val);
    wire signed [63:0] acc_next = acc + {{48{prod[15]}}, prod};
    wire signed [63:0] acc_bias = acc_next + $signed(conv2_biases[oc_idx]);
    wire last_k = (k_ch == 3'd5) && (k_y == 3'd4) && (k_x == 3'd4);

    integer oc_init;
    always @(posedge clk) begin
        if (feat_wr_en) begin
            feat_mem0[feat_wr_addr] <= feat_wr_data0;
            feat_mem1[feat_wr_addr] <= feat_wr_data1;
            feat_mem2[feat_wr_addr] <= feat_wr_data2;
            feat_mem3[feat_wr_addr] <= feat_wr_data3;
            feat_mem4[feat_wr_addr] <= feat_wr_data4;
            feat_mem5[feat_wr_addr] <= feat_wr_data5;
        end
        feat_rd_data0 <= feat_mem0[feat_rd_addr];
        feat_rd_data1 <= feat_mem1[feat_rd_addr];
        feat_rd_data2 <= feat_mem2[feat_rd_addr];
        feat_rd_data3 <= feat_mem3[feat_rd_addr];
        feat_rd_data4 <= feat_mem4[feat_rd_addr];
        feat_rd_data5 <= feat_mem5[feat_rd_addr];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c_state <= C_IDLE;
            feat_idx <= 8'd0;
            pos_x <= 3'd0;
            pos_y <= 3'd0;
            oc_idx <= 4'd0;
            k_ch <= 3'd0;
            k_y <= 3'd0;
            k_x <= 3'd0;
            acc <= 64'd0;
            mac_phase <= 1'b0;
            conv2_valid <= 1'b0;
            q_idx <= 4'd0;
            feat_wr_en <= 1'b0;
            feat_wr_addr <= 8'd0;
            feat_wr_data0 <= 8'sd0;
            feat_wr_data1 <= 8'sd0;
            feat_wr_data2 <= 8'sd0;
            feat_wr_data3 <= 8'sd0;
            feat_wr_data4 <= 8'sd0;
            feat_wr_data5 <= 8'sd0;
            feat_rd_addr <= 8'd0;
            for (oc_init = 0; oc_init < 16; oc_init = oc_init + 1)
                conv2_out_buf[oc_init] <= 32'sd0;
        end else begin
            conv2_valid <= 1'b0;
            feat_wr_en <= 1'b0;
            case (c_state)
                C_IDLE: begin
                    if (weights_ready && valid_in) begin
                        feat_wr_en <= 1'b1;
                        feat_wr_addr <= 8'd0;
                        feat_wr_data0 <= in_ch0;
                        feat_wr_data1 <= in_ch1;
                        feat_wr_data2 <= in_ch2;
                        feat_wr_data3 <= in_ch3;
                        feat_wr_data4 <= in_ch4;
                        feat_wr_data5 <= in_ch5;
                        feat_idx <= 8'd1;
                        c_state <= C_LOAD;
                    end
                end
                C_LOAD: begin
                    if (valid_in) begin
                        feat_wr_en <= 1'b1;
                        feat_wr_addr <= feat_idx;
                        feat_wr_data0 <= in_ch0;
                        feat_wr_data1 <= in_ch1;
                        feat_wr_data2 <= in_ch2;
                        feat_wr_data3 <= in_ch3;
                        feat_wr_data4 <= in_ch4;
                        feat_wr_data5 <= in_ch5;
                        if (feat_idx == 8'd143) begin
                            feat_idx <= 8'd0;
                            pos_x <= 3'd0;
                            pos_y <= 3'd0;
                            oc_idx <= 4'd0;
                            k_ch <= 3'd0;
                            k_y <= 3'd0;
                            k_x <= 3'd0;
                            acc <= 64'd0;
                            mac_phase <= 1'b0;
                            c_state <= C_MAC;
                        end else begin
                            feat_idx <= feat_idx + 1'b1;
                        end
                    end
                end
                C_MAC: begin
                    if (weights_ready) begin
                        if (!mac_phase) begin
                            feat_rd_addr <= in_index;
                            mac_phase <= 1'b1;
                        end else begin
                            if (last_k) begin
                                if (acc_bias[63])
                                    conv2_out_buf[oc_idx] <= 0;
                                else
                                    conv2_out_buf[oc_idx] <= acc_bias[31:0];
                                if (oc_idx == 4'd15) begin
                                    c_state <= C_OUT;
                                    oc_idx <= 4'd0;
                                    q_idx <= 4'd0;
                                end else begin
                                    oc_idx <= oc_idx + 1'b1;
                                end
                                k_ch <= 3'd0;
                                k_y <= 3'd0;
                                k_x <= 3'd0;
                                acc <= 64'd0;
                            end else begin
                                acc <= acc_next;
                                if (k_x == 3'd4) begin
                                    k_x <= 3'd0;
                                    if (k_y == 3'd4) begin
                                        k_y <= 3'd0;
                                        if (k_ch == 3'd5)
                                            k_ch <= 3'd0;
                                        else
                                            k_ch <= k_ch + 1'b1;
                                    end else begin
                                        k_y <= k_y + 1'b1;
                                    end
                                end else begin
                                    k_x <= k_x + 1'b1;
                                end
                            end
                            mac_phase <= 1'b0;
                        end
                    end
                end
                C_OUT: begin
                    conv2_q_buf[q_idx] <= conv2_q;
                    if (q_idx == 4'd15) begin
                        conv2_valid <= 1'b1;
                        mac_phase <= 1'b0;
                        oc_idx <= 4'd0;
                        k_ch <= 3'd0;
                        k_y <= 3'd0;
                        k_x <= 3'd0;
                        acc <= 64'd0;
                        q_idx <= 4'd0;
                        if (pos_x == 3'd7) begin
                            pos_x <= 3'd0;
                            if (pos_y == 3'd7) begin
                                pos_y <= 3'd0;
                                c_state <= C_IDLE;
                            end else begin
                                pos_y <= pos_y + 1'b1;
                                c_state <= C_MAC;
                            end
                        end else begin
                            pos_x <= pos_x + 1'b1;
                            c_state <= C_MAC;
                        end
                    end else begin
                        q_idx <= q_idx + 1'b1;
                    end
                end
                default: begin
                    c_state <= C_IDLE;
                end
            endcase
        end
    end

    // ==========================================
    // 3. Max pool 2x2 (16 channels)
    // ==========================================

    wire signed [31:0] conv2_q_sel =
        (q_idx == 4'd0)  ? conv2_out_buf[0]  :
        (q_idx == 4'd1)  ? conv2_out_buf[1]  :
        (q_idx == 4'd2)  ? conv2_out_buf[2]  :
        (q_idx == 4'd3)  ? conv2_out_buf[3]  :
        (q_idx == 4'd4)  ? conv2_out_buf[4]  :
        (q_idx == 4'd5)  ? conv2_out_buf[5]  :
        (q_idx == 4'd6)  ? conv2_out_buf[6]  :
        (q_idx == 4'd7)  ? conv2_out_buf[7]  :
        (q_idx == 4'd8)  ? conv2_out_buf[8]  :
        (q_idx == 4'd9)  ? conv2_out_buf[9]  :
        (q_idx == 4'd10) ? conv2_out_buf[10] :
        (q_idx == 4'd11) ? conv2_out_buf[11] :
        (q_idx == 4'd12) ? conv2_out_buf[12] :
        (q_idx == 4'd13) ? conv2_out_buf[13] :
        (q_idx == 4'd14) ? conv2_out_buf[14] :
                           conv2_out_buf[15];
    wire signed [7:0] conv2_q;
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV2), .SHIFT_VAL(`Q_SHIFT)) u_q2 (.acc_in(conv2_q_sel), .q_out(conv2_q));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv2_valid_r <= 1'b0;
        end else begin
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
        .in_ch0   (conv2_q_buf[0]),
        .in_ch1   (conv2_q_buf[1]),
        .in_ch2   (conv2_q_buf[2]),
        .in_ch3   (conv2_q_buf[3]),
        .in_ch4   (conv2_q_buf[4]),
        .in_ch5   (conv2_q_buf[5]),
        .in_ch6   (conv2_q_buf[6]),
        .in_ch7   (conv2_q_buf[7]),
        .in_ch8   (conv2_q_buf[8]),
        .in_ch9   (conv2_q_buf[9]),
        .in_ch10  (conv2_q_buf[10]),
        .in_ch11  (conv2_q_buf[11]),
        .in_ch12  (conv2_q_buf[12]),
        .in_ch13  (conv2_q_buf[13]),
        .in_ch14  (conv2_q_buf[14]),
        .in_ch15  (conv2_q_buf[15]),
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
