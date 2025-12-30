`timescale 1ns/1ps
`include "quant_params.vh"

module layer1_block(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [7:0] pixel_in,

    output wire signed [7:0] result_ch0,
    output wire signed [7:0] result_ch1,
    output wire signed [7:0] result_ch2,
    output wire signed [7:0] result_ch3,
    output wire signed [7:0] result_ch4,
    output wire signed [7:0] result_ch5,
    output wire result_valid
);

    // ==========================================
    // 1. Weight ROM signals and loader
    // ==========================================
    wire signed [7:0] conv1_weight_q;
    wire signed [31:0] conv1_bias_q;

    wire [7:0] conv1_weight_addr;
    reg [2:0] conv1_bias_addr;

    reg signed [31:0] conv1_bias_ch0;
    reg signed [31:0] conv1_bias_ch1;
    reg signed [31:0] conv1_bias_ch2;
    reg signed [31:0] conv1_bias_ch3;
    reg signed [31:0] conv1_bias_ch4;
    reg signed [31:0] conv1_bias_ch5;

    localparam LOAD_BIAS    = 1'd0;
    localparam LOAD_DONE    = 1'd1;

    reg        load_state;

    reg [2:0] bias_idx;
    reg       bias_capture;
    reg [2:0] bias_idx_d1;
    reg       bias_capture_d1;

    wire weights_ready = (load_state == LOAD_DONE);

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
        .DEPTH(8),
        .MEM_FILE("rtl/weights/CONV1_BIASES_INT32.mem")
    ) u_conv1_brom (
        .clk  (clk),
        .addr (conv1_bias_addr),
        .q    (conv1_bias_q)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv1_bias_addr <= 3'd0;
            load_state <= LOAD_BIAS;
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
        end else begin
            case (load_state)
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
    // 2. CONV1 sequential core (6 channels, 1 MAC)
    // ==========================================
    (* ramstyle = "M9K" *) reg signed [7:0] img_mem [0:783];
    reg [9:0] img_wr_addr;
    reg signed [7:0] img_wr_data;
    reg img_wr_en;
    reg [9:0] img_rd_addr;
    reg signed [7:0] img_rd_data;
    reg [9:0] img_idx;
    reg [4:0] pos_x;
    reg [4:0] pos_y;
    reg [2:0] k_x;
    reg [2:0] k_y;
    reg [2:0] oc_idx;
    reg signed [63:0] acc;
    reg signed [31:0] conv1_out_buf [0:5];
    reg mac_phase;

    reg conv_valid;

    reg conv_valid_r;
    reg signed [7:0] conv1_q_buf [0:5];
    reg [2:0] q_idx;

    localparam C1_IDLE = 2'd0;
    localparam C1_LOAD = 2'd1;
    localparam C1_MAC  = 2'd2;
    localparam C1_OUT  = 2'd3;
    reg [1:0] c1_state;

    wire [5:0] k_y_5 = {k_y, 2'b0} + k_y;
    wire [5:0] k_flat = k_y_5 + k_x;
    wire [7:0] oc_idx_ext = {5'd0, oc_idx};
    wire [7:0] oc25 = (oc_idx_ext << 4) + (oc_idx_ext << 3) + oc_idx_ext;
    assign conv1_weight_addr = oc25 + {3'd0, k_flat};
    wire signed [7:0] w_sel = conv1_weight_q;

    wire signed [31:0] bias_sel =
        (oc_idx == 3'd0) ? conv1_bias_ch0 :
        (oc_idx == 3'd1) ? conv1_bias_ch1 :
        (oc_idx == 3'd2) ? conv1_bias_ch2 :
        (oc_idx == 3'd3) ? conv1_bias_ch3 :
        (oc_idx == 3'd4) ? conv1_bias_ch4 :
                           conv1_bias_ch5;

    wire [5:0] base_row = pos_y + k_y;
    wire [5:0] base_col = pos_x + k_x;
    wire [9:0] row28 = {base_row, 4'b0} + {base_row, 3'b0} + {base_row, 2'b0};
    wire [9:0] img_index = row28 + base_col;
    wire signed [7:0] img_val = img_rd_data;
    wire signed [15:0] prod = img_val * w_sel;
    wire signed [63:0] acc_next = acc + {{48{prod[15]}}, prod};
    wire signed [63:0] acc_bias = acc_next + {{32{bias_sel[31]}}, bias_sel};
    wire last_k = (k_y == 3'd4) && (k_x == 3'd4);

    integer oc_init;
    always @(posedge clk) begin
        if (img_wr_en)
            img_mem[img_wr_addr] <= img_wr_data;
    end

    always @(*) begin
        img_rd_data = img_mem[img_rd_addr];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            img_idx <= 10'd0;
            pos_x <= 5'd0;
            pos_y <= 5'd0;
            k_x <= 3'd0;
            k_y <= 3'd0;
            oc_idx <= 3'd0;
            acc <= 64'd0;
            mac_phase <= 1'b0;
            c1_state <= C1_IDLE;
            conv_valid <= 1'b0;
            q_idx <= 3'd0;
            img_wr_en <= 1'b0;
            img_wr_addr <= 10'd0;
            img_wr_data <= 8'sd0;
            img_rd_addr <= 10'd0;
            for (oc_init = 0; oc_init < 6; oc_init = oc_init + 1)
                conv1_out_buf[oc_init] <= 32'sd0;
        end else begin
            conv_valid <= 1'b0;
            img_wr_en <= 1'b0;
            case (c1_state)
                C1_IDLE: begin
                    if (weights_ready && valid_in) begin
                        img_wr_en <= 1'b1;
                        img_wr_addr <= 10'd0;
                        img_wr_data <= pixel_in_s;
                        img_idx <= 10'd1;
                        c1_state <= C1_LOAD;
                    end
                end
                C1_LOAD: begin
                    if (valid_in) begin
                        img_wr_en <= 1'b1;
                        img_wr_addr <= img_idx;
                        img_wr_data <= pixel_in_s;
                        if (img_idx == 10'd783) begin
                            img_idx <= 10'd0;
                            pos_x <= 5'd0;
                            pos_y <= 5'd0;
                            oc_idx <= 3'd0;
                            k_x <= 3'd0;
                            k_y <= 3'd0;
                            acc <= 64'd0;
                            mac_phase <= 1'b0;
                            c1_state <= C1_MAC;
                        end else begin
                            img_idx <= img_idx + 1'b1;
                        end
                    end
                end
                C1_MAC: begin
                    if (weights_ready) begin
                        if (!mac_phase) begin
                            img_rd_addr <= img_index;
                            mac_phase <= 1'b1;
                        end else begin
                            if (last_k) begin
                                if (acc_bias[63])
                                    conv1_out_buf[oc_idx] <= 0;
                                else
                                    conv1_out_buf[oc_idx] <= acc_bias[31:0];
                                if (oc_idx == 3'd5) begin
                                    c1_state <= C1_OUT;
                                    oc_idx <= 3'd0;
                                    q_idx <= 3'd0;
                                end else begin
                                    oc_idx <= oc_idx + 1'b1;
                                end
                                k_x <= 3'd0;
                                k_y <= 3'd0;
                                acc <= 64'd0;
                            end else begin
                                acc <= acc_next;
                                if (k_x == 3'd4) begin
                                    k_x <= 3'd0;
                                    k_y <= k_y + 1'b1;
                                end else begin
                                    k_x <= k_x + 1'b1;
                                end
                            end
                            mac_phase <= 1'b0;
                        end
                    end
                end
                C1_OUT: begin
                    conv1_q_buf[q_idx] <= conv1_q;
                    if (q_idx == 3'd5) begin
                        conv_valid <= 1'b1;
                        mac_phase <= 1'b0;
                        oc_idx <= 3'd0;
                        k_x <= 3'd0;
                        k_y <= 3'd0;
                        acc <= 64'd0;
                        q_idx <= 3'd0;
                        if (pos_x == 5'd23) begin
                            pos_x <= 5'd0;
                            if (pos_y == 5'd23) begin
                                pos_y <= 5'd0;
                                c1_state <= C1_IDLE;
                            end else begin
                                pos_y <= pos_y + 1'b1;
                                c1_state <= C1_MAC;
                            end
                        end else begin
                            pos_x <= pos_x + 1'b1;
                            c1_state <= C1_MAC;
                        end
                    end else begin
                        q_idx <= q_idx + 1'b1;
                    end
                end
                default: c1_state <= C1_IDLE;
            endcase
        end
    end

    // ==========================================
    // 3. Max pool 2x2 (6 channels)
    // ==========================================

    wire signed [31:0] conv1_q_sel =
        (q_idx == 3'd0) ? conv1_out_buf[0] :
        (q_idx == 3'd1) ? conv1_out_buf[1] :
        (q_idx == 3'd2) ? conv1_out_buf[2] :
        (q_idx == 3'd3) ? conv1_out_buf[3] :
        (q_idx == 3'd4) ? conv1_out_buf[4] :
                          conv1_out_buf[5];
    wire signed [7:0] conv1_q;
    scaler_quantize #(.MULT_VAL(`Q_MULT_CONV1), .SHIFT_VAL(`Q_SHIFT)) u_q1 (.acc_in(conv1_q_sel), .q_out(conv1_q));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            conv_valid_r <= 1'b0;
        end else begin
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
        .in_ch0   (conv1_q_buf[0]),
        .in_ch1   (conv1_q_buf[1]),
        .in_ch2   (conv1_q_buf[2]),
        .in_ch3   (conv1_q_buf[3]),
        .in_ch4   (conv1_q_buf[4]),
        .in_ch5   (conv1_q_buf[5]),
        .out_ch0  (result_ch0),
        .out_ch1  (result_ch1),
        .out_ch2  (result_ch2),
        .out_ch3  (result_ch3),
        .out_ch4  (result_ch4),
        .out_ch5  (result_ch5),
        .out_valid(result_valid)
    );

endmodule
