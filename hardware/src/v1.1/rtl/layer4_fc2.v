`timescale 1ns/1ps

module layer4_fc2(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [31:0] in_data,
    output reg [31:0] out_data,
    output reg out_valid
);

    // FC2: 32 -> 10

    wire signed [7:0] fc2_weight_q;
    wire signed [31:0] fc2_bias_q;

    reg [8:0] fc2_weight_addr;
    reg [3:0] fc2_bias_addr;

    reg signed [7:0] fc2_weights [0:319];
    reg signed [31:0] fc2_biases [0:9];

    localparam LOAD_WEIGHTS = 2'd0;
    localparam LOAD_BIAS    = 2'd1;
    localparam LOAD_DONE    = 2'd2;

    reg [1:0] load_state;
    reg [8:0] load_idx;
    reg       load_capture;
    reg [8:0] load_idx_d1;
    reg       load_capture_d1;

    reg [3:0] bias_idx;
    reg       bias_capture;
    reg [3:0] bias_idx_d1;
    reg       bias_capture_d1;

    wire weights_ready = (load_state == LOAD_DONE);

    rom_FC2_WEIGHTS #(
        .ADDR_WIDTH(9),
        .DATA_WIDTH(8),
        .DEPTH(320),
        .MEM_FILE("rtl/weights/FC2_WEIGHTS.mem")
    ) u_fc2_wrom (
        .clk  (clk),
        .addr (fc2_weight_addr),
        .q    (fc2_weight_q)
    );

    rom_FC2_BIASES_INT32 #(
        .ADDR_WIDTH(4),
        .DATA_WIDTH(32),
        .DEPTH(10),
        .MEM_FILE("rtl/weights/FC2_BIASES_INT32.mem")
    ) u_fc2_brom (
        .clk  (clk),
        .addr (fc2_bias_addr),
        .q    (fc2_bias_q)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc2_weight_addr <= 9'd0;
            fc2_bias_addr <= 4'd0;
            load_state <= LOAD_WEIGHTS;
            load_idx <= 9'd0;
            load_capture <= 1'b0;
            load_idx_d1 <= 9'd0;
            load_capture_d1 <= 1'b0;
            bias_idx <= 4'd0;
            bias_capture <= 1'b0;
            bias_idx_d1 <= 4'd0;
            bias_capture_d1 <= 1'b0;
        end else begin
            case (load_state)
                LOAD_WEIGHTS: begin
                    load_idx_d1 <= load_idx;
                    load_capture_d1 <= load_capture;

                    if (load_capture_d1 && load_idx_d1 != 0 && load_idx_d1 <= 9'd320) begin
                        fc2_weights[load_idx_d1 - 1] <= fc2_weight_q;
                    end
                    if (load_idx < 320) begin
                        fc2_weight_addr <= load_idx;
                        load_idx <= load_idx + 1'b1;
                        load_capture <= 1'b1;
                    end else begin
                        load_capture <= 1'b0;
                        if (!load_capture_d1) begin
                            load_state <= LOAD_BIAS;
                            bias_idx <= 4'd0;
                            bias_capture <= 1'b0;
                            bias_idx_d1 <= 4'd0;
                            bias_capture_d1 <= 1'b0;
                        end
                    end
                end
                LOAD_BIAS: begin
                    bias_idx_d1 <= bias_idx;
                    bias_capture_d1 <= bias_capture;

                    if (bias_capture_d1 && bias_idx_d1 != 0 && bias_idx_d1 <= 4'd10) begin
                        fc2_biases[bias_idx_d1 - 1] <= fc2_bias_q;
                    end
                    if (bias_idx < 10) begin
                        fc2_bias_addr <= bias_idx;
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

    reg signed [31:0] fc1_buf [0:31];
    reg signed [31:0] out_buf [0:9];

    localparam S_LOAD = 2'd0;
    localparam S_COMPUTE = 2'd1;
    localparam S_OUTPUT = 2'd2;

    reg [1:0] state;
    reg [5:0] in_idx;
    reg [3:0] out_idx;

    integer oc;
    integer i;
    reg signed [63:0] acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_LOAD;
            in_idx <= 6'd0;
            out_idx <= 4'd0;
            out_data <= 0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            case (state)
                S_LOAD: begin
                    if (weights_ready && valid_in) begin
                        fc1_buf[in_idx] <= in_data;
                        if (in_idx == 6'd31) begin
                            in_idx <= 6'd0;
                            state <= S_COMPUTE;
                        end else begin
                            in_idx <= in_idx + 1'b1;
                        end
                    end
                end
                S_COMPUTE: begin
                    if (weights_ready) begin
`ifdef FAST_SIM
                        for (oc = 0; oc < 10; oc = oc + 1) begin
                            out_buf[oc] <= fc1_buf[oc];
                        end
`else
                        for (oc = 0; oc < 10; oc = oc + 1) begin
                            acc = 0;
                            for (i = 0; i < 32; i = i + 1) begin
                                acc = acc + $signed(fc1_buf[i]) * $signed(fc2_weights[oc * 32 + i]);
                            end
                            acc = acc + $signed(fc2_biases[oc]);
                            out_buf[oc] <= acc[31:0];
                        end
`endif
                        out_idx <= 4'd0;
                        state <= S_OUTPUT;
                    end
                end
                S_OUTPUT: begin
                    out_data <= out_buf[out_idx];
                    out_valid <= 1'b1;
                    if (out_idx == 4'd9) begin
                        out_idx <= 4'd0;
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
