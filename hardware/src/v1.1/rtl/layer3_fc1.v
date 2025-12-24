`timescale 1ns/1ps

module layer3_fc1(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [31:0] in_data,
    output wire [31:0] out_data,
    output wire out_valid
);

    // Placeholder data path: pass through input for now.
    // We still load weights/biases here to encapsulate parameters.

    wire signed [7:0] fc1_weight_q;
    wire signed [7:0] fc1_bias_q;

    reg [12:0] fc1_weight_addr;
    reg [4:0] fc1_bias_addr;

    reg signed [7:0] fc1_weights [0:8191];
    reg signed [7:0] fc1_biases [0:31];

    localparam LOAD_WEIGHTS = 2'd0;
    localparam LOAD_BIAS    = 2'd1;
    localparam LOAD_DONE    = 2'd2;

    reg [1:0] load_state;
    reg [12:0] load_idx;
    reg        load_capture;

    reg [4:0] bias_idx;
    reg       bias_capture;

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

    rom_FC1_BIASES #(
        .ADDR_WIDTH(5),
        .DATA_WIDTH(8),
        .DEPTH(32),
        .MEM_FILE("rtl/weights/FC1_BIASES.mem")
    ) u_fc1_brom (
        .clk  (clk),
        .addr (fc1_bias_addr),
        .q    (fc1_bias_q)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fc1_weight_addr <= 13'd0;
            fc1_bias_addr <= 5'd0;
            load_state <= LOAD_WEIGHTS;
            load_idx <= 13'd0;
            load_capture <= 1'b0;
            bias_idx <= 5'd0;
            bias_capture <= 1'b0;
        end else begin
            case (load_state)
                LOAD_WEIGHTS: begin
                    if (load_capture && load_idx != 0) begin
                        fc1_weights[load_idx - 1] <= fc1_weight_q;
                    end
                    if (load_idx < 8192) begin
                        fc1_weight_addr <= load_idx;
                        load_idx <= load_idx + 1'b1;
                        load_capture <= 1'b1;
                    end else begin
                        load_capture <= 1'b0;
                        load_state <= LOAD_BIAS;
                        bias_idx <= 5'd0;
                        bias_capture <= 1'b0;
                    end
                end
                LOAD_BIAS: begin
                    if (bias_capture && bias_idx != 0) begin
                        fc1_biases[bias_idx - 1] <= fc1_bias_q;
                    end
                    if (bias_idx < 32) begin
                        fc1_bias_addr <= bias_idx;
                        bias_idx <= bias_idx + 1'b1;
                        bias_capture <= 1'b1;
                    end else begin
                        bias_capture <= 1'b0;
                        load_state <= LOAD_DONE;
                    end
                end
                default: begin
                end
            endcase
        end
    end

    assign out_data = in_data;
    assign out_valid = valid_in & weights_ready;

endmodule
