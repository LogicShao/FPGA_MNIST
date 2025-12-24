`timescale 1ns/1ps

module layer4_fc2(
    input wire clk,
    input wire rst_n,
    input wire valid_in,
    input wire [31:0] in_data,
    output wire [31:0] out_data,
    output wire out_valid
);

    // Placeholder data path: pass through input for now.
    // We still load weights/biases here to encapsulate parameters.

    wire signed [7:0] fc2_weight_q;
    wire signed [7:0] fc2_bias_q;

    reg [8:0] fc2_weight_addr;
    reg [3:0] fc2_bias_addr;

    reg signed [7:0] fc2_weights [0:319];
    reg signed [7:0] fc2_biases [0:9];

    localparam LOAD_WEIGHTS = 2'd0;
    localparam LOAD_BIAS    = 2'd1;
    localparam LOAD_DONE    = 2'd2;

    reg [1:0] load_state;
    reg [8:0] load_idx;
    reg       load_capture;

    reg [3:0] bias_idx;
    reg       bias_capture;

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

    rom_FC2_BIASES #(
        .ADDR_WIDTH(4),
        .DATA_WIDTH(8),
        .DEPTH(10),
        .MEM_FILE("rtl/weights/FC2_BIASES.mem")
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
            bias_idx <= 4'd0;
            bias_capture <= 1'b0;
        end else begin
            case (load_state)
                LOAD_WEIGHTS: begin
                    if (load_capture && load_idx != 0) begin
                        fc2_weights[load_idx - 1] <= fc2_weight_q;
                    end
                    if (load_idx < 320) begin
                        fc2_weight_addr <= load_idx;
                        load_idx <= load_idx + 1'b1;
                        load_capture <= 1'b1;
                    end else begin
                        load_capture <= 1'b0;
                        load_state <= LOAD_BIAS;
                        bias_idx <= 4'd0;
                        bias_capture <= 1'b0;
                    end
                end
                LOAD_BIAS: begin
                    if (bias_capture && bias_idx != 0) begin
                        fc2_biases[bias_idx - 1] <= fc2_bias_q;
                    end
                    if (bias_idx < 10) begin
                        fc2_bias_addr <= bias_idx;
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
