`timescale 1ns/1ps

module scaler_quantize #(
    parameter MULT_VAL = 1,
    parameter SHIFT_VAL = 16
)(
    input wire signed [31:0] acc_in,
    output reg signed [7:0] q_out
);

    wire signed [63:0] product = acc_in * $signed(MULT_VAL);
    wire signed [63:0] rounded = product + $signed(1 << (SHIFT_VAL - 1));
    wire signed [63:0] shifted = rounded >>> SHIFT_VAL;

    always @(*) begin
        if (shifted > 127)
            q_out = 8'sd127;
        else if (shifted < -128)
            q_out = -8'sd128;
        else
            q_out = shifted[7:0];
    end

endmodule
