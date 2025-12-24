`timescale 1ns/1ps

module max_pool_2x2 #(
    parameter IN_WIDTH = 24,
    parameter IN_HEIGHT = 24
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,

    input wire signed [31:0] in_ch0,
    input wire signed [31:0] in_ch1,
    input wire signed [31:0] in_ch2,
    input wire signed [31:0] in_ch3,
    input wire signed [31:0] in_ch4,
    input wire signed [31:0] in_ch5,

    output reg signed [31:0] out_ch0,
    output reg signed [31:0] out_ch1,
    output reg signed [31:0] out_ch2,
    output reg signed [31:0] out_ch3,
    output reg signed [31:0] out_ch4,
    output reg signed [31:0] out_ch5,
    output reg out_valid
);

    reg signed [31:0] linebuf_ch0 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch1 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch2 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch3 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch4 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch5 [0:IN_WIDTH-1];

    reg signed [31:0] left_ch0;
    reg signed [31:0] left_ch1;
    reg signed [31:0] left_ch2;
    reg signed [31:0] left_ch3;
    reg signed [31:0] left_ch4;
    reg signed [31:0] left_ch5;

    reg signed [31:0] prev_row_left_ch0;
    reg signed [31:0] prev_row_left_ch1;
    reg signed [31:0] prev_row_left_ch2;
    reg signed [31:0] prev_row_left_ch3;
    reg signed [31:0] prev_row_left_ch4;
    reg signed [31:0] prev_row_left_ch5;

    reg [5:0] col;
    reg [5:0] row;

    integer i;

    function signed [31:0] max2;
        input signed [31:0] a;
        input signed [31:0] b;
        begin
            max2 = (a >= b) ? a : b;
        end
    endfunction

    function signed [31:0] max4;
        input signed [31:0] a;
        input signed [31:0] b;
        input signed [31:0] c;
        input signed [31:0] d;
        begin
            max4 = max2(max2(a, b), max2(c, d));
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col <= 0;
            row <= 0;
            out_valid <= 1'b0;
            out_ch0 <= 0;
            out_ch1 <= 0;
            out_ch2 <= 0;
            out_ch3 <= 0;
            out_ch4 <= 0;
            out_ch5 <= 0;
            left_ch0 <= 0;
            left_ch1 <= 0;
            left_ch2 <= 0;
            left_ch3 <= 0;
            left_ch4 <= 0;
            left_ch5 <= 0;
            prev_row_left_ch0 <= 0;
            prev_row_left_ch1 <= 0;
            prev_row_left_ch2 <= 0;
            prev_row_left_ch3 <= 0;
            prev_row_left_ch4 <= 0;
            prev_row_left_ch5 <= 0;
            for (i = 0; i < IN_WIDTH; i = i + 1) begin
                linebuf_ch0[i] <= 0;
                linebuf_ch1[i] <= 0;
                linebuf_ch2[i] <= 0;
                linebuf_ch3[i] <= 0;
                linebuf_ch4[i] <= 0;
                linebuf_ch5[i] <= 0;
            end
        end else if (valid_in) begin
            // Read previous-row values before overwriting line buffers.
            // These are used for 2x2 max pooling at odd row/col.
            out_valid <= (row[0] == 1'b1) && (col[0] == 1'b1);
            if ((row[0] == 1'b1) && (col[0] == 1'b1)) begin
                out_ch0 <= max4(in_ch0, left_ch0, linebuf_ch0[col], prev_row_left_ch0);
                out_ch1 <= max4(in_ch1, left_ch1, linebuf_ch1[col], prev_row_left_ch1);
                out_ch2 <= max4(in_ch2, left_ch2, linebuf_ch2[col], prev_row_left_ch2);
                out_ch3 <= max4(in_ch3, left_ch3, linebuf_ch3[col], prev_row_left_ch3);
                out_ch4 <= max4(in_ch4, left_ch4, linebuf_ch4[col], prev_row_left_ch4);
                out_ch5 <= max4(in_ch5, left_ch5, linebuf_ch5[col], prev_row_left_ch5);
            end else begin
                out_ch0 <= out_ch0;
                out_ch1 <= out_ch1;
                out_ch2 <= out_ch2;
                out_ch3 <= out_ch3;
                out_ch4 <= out_ch4;
                out_ch5 <= out_ch5;
            end

            // Capture previous-row values for next column.
            prev_row_left_ch0 <= linebuf_ch0[col];
            prev_row_left_ch1 <= linebuf_ch1[col];
            prev_row_left_ch2 <= linebuf_ch2[col];
            prev_row_left_ch3 <= linebuf_ch3[col];
            prev_row_left_ch4 <= linebuf_ch4[col];
            prev_row_left_ch5 <= linebuf_ch5[col];

            // Update line buffers with current row.
            linebuf_ch0[col] <= in_ch0;
            linebuf_ch1[col] <= in_ch1;
            linebuf_ch2[col] <= in_ch2;
            linebuf_ch3[col] <= in_ch3;
            linebuf_ch4[col] <= in_ch4;
            linebuf_ch5[col] <= in_ch5;

            // Track current row left values.
            left_ch0 <= in_ch0;
            left_ch1 <= in_ch1;
            left_ch2 <= in_ch2;
            left_ch3 <= in_ch3;
            left_ch4 <= in_ch4;
            left_ch5 <= in_ch5;

            if (col == IN_WIDTH - 1) begin
                col <= 0;
                if (row == IN_HEIGHT - 1)
                    row <= 0;
                else
                    row <= row + 1'b1;
            end else begin
                col <= col + 1'b1;
            end
        end else begin
            out_valid <= 1'b0;
        end
    end

endmodule
