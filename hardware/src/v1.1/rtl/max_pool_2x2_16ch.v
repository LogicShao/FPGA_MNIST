`timescale 1ns/1ps

module max_pool_2x2_16ch #(
    parameter IN_WIDTH = 8,
    parameter IN_HEIGHT = 8
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
    input wire signed [31:0] in_ch6,
    input wire signed [31:0] in_ch7,
    input wire signed [31:0] in_ch8,
    input wire signed [31:0] in_ch9,
    input wire signed [31:0] in_ch10,
    input wire signed [31:0] in_ch11,
    input wire signed [31:0] in_ch12,
    input wire signed [31:0] in_ch13,
    input wire signed [31:0] in_ch14,
    input wire signed [31:0] in_ch15,

    output reg signed [31:0] out_ch0,
    output reg signed [31:0] out_ch1,
    output reg signed [31:0] out_ch2,
    output reg signed [31:0] out_ch3,
    output reg signed [31:0] out_ch4,
    output reg signed [31:0] out_ch5,
    output reg signed [31:0] out_ch6,
    output reg signed [31:0] out_ch7,
    output reg signed [31:0] out_ch8,
    output reg signed [31:0] out_ch9,
    output reg signed [31:0] out_ch10,
    output reg signed [31:0] out_ch11,
    output reg signed [31:0] out_ch12,
    output reg signed [31:0] out_ch13,
    output reg signed [31:0] out_ch14,
    output reg signed [31:0] out_ch15,
    output reg out_valid
);

    reg signed [31:0] linebuf_ch0 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch1 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch2 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch3 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch4 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch5 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch6 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch7 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch8 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch9 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch10 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch11 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch12 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch13 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch14 [0:IN_WIDTH-1];
    reg signed [31:0] linebuf_ch15 [0:IN_WIDTH-1];

    reg signed [31:0] left_ch0;
    reg signed [31:0] left_ch1;
    reg signed [31:0] left_ch2;
    reg signed [31:0] left_ch3;
    reg signed [31:0] left_ch4;
    reg signed [31:0] left_ch5;
    reg signed [31:0] left_ch6;
    reg signed [31:0] left_ch7;
    reg signed [31:0] left_ch8;
    reg signed [31:0] left_ch9;
    reg signed [31:0] left_ch10;
    reg signed [31:0] left_ch11;
    reg signed [31:0] left_ch12;
    reg signed [31:0] left_ch13;
    reg signed [31:0] left_ch14;
    reg signed [31:0] left_ch15;

    reg signed [31:0] prev_row_left_ch0;
    reg signed [31:0] prev_row_left_ch1;
    reg signed [31:0] prev_row_left_ch2;
    reg signed [31:0] prev_row_left_ch3;
    reg signed [31:0] prev_row_left_ch4;
    reg signed [31:0] prev_row_left_ch5;
    reg signed [31:0] prev_row_left_ch6;
    reg signed [31:0] prev_row_left_ch7;
    reg signed [31:0] prev_row_left_ch8;
    reg signed [31:0] prev_row_left_ch9;
    reg signed [31:0] prev_row_left_ch10;
    reg signed [31:0] prev_row_left_ch11;
    reg signed [31:0] prev_row_left_ch12;
    reg signed [31:0] prev_row_left_ch13;
    reg signed [31:0] prev_row_left_ch14;
    reg signed [31:0] prev_row_left_ch15;

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
            out_ch0 <= 0; out_ch1 <= 0; out_ch2 <= 0; out_ch3 <= 0;
            out_ch4 <= 0; out_ch5 <= 0; out_ch6 <= 0; out_ch7 <= 0;
            out_ch8 <= 0; out_ch9 <= 0; out_ch10 <= 0; out_ch11 <= 0;
            out_ch12 <= 0; out_ch13 <= 0; out_ch14 <= 0; out_ch15 <= 0;
            left_ch0 <= 0; left_ch1 <= 0; left_ch2 <= 0; left_ch3 <= 0;
            left_ch4 <= 0; left_ch5 <= 0; left_ch6 <= 0; left_ch7 <= 0;
            left_ch8 <= 0; left_ch9 <= 0; left_ch10 <= 0; left_ch11 <= 0;
            left_ch12 <= 0; left_ch13 <= 0; left_ch14 <= 0; left_ch15 <= 0;
            prev_row_left_ch0 <= 0; prev_row_left_ch1 <= 0; prev_row_left_ch2 <= 0; prev_row_left_ch3 <= 0;
            prev_row_left_ch4 <= 0; prev_row_left_ch5 <= 0; prev_row_left_ch6 <= 0; prev_row_left_ch7 <= 0;
            prev_row_left_ch8 <= 0; prev_row_left_ch9 <= 0; prev_row_left_ch10 <= 0; prev_row_left_ch11 <= 0;
            prev_row_left_ch12 <= 0; prev_row_left_ch13 <= 0; prev_row_left_ch14 <= 0; prev_row_left_ch15 <= 0;
            for (i = 0; i < IN_WIDTH; i = i + 1) begin
                linebuf_ch0[i] <= 0; linebuf_ch1[i] <= 0; linebuf_ch2[i] <= 0; linebuf_ch3[i] <= 0;
                linebuf_ch4[i] <= 0; linebuf_ch5[i] <= 0; linebuf_ch6[i] <= 0; linebuf_ch7[i] <= 0;
                linebuf_ch8[i] <= 0; linebuf_ch9[i] <= 0; linebuf_ch10[i] <= 0; linebuf_ch11[i] <= 0;
                linebuf_ch12[i] <= 0; linebuf_ch13[i] <= 0; linebuf_ch14[i] <= 0; linebuf_ch15[i] <= 0;
            end
        end else if (valid_in) begin
            out_valid <= (row[0] == 1'b1) && (col[0] == 1'b1);
            if ((row[0] == 1'b1) && (col[0] == 1'b1)) begin
                out_ch0 <= max4(in_ch0, left_ch0, linebuf_ch0[col], prev_row_left_ch0);
                out_ch1 <= max4(in_ch1, left_ch1, linebuf_ch1[col], prev_row_left_ch1);
                out_ch2 <= max4(in_ch2, left_ch2, linebuf_ch2[col], prev_row_left_ch2);
                out_ch3 <= max4(in_ch3, left_ch3, linebuf_ch3[col], prev_row_left_ch3);
                out_ch4 <= max4(in_ch4, left_ch4, linebuf_ch4[col], prev_row_left_ch4);
                out_ch5 <= max4(in_ch5, left_ch5, linebuf_ch5[col], prev_row_left_ch5);
                out_ch6 <= max4(in_ch6, left_ch6, linebuf_ch6[col], prev_row_left_ch6);
                out_ch7 <= max4(in_ch7, left_ch7, linebuf_ch7[col], prev_row_left_ch7);
                out_ch8 <= max4(in_ch8, left_ch8, linebuf_ch8[col], prev_row_left_ch8);
                out_ch9 <= max4(in_ch9, left_ch9, linebuf_ch9[col], prev_row_left_ch9);
                out_ch10 <= max4(in_ch10, left_ch10, linebuf_ch10[col], prev_row_left_ch10);
                out_ch11 <= max4(in_ch11, left_ch11, linebuf_ch11[col], prev_row_left_ch11);
                out_ch12 <= max4(in_ch12, left_ch12, linebuf_ch12[col], prev_row_left_ch12);
                out_ch13 <= max4(in_ch13, left_ch13, linebuf_ch13[col], prev_row_left_ch13);
                out_ch14 <= max4(in_ch14, left_ch14, linebuf_ch14[col], prev_row_left_ch14);
                out_ch15 <= max4(in_ch15, left_ch15, linebuf_ch15[col], prev_row_left_ch15);
            end

            prev_row_left_ch0 <= linebuf_ch0[col];
            prev_row_left_ch1 <= linebuf_ch1[col];
            prev_row_left_ch2 <= linebuf_ch2[col];
            prev_row_left_ch3 <= linebuf_ch3[col];
            prev_row_left_ch4 <= linebuf_ch4[col];
            prev_row_left_ch5 <= linebuf_ch5[col];
            prev_row_left_ch6 <= linebuf_ch6[col];
            prev_row_left_ch7 <= linebuf_ch7[col];
            prev_row_left_ch8 <= linebuf_ch8[col];
            prev_row_left_ch9 <= linebuf_ch9[col];
            prev_row_left_ch10 <= linebuf_ch10[col];
            prev_row_left_ch11 <= linebuf_ch11[col];
            prev_row_left_ch12 <= linebuf_ch12[col];
            prev_row_left_ch13 <= linebuf_ch13[col];
            prev_row_left_ch14 <= linebuf_ch14[col];
            prev_row_left_ch15 <= linebuf_ch15[col];

            linebuf_ch0[col] <= in_ch0;
            linebuf_ch1[col] <= in_ch1;
            linebuf_ch2[col] <= in_ch2;
            linebuf_ch3[col] <= in_ch3;
            linebuf_ch4[col] <= in_ch4;
            linebuf_ch5[col] <= in_ch5;
            linebuf_ch6[col] <= in_ch6;
            linebuf_ch7[col] <= in_ch7;
            linebuf_ch8[col] <= in_ch8;
            linebuf_ch9[col] <= in_ch9;
            linebuf_ch10[col] <= in_ch10;
            linebuf_ch11[col] <= in_ch11;
            linebuf_ch12[col] <= in_ch12;
            linebuf_ch13[col] <= in_ch13;
            linebuf_ch14[col] <= in_ch14;
            linebuf_ch15[col] <= in_ch15;

            left_ch0 <= in_ch0;
            left_ch1 <= in_ch1;
            left_ch2 <= in_ch2;
            left_ch3 <= in_ch3;
            left_ch4 <= in_ch4;
            left_ch5 <= in_ch5;
            left_ch6 <= in_ch6;
            left_ch7 <= in_ch7;
            left_ch8 <= in_ch8;
            left_ch9 <= in_ch9;
            left_ch10 <= in_ch10;
            left_ch11 <= in_ch11;
            left_ch12 <= in_ch12;
            left_ch13 <= in_ch13;
            left_ch14 <= in_ch14;
            left_ch15 <= in_ch15;

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
