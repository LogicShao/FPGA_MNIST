`timescale 1ns/1ps

module tb_layer2_block_wave;
    localparam integer FEAT_PIXELS = 144;
    localparam integer MAX_WARMUP = 200000;
    localparam integer MAX_WAIT = 3000000;

    reg clk;
    reg rst_n;
    reg valid_in;
    reg signed [7:0] in_ch0;
    reg signed [7:0] in_ch1;
    reg signed [7:0] in_ch2;
    reg signed [7:0] in_ch3;
    reg signed [7:0] in_ch4;
    reg signed [7:0] in_ch5;

    wire signed [7:0] out_ch0;
    wire signed [7:0] out_ch1;
    wire signed [7:0] out_ch2;
    wire signed [7:0] out_ch3;
    wire signed [7:0] out_ch4;
    wire signed [7:0] out_ch5;
    wire signed [7:0] out_ch6;
    wire signed [7:0] out_ch7;
    wire signed [7:0] out_ch8;
    wire signed [7:0] out_ch9;
    wire signed [7:0] out_ch10;
    wire signed [7:0] out_ch11;
    wire signed [7:0] out_ch12;
    wire signed [7:0] out_ch13;
    wire signed [7:0] out_ch14;
    wire signed [7:0] out_ch15;
    wire out_valid;

    integer idx;
    integer out_count;
    integer wait_cycles;
    integer v;

    layer2_block dut (
        .clk     (clk),
        .rst_n   (rst_n),
        .valid_in(valid_in),
        .in_ch0  (in_ch0),
        .in_ch1  (in_ch1),
        .in_ch2  (in_ch2),
        .in_ch3  (in_ch3),
        .in_ch4  (in_ch4),
        .in_ch5  (in_ch5),
        .out_ch0 (out_ch0),
        .out_ch1 (out_ch1),
        .out_ch2 (out_ch2),
        .out_ch3 (out_ch3),
        .out_ch4 (out_ch4),
        .out_ch5 (out_ch5),
        .out_ch6 (out_ch6),
        .out_ch7 (out_ch7),
        .out_ch8 (out_ch8),
        .out_ch9 (out_ch9),
        .out_ch10(out_ch10),
        .out_ch11(out_ch11),
        .out_ch12(out_ch12),
        .out_ch13(out_ch13),
        .out_ch14(out_ch14),
        .out_ch15(out_ch15),
        .out_valid(out_valid)
    );

    always #10 clk = ~clk;

    initial begin
        $dumpfile("tb_layer2_block_wave.vcd");
        $dumpvars(0, clk, rst_n, valid_in,
                  in_ch0, in_ch1, in_ch2, in_ch3, in_ch4, in_ch5,
                  out_valid,
                  out_ch0, out_ch1, out_ch2, out_ch3, out_ch4, out_ch5,
                  out_ch6, out_ch7, out_ch8, out_ch9, out_ch10, out_ch11,
                  out_ch12, out_ch13, out_ch14, out_ch15);
        $dumpvars(0, dut.load_state, dut.c_state, dut.pos_x, dut.pos_y, dut.oc_idx,
                  dut.k_ch, dut.k_y, dut.k_x, dut.mac_phase, dut.q_idx,
                  dut.feat_rd_addr, dut.feat_rd_data0, dut.feat_wr_addr, dut.feat_wr_en,
                  dut.conv2_weight_addr, dut.conv2_weight_q);
        $dumpoff;

        clk = 0;
        rst_n = 0;
        valid_in = 0;
        in_ch0 = 0;
        in_ch1 = 0;
        in_ch2 = 0;
        in_ch3 = 0;
        in_ch4 = 0;
        in_ch5 = 0;
        out_count = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;

        idx = 0;
        while ((dut.load_state != 1'b1) && (idx < MAX_WARMUP)) begin
            @(posedge clk);
            idx = idx + 1;
        end
        if (idx >= MAX_WARMUP)
            $display("WARN: weight load timeout after %0d cycles", MAX_WARMUP);

        $dumpon;
        for (idx = 0; idx < FEAT_PIXELS; idx = idx + 1) begin
            @(negedge clk);
            valid_in = 1;
            v = (idx % 32) - 16;
            in_ch0 = v;
            in_ch1 = v + 1;
            in_ch2 = v + 2;
            in_ch3 = v + 3;
            in_ch4 = v + 4;
            in_ch5 = v + 5;
        end
        @(negedge clk);
        valid_in = 0;
        in_ch0 = 0;
        in_ch1 = 0;
        in_ch2 = 0;
        in_ch3 = 0;
        in_ch4 = 0;
        in_ch5 = 0;

        wait_cycles = 0;
        while ((out_count < 4) && (wait_cycles < MAX_WAIT)) begin
            @(posedge clk);
            if (out_valid) begin
                $display("OUT%0d: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                         out_count,
                         out_ch0, out_ch1, out_ch2, out_ch3,
                         out_ch4, out_ch5, out_ch6, out_ch7,
                         out_ch8, out_ch9, out_ch10, out_ch11,
                         out_ch12, out_ch13, out_ch14, out_ch15);
                out_count = out_count + 1;
            end
            wait_cycles = wait_cycles + 1;
        end
        if (wait_cycles >= MAX_WAIT)
            $display("WARN: output wait timeout after %0d cycles", MAX_WAIT);

        $dumpoff;
        $finish;
    end
endmodule
