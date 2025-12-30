`timescale 1ns/1ps

module tb_layer1_block_wave;
    localparam integer IMG_PIXELS = 784;
    localparam integer MAX_WARMUP = 200000;
    localparam integer MAX_WAIT = 1000000;

    reg clk;
    reg rst_n;
    reg valid_in;
    reg [7:0] pixel_in;

    wire signed [7:0] result_ch0;
    wire signed [7:0] result_ch1;
    wire signed [7:0] result_ch2;
    wire signed [7:0] result_ch3;
    wire signed [7:0] result_ch4;
    wire signed [7:0] result_ch5;
    wire result_valid;

    reg [7:0] image [0:IMG_PIXELS-1];
    integer idx;
    integer out_count;
    integer wait_cycles;
    integer fh;

    layer1_block dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .valid_in     (valid_in),
        .pixel_in     (pixel_in),
        .result_ch0   (result_ch0),
        .result_ch1   (result_ch1),
        .result_ch2   (result_ch2),
        .result_ch3   (result_ch3),
        .result_ch4   (result_ch4),
        .result_ch5   (result_ch5),
        .result_valid (result_valid)
    );

    always #10 clk = ~clk;

    initial begin
        fh = $fopen("tb/test_image.mem", "r");
        if (fh != 0) begin
            $fclose(fh);
            $readmemh("tb/test_image.mem", image);
            $display("Loaded image from tb/test_image.mem");
        end else begin
            for (idx = 0; idx < IMG_PIXELS; idx = idx + 1)
                image[idx] = idx[7:0];
            $display("Using synthetic image pattern");
        end
    end

    initial begin
        $dumpfile("tb_layer1_block_wave.vcd");
        $dumpvars(0, clk, rst_n, valid_in, pixel_in, result_valid,
                  result_ch0, result_ch1, result_ch2, result_ch3, result_ch4, result_ch5);
        $dumpvars(0, dut.load_state, dut.c1_state, dut.pos_x, dut.pos_y, dut.oc_idx,
                  dut.k_x, dut.k_y, dut.mac_phase, dut.img_rd_addr, dut.img_rd_data,
                  dut.img_wr_addr, dut.img_wr_en, dut.conv1_weight_addr, dut.conv1_weight_q);
        $dumpoff;

        clk = 0;
        rst_n = 0;
        valid_in = 0;
        pixel_in = 0;
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
        for (idx = 0; idx < IMG_PIXELS; idx = idx + 1) begin
            @(negedge clk);
            valid_in = 1;
            pixel_in = image[idx];
        end
        @(negedge clk);
        valid_in = 0;
        pixel_in = 0;

        wait_cycles = 0;
        while ((out_count < 4) && (wait_cycles < MAX_WAIT)) begin
            @(posedge clk);
            if (result_valid) begin
                $display("OUT%0d: %0d %0d %0d %0d %0d %0d",
                         out_count, result_ch0, result_ch1, result_ch2,
                         result_ch3, result_ch4, result_ch5);
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
