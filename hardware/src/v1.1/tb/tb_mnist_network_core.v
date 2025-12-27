`timescale 1ns/1ps

module tb_mnist_network_core;
    localparam integer IMG_WIDTH = 28;
    localparam integer IMG_HEIGHT = 28;
    localparam integer DATA_WIDTH = 8;
    localparam integer IMG_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    localparam integer OUT_PIXELS = 10;
    localparam integer MAX_SIM_CYCLES = 5000000;
    localparam integer MAX_WARMUP = 200000;
    localparam integer FEED_PIXELS = IMG_PIXELS;

    reg clk;
    reg rst_n;
    reg valid_in;
    reg [DATA_WIDTH-1:0] pixel_in;

    wire [31:0] result;
    wire result_valid;

    reg [DATA_WIDTH-1:0] image [0:IMG_PIXELS-1];

    integer idx;
    integer output_count;
    integer sim_cycles;
    integer vin_count;
    integer l1_count;
    integer l2_count;
    integer l3_count;
    integer l4_count;
    reg signed [31:0] out_vec [0:9];
    integer out_idx;
    reg signed [31:0] max_val;
    integer max_idx;
    reg signed [31:0] fc1_vec [0:31];
    integer fc1_idx;
    reg signed [31:0] pool1_stream [0:15][0:5];
    integer pool1_idx;
    reg signed [31:0] pool2_stream [0:15][0:15];
    integer pool2_idx;

    mnist_network_core dut (
        .clk (clk),
        .rst_n (rst_n),
        .valid_in (valid_in),
        .pixel_in (pixel_in),
        .result (result),
        .result_valid (result_valid)
    );

    always #10 clk = ~clk;

    integer fh;

    initial begin
        fh = $fopen("tb/test_image.mem", "r");
        if (fh != 0) begin
            $fclose(fh);
            $readmemh("tb/test_image.mem", image);
            $display("Loaded image from tb/test_image.mem");
        end else begin
            for (idx = 0; idx < IMG_PIXELS; idx = idx + 1) begin
                image[idx] = (idx * 3) & 8'hFF;
            end
            $display("Using synthetic image pattern");
        end
    end

    initial begin
        $dumpfile("tb_mnist_network_core.vcd");
        $dumpvars(0, dut);
        $dumpvars(0, clk, rst_n, valid_in, pixel_in, result, result_valid);
        $dumpoff;

        clk = 0;
        rst_n = 0;
        valid_in = 0;
        pixel_in = 0;
        output_count = 0;
        sim_cycles = 0;
        vin_count = 0;
        l1_count = 0;
        l2_count = 0;
        l3_count = 0;
        l4_count = 0;
        out_idx = 0;
        max_val = 0;
        max_idx = 0;
        fc1_idx = 0;
        pool1_idx = 0;
        pool2_idx = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;

        // Wait for all layer weight loaders to complete (with timeout)
        idx = 0;
        while (!(dut.u_layer1.load_state == 2'd2 &&
                 dut.u_layer2.load_state == 2'd2 &&
                 dut.u_layer3.load_state == 2'd2 &&
                 dut.u_layer4.load_state == 2'd2) &&
               (idx < MAX_WARMUP)) begin
            @(posedge clk);
            idx = idx + 1;
        end
        if (idx >= MAX_WARMUP) begin
            $display("WARN: weight load timeout after %0d cycles", MAX_WARMUP);
        end
        $dumpon;

        for (idx = 0; idx < FEED_PIXELS; idx = idx + 1) begin
            @(negedge clk);
            valid_in = 1;
            pixel_in = image[idx];
        end

        @(negedge clk);
        valid_in = 0;
        pixel_in = 0;

        wait (output_count == OUT_PIXELS);
        #40;
        $display("Pool1 (first 4 positions, ch0..5):");
        for (idx = 0; idx < 4; idx = idx + 1) begin
            $display("  pos%0d: %0d %0d %0d %0d %0d %0d",
                     idx,
                     pool1_stream[idx][0], pool1_stream[idx][1], pool1_stream[idx][2],
                     pool1_stream[idx][3], pool1_stream[idx][4], pool1_stream[idx][5]);
        end

        $display("Pool2 (first 4 positions, ch0..15):");
        for (idx = 0; idx < 4; idx = idx + 1) begin
            $display("  pos%0d: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                     idx,
                     pool2_stream[idx][0], pool2_stream[idx][1], pool2_stream[idx][2], pool2_stream[idx][3],
                     pool2_stream[idx][4], pool2_stream[idx][5], pool2_stream[idx][6], pool2_stream[idx][7],
                     pool2_stream[idx][8], pool2_stream[idx][9], pool2_stream[idx][10], pool2_stream[idx][11],
                     pool2_stream[idx][12], pool2_stream[idx][13], pool2_stream[idx][14], pool2_stream[idx][15]);
        end

        $display("FC1 outputs:");
        for (idx = 0; idx < 32; idx = idx + 1) begin
            $display("  [%0d] = %0d (0x%08h)", idx, fc1_vec[idx], fc1_vec[idx]);
        end

        $display("FC2 outputs:");
        for (idx = 0; idx < OUT_PIXELS; idx = idx + 1) begin
            $display("  [%0d] = %0d (0x%08h)", idx, out_vec[idx], out_vec[idx]);
        end
        max_val = out_vec[0];
        max_idx = 0;
        for (idx = 1; idx < OUT_PIXELS; idx = idx + 1) begin
            if (out_vec[idx] > max_val) begin
                max_val = out_vec[idx];
                max_idx = idx;
            end
        end
        $display("PRED = %0d (max=%0d)", max_idx, max_val);
        $display("PASS: %0d outputs checked", OUT_PIXELS);
        $dumpoff;
        $finish;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_count <= 0;
            vin_count <= 0;
            l1_count <= 0;
            l2_count <= 0;
            l3_count <= 0;
            l4_count <= 0;
            out_idx <= 0;
            fc1_idx <= 0;
            pool1_idx <= 0;
            pool2_idx <= 0;
        end else if (result_valid) begin
            output_count <= output_count + 1;
            out_vec[out_idx] <= result;
            out_idx <= out_idx + 1;
        end
        if (rst_n) begin
            if (valid_in)
                vin_count <= vin_count + 1;
            if (dut.u_layer1.result_valid)
                l1_count <= l1_count + 1;
            if (dut.u_layer2.out_valid)
                l2_count <= l2_count + 1;
            if (dut.u_layer3.out_valid)
                l3_count <= l3_count + 1;
            if (dut.u_layer4.out_valid)
                l4_count <= l4_count + 1;
            if (dut.u_layer1.result_valid && pool1_idx < 16) begin
                pool1_stream[pool1_idx][0] <= dut.u_layer1.result_ch0;
                pool1_stream[pool1_idx][1] <= dut.u_layer1.result_ch1;
                pool1_stream[pool1_idx][2] <= dut.u_layer1.result_ch2;
                pool1_stream[pool1_idx][3] <= dut.u_layer1.result_ch3;
                pool1_stream[pool1_idx][4] <= dut.u_layer1.result_ch4;
                pool1_stream[pool1_idx][5] <= dut.u_layer1.result_ch5;
                pool1_idx <= pool1_idx + 1;
            end
            if (dut.u_layer2.out_valid && pool2_idx < 16) begin
                pool2_stream[pool2_idx][0] <= dut.u_layer2.out_ch0;
                pool2_stream[pool2_idx][1] <= dut.u_layer2.out_ch1;
                pool2_stream[pool2_idx][2] <= dut.u_layer2.out_ch2;
                pool2_stream[pool2_idx][3] <= dut.u_layer2.out_ch3;
                pool2_stream[pool2_idx][4] <= dut.u_layer2.out_ch4;
                pool2_stream[pool2_idx][5] <= dut.u_layer2.out_ch5;
                pool2_stream[pool2_idx][6] <= dut.u_layer2.out_ch6;
                pool2_stream[pool2_idx][7] <= dut.u_layer2.out_ch7;
                pool2_stream[pool2_idx][8] <= dut.u_layer2.out_ch8;
                pool2_stream[pool2_idx][9] <= dut.u_layer2.out_ch9;
                pool2_stream[pool2_idx][10] <= dut.u_layer2.out_ch10;
                pool2_stream[pool2_idx][11] <= dut.u_layer2.out_ch11;
                pool2_stream[pool2_idx][12] <= dut.u_layer2.out_ch12;
                pool2_stream[pool2_idx][13] <= dut.u_layer2.out_ch13;
                pool2_stream[pool2_idx][14] <= dut.u_layer2.out_ch14;
                pool2_stream[pool2_idx][15] <= dut.u_layer2.out_ch15;
                pool2_idx <= pool2_idx + 1;
            end
            if (dut.u_layer3.out_valid && fc1_idx < 32) begin
                fc1_vec[fc1_idx] <= dut.u_layer3.out_data;
                fc1_idx <= fc1_idx + 1;
            end
        end
    end

    always @(posedge clk) begin
        sim_cycles <= sim_cycles + 1;
        if ((sim_cycles % 50000) == 0) begin
            $display("T=%0t rst_n=%b l1=%0d l2=%0d l3=%0d l4=%0d vin=%0d l1v=%0d l2v=%0d l3v=%0d l4v=%0d",
                     $time, rst_n,
                     dut.u_layer1.load_state,
                     dut.u_layer2.load_state,
                     dut.u_layer3.load_state,
                     dut.u_layer4.load_state,
                     vin_count, l1_count, l2_count, l3_count, l4_count);
        end
        if (sim_cycles >= MAX_SIM_CYCLES) begin
            $display("TIMEOUT after %0d cycles, outputs=%0d vin=%0d l1v=%0d l2v=%0d l3v=%0d l4v=%0d",
                     sim_cycles, output_count, vin_count, l1_count, l2_count, l3_count, l4_count);
            $finish;
        end
    end

endmodule
