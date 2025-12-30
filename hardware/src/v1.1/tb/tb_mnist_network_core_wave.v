`timescale 1ns/1ps

module tb_mnist_network_core_wave;
    localparam integer IMG_WIDTH = 28;
    localparam integer IMG_HEIGHT = 28;
    localparam integer DATA_WIDTH = 8;
    localparam integer IMG_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    localparam integer OUT_PIXELS = 10;
    localparam integer MAX_WARMUP = 200000;
    localparam integer MAX_WAIT = 5000000;

    reg clk;
    reg rst_n;
    reg valid_in;
    reg [DATA_WIDTH-1:0] pixel_in;

    wire [31:0] result;
    wire result_valid;

    reg [DATA_WIDTH-1:0] image [0:IMG_PIXELS-1];
    integer idx;
    integer out_count;
    integer wait_cycles;
    integer fh;

    mnist_network_core dut (
        .clk (clk),
        .rst_n (rst_n),
        .valid_in (valid_in),
        .pixel_in (pixel_in),
        .result (result),
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
                image[idx] = (idx * 3) & 8'hFF;
            $display("Using synthetic image pattern");
        end
    end

    initial begin
        $dumpfile("tb_mnist_network_core_wave.vcd");
        $dumpvars(0, clk, rst_n, valid_in, pixel_in, result, result_valid);
        $dumpvars(0, dut.u_layer1.result_valid, dut.u_layer2.out_valid,
                  dut.u_layer3.out_valid, dut.u_layer4.out_valid);
        $dumpvars(0, dut.u_layer1.load_state, dut.u_layer2.load_state,
                  dut.u_layer3.load_state, dut.u_layer4.load_state);
        $dumpoff;

        clk = 0;
        rst_n = 0;
        valid_in = 0;
        pixel_in = 0;
        out_count = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;

        idx = 0;
        while (!(dut.u_layer1.load_state == 1'b1 &&
                 dut.u_layer2.load_state == 1'b1 &&
                 dut.u_layer3.load_state == 1'b1 &&
                 dut.u_layer4.load_state == 1'b1) &&
               (idx < MAX_WARMUP)) begin
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
        while ((out_count < OUT_PIXELS) && (wait_cycles < MAX_WAIT)) begin
            @(posedge clk);
            if (result_valid) begin
                $display("OUT%0d: %0d", out_count, result);
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
