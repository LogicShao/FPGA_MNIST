`timescale 1ns/1ps

module tb_mnist_network_core;
    localparam integer IMG_WIDTH = 28;
    localparam integer IMG_HEIGHT = 28;
    localparam integer DATA_WIDTH = 8;
    localparam integer IMG_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    localparam integer OUT_W = (IMG_WIDTH - 4) / 2;
    localparam integer OUT_H = (IMG_HEIGHT - 4) / 2;
    localparam integer OUT_PIXELS = OUT_W * OUT_H;

    reg clk;
    reg rst_n;
    reg valid_in;
    reg [DATA_WIDTH-1:0] pixel_in;

    wire [31:0] result;
    wire result_valid;

    reg [DATA_WIDTH-1:0] image [0:IMG_PIXELS-1];

    integer idx;
    integer output_count;

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
        for (idx = 0; idx < IMG_PIXELS; idx = idx + 1) begin
            image[idx] = (idx * 3) & 8'hFF;
        end
    end

    initial begin
        $dumpfile("tb_mnist_network_core.vcd");
        $dumpvars(0, tb_mnist_network_core);

        clk = 0;
        rst_n = 0;
        valid_in = 0;
        pixel_in = 0;
        output_count = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;

        // Wait for all layer weight loaders to complete
        repeat (12000) @(posedge clk);

        for (idx = 0; idx < IMG_PIXELS; idx = idx + 1) begin
            @(negedge clk);
            valid_in = 1;
            pixel_in = image[idx];
        end

        @(negedge clk);
        valid_in = 0;
        pixel_in = 0;

        wait (output_count == OUT_PIXELS);
        #40;
        $display("PASS: %0d outputs checked", OUT_PIXELS);
        $finish;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_count <= 0;
        end else if (result_valid) begin
            output_count <= output_count + 1;
        end
    end

endmodule
