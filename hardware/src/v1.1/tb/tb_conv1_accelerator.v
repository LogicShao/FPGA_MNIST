`timescale 1ns/1ps

module tb_conv1_accelerator;
    localparam integer IMG_WIDTH = 28;
    localparam integer IMG_HEIGHT = 28;
    localparam integer DATA_WIDTH = 8;
    localparam integer WEIGHT_WIDTH = 8;
    localparam integer OUT_WIDTH = 32;
    localparam integer OUT_W = IMG_WIDTH - 4;
    localparam integer OUT_H = IMG_HEIGHT - 4;
    localparam integer IMG_PIXELS = IMG_WIDTH * IMG_HEIGHT;
    localparam integer OUT_PIXELS = OUT_W * OUT_H;

    localparam signed [WEIGHT_WIDTH-1:0] K00 = 8'sd1;
    localparam signed [WEIGHT_WIDTH-1:0] K01 = 8'sd2;
    localparam signed [WEIGHT_WIDTH-1:0] K02 = 8'sd3;
    localparam signed [WEIGHT_WIDTH-1:0] K03 = 8'sd4;
    localparam signed [WEIGHT_WIDTH-1:0] K04 = 8'sd5;
    localparam signed [WEIGHT_WIDTH-1:0] K10 = 8'sd6;
    localparam signed [WEIGHT_WIDTH-1:0] K11 = 8'sd7;
    localparam signed [WEIGHT_WIDTH-1:0] K12 = 8'sd8;
    localparam signed [WEIGHT_WIDTH-1:0] K13 = 8'sd9;
    localparam signed [WEIGHT_WIDTH-1:0] K14 = 8'sd10;
    localparam signed [WEIGHT_WIDTH-1:0] K20 = 8'sd11;
    localparam signed [WEIGHT_WIDTH-1:0] K21 = 8'sd12;
    localparam signed [WEIGHT_WIDTH-1:0] K22 = 8'sd13;
    localparam signed [WEIGHT_WIDTH-1:0] K23 = 8'sd14;
    localparam signed [WEIGHT_WIDTH-1:0] K24 = 8'sd15;
    localparam signed [WEIGHT_WIDTH-1:0] K30 = 8'sd16;
    localparam signed [WEIGHT_WIDTH-1:0] K31 = 8'sd17;
    localparam signed [WEIGHT_WIDTH-1:0] K32 = 8'sd18;
    localparam signed [WEIGHT_WIDTH-1:0] K33 = 8'sd19;
    localparam signed [WEIGHT_WIDTH-1:0] K34 = 8'sd20;
    localparam signed [WEIGHT_WIDTH-1:0] K40 = 8'sd21;
    localparam signed [WEIGHT_WIDTH-1:0] K41 = 8'sd22;
    localparam signed [WEIGHT_WIDTH-1:0] K42 = 8'sd23;
    localparam signed [WEIGHT_WIDTH-1:0] K43 = 8'sd24;
    localparam signed [WEIGHT_WIDTH-1:0] K44 = 8'sd25;
    localparam signed [WEIGHT_WIDTH-1:0] BIAS = 8'sd0;

    reg clk;
    reg rst_n;
    reg valid_in;
    reg [DATA_WIDTH-1:0] pixel_in;

    wire signed [OUT_WIDTH-1:0] result_ch0;
    wire result_valid;

    reg [DATA_WIDTH-1:0] image [0:IMG_PIXELS-1];
    reg signed [OUT_WIDTH-1:0] expected [0:OUT_PIXELS-1];
    reg signed [WEIGHT_WIDTH-1:0] kernel [0:24];

    integer x;
    integer y;
    integer kx;
    integer ky;
    integer idx;
    integer out_idx;
    integer sum;
    integer output_count;
    integer error_count;

    conv_accelerator #(
        .IMG_WIDTH (IMG_WIDTH),
        .DATA_WIDTH (DATA_WIDTH),
        .WEIGHT_WIDTH (WEIGHT_WIDTH),
        .OUT_WIDTH (OUT_WIDTH)
    ) dut (
        .clk (clk),
        .rst_n (rst_n),
        .valid_in (valid_in),
        .pixel_in (pixel_in),
        .w00 (K00), .w01 (K01), .w02 (K02), .w03 (K03), .w04 (K04),
        .w10 (K10), .w11 (K11), .w12 (K12), .w13 (K13), .w14 (K14),
        .w20 (K20), .w21 (K21), .w22 (K22), .w23 (K23), .w24 (K24),
        .w30 (K30), .w31 (K31), .w32 (K32), .w33 (K33), .w34 (K34),
        .w40 (K40), .w41 (K41), .w42 (K42), .w43 (K43), .w44 (K44),
        .bias (BIAS),
        .result_ch0 (result_ch0),
        .result_valid (result_valid)
    );

    always #10 clk = ~clk;

    initial begin
        kernel[0]  = K00; kernel[1]  = K01; kernel[2]  = K02; kernel[3]  = K03; kernel[4]  = K04;
        kernel[5]  = K10; kernel[6]  = K11; kernel[7]  = K12; kernel[8]  = K13; kernel[9]  = K14;
        kernel[10] = K20; kernel[11] = K21; kernel[12] = K22; kernel[13] = K23; kernel[14] = K24;
        kernel[15] = K30; kernel[16] = K31; kernel[17] = K32; kernel[18] = K33; kernel[19] = K34;
        kernel[20] = K40; kernel[21] = K41; kernel[22] = K42; kernel[23] = K43; kernel[24] = K44;

        for (y = 0; y < IMG_HEIGHT; y = y + 1) begin
            for (x = 0; x < IMG_WIDTH; x = x + 1) begin
                image[y * IMG_WIDTH + x] = (x * 9 + y * 13);
            end
        end

        out_idx = 0;
        for (y = 0; y < OUT_H; y = y + 1) begin
            for (x = 0; x < OUT_W; x = x + 1) begin
                sum = 0;
                for (ky = 0; ky < 5; ky = ky + 1) begin
                    for (kx = 0; kx < 5; kx = kx + 1) begin
                        sum = sum +
                              $signed({1'b0, image[(y + ky) * IMG_WIDTH + (x + kx)]}) *
                              $signed(kernel[ky * 5 + kx]);
                    end
                end
                sum = sum + $signed(BIAS);
                if (sum < 0)
                    expected[out_idx] = 0;
                else
                    expected[out_idx] = sum;
                out_idx = out_idx + 1;
            end
        end
    end

    initial begin
        $dumpfile("sim/tb_conv1_accelerator.vcd");
        $dumpvars(0, tb_conv1_accelerator);

        clk = 0;
        rst_n = 0;
        valid_in = 0;
        pixel_in = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

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

        if (error_count == 0)
            $display("PASS: %0d outputs checked", OUT_PIXELS);
        else
            $display("FAIL: %0d mismatches out of %0d", error_count, OUT_PIXELS);

        $finish;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            output_count <= 0;
            error_count <= 0;
        end else if (result_valid) begin
            if (output_count < OUT_PIXELS) begin
                if (result_ch0 !== expected[output_count]) begin
                    $display("Mismatch at %0d exp=%0d got=%0d", output_count, expected[output_count], result_ch0);
                    error_count <= error_count + 1;
                end
                output_count <= output_count + 1;
            end else begin
                $display("Unexpected extra output at %0t: %0d", $time, result_ch0);
                error_count <= error_count + 1;
            end
        end
    end

endmodule
