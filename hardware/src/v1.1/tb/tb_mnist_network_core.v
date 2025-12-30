`timescale 1ns/1ps
`include "quant_params.vh"

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
    reg signed [7:0] fc1_vec [0:31];
    integer fc1_idx;
    reg signed [7:0] conv1_stream [0:95][0:5];
    integer conv1_idx;
    reg signed [7:0] pool1_stream [0:15][0:5];
    integer pool1_idx;
    reg signed [7:0] pool2_stream [0:15][0:15];
    integer pool2_idx;
    reg signed [7:0] conv2_q_stream [0:15][0:15];
    integer conv2_q_idx;
    reg signed [31:0] conv2_acc_stream [0:15];
    reg conv2_acc_done;
    reg conv2_dbg_done;
    integer conv2_mac_count;
    reg conv2_mac_active;
    integer y;

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
`ifndef QUIET_SIM
        $dumpfile("tb_mnist_network_core.vcd");
`ifdef FULL_DUMP
        $dumpvars(0, dut);
`else
        $dumpvars(0, clk, rst_n, valid_in, pixel_in, result, result_valid);
        $dumpvars(0, dut.u_layer1.result_valid, dut.u_layer2.out_valid,
                  dut.u_layer3.out_valid, dut.u_layer4.out_valid);
        $dumpvars(0, dut.u_layer1.load_state, dut.u_layer2.load_state,
                  dut.u_layer3.load_state, dut.u_layer4.load_state);
`endif
        $dumpoff;
`endif

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
        conv1_idx = 0;
        pool1_idx = 0;
        pool2_idx = 0;
        conv2_q_idx = 0;
        conv2_acc_done = 0;
        conv2_dbg_done = 0;
        conv2_mac_count = 0;
        conv2_mac_active = 0;

        repeat (5) @(posedge clk);
        rst_n = 1;

        // Wait for all layer weight loaders to complete (with timeout)
        idx = 0;
        while (!(dut.u_layer1.load_state == 1'b1 &&
                 dut.u_layer2.load_state == 1'b1 &&
                 dut.u_layer3.load_state == 1'b1 &&
                 dut.u_layer4.load_state == 1'b1) &&
               (idx < MAX_WARMUP)) begin
            @(posedge clk);
            idx = idx + 1;
        end
        if (idx >= MAX_WARMUP) begin
            $display("WARN: weight load timeout after %0d cycles", MAX_WARMUP);
        end
`ifndef QUIET_SIM
        $dumpon;
`endif

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
`ifndef QUIET_SIM
        $display("Conv1 (first 32 positions, ch0..5):");
        for (idx = 0; idx < 32; idx = idx + 1) begin
            $display("  pos%0d: %0d %0d %0d %0d %0d %0d",
                     idx,
                     conv1_stream[idx][0], conv1_stream[idx][1], conv1_stream[idx][2],
                     conv1_stream[idx][3], conv1_stream[idx][4], conv1_stream[idx][5]);
        end
        $display("Conv1 (row2 col0..7, pos48..55):");
        for (idx = 48; idx < 56; idx = idx + 1) begin
            $display("  pos%0d: %0d %0d %0d %0d %0d %0d",
                     idx,
                     conv1_stream[idx][0], conv1_stream[idx][1], conv1_stream[idx][2],
                     conv1_stream[idx][3], conv1_stream[idx][4], conv1_stream[idx][5]);
        end
        $display("Conv1 (row3 col2..3, pos74..75):");
        for (idx = 74; idx < 76; idx = idx + 1) begin
            $display("  pos%0d: %0d %0d %0d %0d %0d %0d",
                     idx,
                     conv1_stream[idx][0], conv1_stream[idx][1], conv1_stream[idx][2],
                     conv1_stream[idx][3], conv1_stream[idx][4], conv1_stream[idx][5]);
        end
        $display("Conv1 biases:");
        $display("  %0d %0d %0d %0d %0d %0d",
                 dut.u_layer1.conv1_bias_ch0,
                 dut.u_layer1.conv1_bias_ch1,
                 dut.u_layer1.conv1_bias_ch2,
                 dut.u_layer1.conv1_bias_ch3,
                 dut.u_layer1.conv1_bias_ch4,
                 dut.u_layer1.conv1_bias_ch5);
        $display("Quant params: SHIFT=%0d Q_MULT_CONV1=%0d Q_MULT_CONV2=%0d Q_MULT_FC1=%0d Q_MULT_FC2=%0d",
                 `Q_SHIFT, `Q_MULT_CONV1, `Q_MULT_CONV2, `Q_MULT_FC1, `Q_MULT_FC2);

        $display("Pool1 (first 16 positions, ch0..5):");
        for (idx = 0; idx < 16; idx = idx + 1) begin
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

        $display("Conv2 q (first 4 positions, ch0..15):");
        for (idx = 0; idx < 4; idx = idx + 1) begin
            $display("  pos%0d: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                     idx,
                     conv2_q_stream[idx][0], conv2_q_stream[idx][1], conv2_q_stream[idx][2], conv2_q_stream[idx][3],
                     conv2_q_stream[idx][4], conv2_q_stream[idx][5], conv2_q_stream[idx][6], conv2_q_stream[idx][7],
                     conv2_q_stream[idx][8], conv2_q_stream[idx][9], conv2_q_stream[idx][10], conv2_q_stream[idx][11],
                     conv2_q_stream[idx][12], conv2_q_stream[idx][13], conv2_q_stream[idx][14], conv2_q_stream[idx][15]);
        end

        $display("Conv2 out (pos0, pre-quant):");
        $display("  %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                 conv2_acc_stream[0], conv2_acc_stream[1], conv2_acc_stream[2], conv2_acc_stream[3],
                 conv2_acc_stream[4], conv2_acc_stream[5], conv2_acc_stream[6], conv2_acc_stream[7],
                 conv2_acc_stream[8], conv2_acc_stream[9], conv2_acc_stream[10], conv2_acc_stream[11],
                 conv2_acc_stream[12], conv2_acc_stream[13], conv2_acc_stream[14], conv2_acc_stream[15]);

        $display("Feat mem ch0 5x5:");
        for (y = 0; y < 5; y = y + 1) begin
            $display("  %0d %0d %0d %0d %0d",
                     dut.u_layer2.feat_mem0[y*12+0], dut.u_layer2.feat_mem0[y*12+1],
                     dut.u_layer2.feat_mem0[y*12+2], dut.u_layer2.feat_mem0[y*12+3],
                     dut.u_layer2.feat_mem0[y*12+4]);
        end
        $display("Feat mem ch1 5x5:");
        for (y = 0; y < 5; y = y + 1) begin
            $display("  %0d %0d %0d %0d %0d",
                     dut.u_layer2.feat_mem1[y*12+0], dut.u_layer2.feat_mem1[y*12+1],
                     dut.u_layer2.feat_mem1[y*12+2], dut.u_layer2.feat_mem1[y*12+3],
                     dut.u_layer2.feat_mem1[y*12+4]);
        end
        $display("Feat mem ch2 5x5:");
        for (y = 0; y < 5; y = y + 1) begin
            $display("  %0d %0d %0d %0d %0d",
                     dut.u_layer2.feat_mem2[y*12+0], dut.u_layer2.feat_mem2[y*12+1],
                     dut.u_layer2.feat_mem2[y*12+2], dut.u_layer2.feat_mem2[y*12+3],
                     dut.u_layer2.feat_mem2[y*12+4]);
        end
        $display("Feat mem ch3 5x5:");
        for (y = 0; y < 5; y = y + 1) begin
            $display("  %0d %0d %0d %0d %0d",
                     dut.u_layer2.feat_mem3[y*12+0], dut.u_layer2.feat_mem3[y*12+1],
                     dut.u_layer2.feat_mem3[y*12+2], dut.u_layer2.feat_mem3[y*12+3],
                     dut.u_layer2.feat_mem3[y*12+4]);
        end
        $display("Feat mem ch4 5x5:");
        for (y = 0; y < 5; y = y + 1) begin
            $display("  %0d %0d %0d %0d %0d",
                     dut.u_layer2.feat_mem4[y*12+0], dut.u_layer2.feat_mem4[y*12+1],
                     dut.u_layer2.feat_mem4[y*12+2], dut.u_layer2.feat_mem4[y*12+3],
                     dut.u_layer2.feat_mem4[y*12+4]);
        end
        $display("Feat mem ch5 5x5:");
        for (y = 0; y < 5; y = y + 1) begin
            $display("  %0d %0d %0d %0d %0d",
                     dut.u_layer2.feat_mem5[y*12+0], dut.u_layer2.feat_mem5[y*12+1],
                     dut.u_layer2.feat_mem5[y*12+2], dut.u_layer2.feat_mem5[y*12+3],
                     dut.u_layer2.feat_mem5[y*12+4]);
        end

        $display("FC1 outputs:");
        for (idx = 0; idx < 32; idx = idx + 1) begin
            $display("  [%0d] = %0d (0x%08h)", idx, fc1_vec[idx], fc1_vec[idx]);
        end

        $display("FC2 outputs:");
        for (idx = 0; idx < OUT_PIXELS; idx = idx + 1) begin
            $display("  [%0d] = %0d (0x%08h)", idx, out_vec[idx], out_vec[idx]);
        end
`endif
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
`ifndef QUIET_SIM
        $dumpoff;
`endif
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
            conv2_q_idx <= 0;
        end else if (result_valid) begin
            output_count <= output_count + 1;
            out_vec[out_idx] <= result;
            out_idx <= out_idx + 1;
        end
        if (rst_n) begin
            if (dut.u_layer2.c_state == 2'd2 &&
                dut.u_layer2.pos_x == 3'd0 &&
                dut.u_layer2.pos_y == 3'd0 &&
                dut.u_layer2.oc_idx == 4'd0) begin
                if (!conv2_mac_active) begin
                    conv2_mac_active <= 1'b1;
                    conv2_mac_count <= 0;
                end
                if (dut.u_layer2.mac_phase == 1'b1)
                    conv2_mac_count <= conv2_mac_count + 1;
                if (dut.u_layer2.mac_phase == 1'b1 &&
                    dut.u_layer2.k_ch == 3'd5 &&
                    dut.u_layer2.k_y == 3'd4 &&
                    dut.u_layer2.k_x == 3'd4) begin
                    $display("Conv2 mac_count (pos0 oc0): %0d", conv2_mac_count + 1);
                    conv2_mac_active <= 1'b0;
                end
            end
            if (!conv2_dbg_done &&
                dut.u_layer2.c_state == 2'd2 &&
                dut.u_layer2.mac_phase == 1'b1 &&
                dut.u_layer2.pos_x == 3'd0 &&
                dut.u_layer2.pos_y == 3'd0 &&
                dut.u_layer2.oc_idx == 4'd0 &&
                dut.u_layer2.k_ch == 3'd0 &&
                dut.u_layer2.k_y == 3'd0 &&
                dut.u_layer2.k_x == 3'd0) begin
                $display("Conv2 dbg: w_addr=%0d w_q=%0d feat=%0d b0=%0d",
                         dut.u_layer2.conv2_weight_addr,
                         $signed(dut.u_layer2.conv2_weight_q),
                         $signed(dut.u_layer2.feat_rd_data0),
                         $signed(dut.u_layer2.conv2_biases[0]));
                conv2_dbg_done <= 1'b1;
            end
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
            if (dut.u_layer1.conv_valid_r && conv1_idx < 96) begin
                conv1_stream[conv1_idx][0] <= dut.u_layer1.conv1_q_buf[0];
                conv1_stream[conv1_idx][1] <= dut.u_layer1.conv1_q_buf[1];
                conv1_stream[conv1_idx][2] <= dut.u_layer1.conv1_q_buf[2];
                conv1_stream[conv1_idx][3] <= dut.u_layer1.conv1_q_buf[3];
                conv1_stream[conv1_idx][4] <= dut.u_layer1.conv1_q_buf[4];
                conv1_stream[conv1_idx][5] <= dut.u_layer1.conv1_q_buf[5];
                if (conv1_idx == 74 || conv1_idx == 75) begin
                    $display("Conv1 idx=%0d pos_x=%0d pos_y=%0d", conv1_idx, dut.u_layer1.pos_x, dut.u_layer1.pos_y);
                end
                conv1_idx <= conv1_idx + 1;
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
            if (dut.u_layer2.conv2_valid && conv2_q_idx < 16) begin
                conv2_q_stream[conv2_q_idx][0] <= dut.u_layer2.conv2_q_buf[0];
                conv2_q_stream[conv2_q_idx][1] <= dut.u_layer2.conv2_q_buf[1];
                conv2_q_stream[conv2_q_idx][2] <= dut.u_layer2.conv2_q_buf[2];
                conv2_q_stream[conv2_q_idx][3] <= dut.u_layer2.conv2_q_buf[3];
                conv2_q_stream[conv2_q_idx][4] <= dut.u_layer2.conv2_q_buf[4];
                conv2_q_stream[conv2_q_idx][5] <= dut.u_layer2.conv2_q_buf[5];
                conv2_q_stream[conv2_q_idx][6] <= dut.u_layer2.conv2_q_buf[6];
                conv2_q_stream[conv2_q_idx][7] <= dut.u_layer2.conv2_q_buf[7];
                conv2_q_stream[conv2_q_idx][8] <= dut.u_layer2.conv2_q_buf[8];
                conv2_q_stream[conv2_q_idx][9] <= dut.u_layer2.conv2_q_buf[9];
                conv2_q_stream[conv2_q_idx][10] <= dut.u_layer2.conv2_q_buf[10];
                conv2_q_stream[conv2_q_idx][11] <= dut.u_layer2.conv2_q_buf[11];
                conv2_q_stream[conv2_q_idx][12] <= dut.u_layer2.conv2_q_buf[12];
                conv2_q_stream[conv2_q_idx][13] <= dut.u_layer2.conv2_q_buf[13];
                conv2_q_stream[conv2_q_idx][14] <= dut.u_layer2.conv2_q_buf[14];
                conv2_q_stream[conv2_q_idx][15] <= dut.u_layer2.conv2_q_buf[15];
                conv2_q_idx <= conv2_q_idx + 1;
            end
            if (!conv2_acc_done && dut.u_layer2.conv2_valid) begin
                conv2_acc_stream[0] <= dut.u_layer2.conv2_out_buf[0];
                conv2_acc_stream[1] <= dut.u_layer2.conv2_out_buf[1];
                conv2_acc_stream[2] <= dut.u_layer2.conv2_out_buf[2];
                conv2_acc_stream[3] <= dut.u_layer2.conv2_out_buf[3];
                conv2_acc_stream[4] <= dut.u_layer2.conv2_out_buf[4];
                conv2_acc_stream[5] <= dut.u_layer2.conv2_out_buf[5];
                conv2_acc_stream[6] <= dut.u_layer2.conv2_out_buf[6];
                conv2_acc_stream[7] <= dut.u_layer2.conv2_out_buf[7];
                conv2_acc_stream[8] <= dut.u_layer2.conv2_out_buf[8];
                conv2_acc_stream[9] <= dut.u_layer2.conv2_out_buf[9];
                conv2_acc_stream[10] <= dut.u_layer2.conv2_out_buf[10];
                conv2_acc_stream[11] <= dut.u_layer2.conv2_out_buf[11];
                conv2_acc_stream[12] <= dut.u_layer2.conv2_out_buf[12];
                conv2_acc_stream[13] <= dut.u_layer2.conv2_out_buf[13];
                conv2_acc_stream[14] <= dut.u_layer2.conv2_out_buf[14];
                conv2_acc_stream[15] <= dut.u_layer2.conv2_out_buf[15];
                conv2_acc_done <= 1'b1;
            end
            if (dut.u_layer3.out_valid && fc1_idx < 32) begin
                fc1_vec[fc1_idx] <= dut.u_layer3.out_data;
                fc1_idx <= fc1_idx + 1;
            end
        end
    end

    always @(posedge clk) begin
        sim_cycles <= sim_cycles + 1;
`ifndef QUIET_SIM
        if ((sim_cycles % 50000) == 0) begin
            $display("T=%0t rst_n=%b l1=%0d l2=%0d l3=%0d l4=%0d vin=%0d l1v=%0d l2v=%0d l3v=%0d l4v=%0d",
                     $time, rst_n,
                     dut.u_layer1.load_state,
                     dut.u_layer2.load_state,
                     dut.u_layer3.load_state,
                     dut.u_layer4.load_state,
                     vin_count, l1_count, l2_count, l3_count, l4_count);
        end
`endif
        if (sim_cycles >= MAX_SIM_CYCLES) begin
            $display("TIMEOUT after %0d cycles, outputs=%0d vin=%0d l1v=%0d l2v=%0d l3v=%0d l4v=%0d",
                     sim_cycles, output_count, vin_count, l1_count, l2_count, l3_count, l4_count);
            $finish;
        end
    end

endmodule
