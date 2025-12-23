`timescale 1ns/1ps

module tb_vector_dot_product();

    reg clk;
    reg rst_n;
    reg valid_in;
    reg sop, eop;
    reg signed [7:0] data_a;
    reg signed [7:0] data_b;
    
    wire signed [31:0] result;
    wire result_valid;

    // 实例化被测模块
    vector_dot_product uut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .sop(sop),
        .eop(eop),
        .data_a(data_a),
        .data_b(data_b),
        .result(result),
        .result_valid(result_valid)
    );

    // 生成时钟 (50MHz, 周期 20ns)
    always #10 clk = ~clk;

    initial begin
        $dumpfile("sim/wave.vcd"); 
        $dumpvars(0, tb_vector_dot_product);

        // 初始化
        clk = 0; rst_n = 0; 
        valid_in = 0; sop = 0; eop = 0; data_a = 0; data_b = 0;
        
        #50 rst_n = 1; // 复位结束
        #20;

        // -----------------------------------
        // 开始发送向量: [2, -3, 4] dot [5, 2, 1]
        // -----------------------------------
        
        // 第 1 个点: 2 * 5
        @(posedge clk);
        valid_in = 1; sop = 1; eop = 0;
        data_a = 2; data_b = 5;

        // 第 2 个点: -3 * 2
        @(posedge clk);
        valid_in = 1; sop = 0; eop = 0;
        data_a = -8'd3; // 注意负数写法
        data_b = 2;

        // 第 3 个点: 4 * 1 (最后一个点)
        @(posedge clk);
        valid_in = 1; sop = 0; eop = 1; // EOP 置高
        data_a = 4; data_b = 1;

        // 结束发送
        @(posedge clk);
        valid_in = 0; sop = 0; eop = 0;
        
        // 等待结果出来
        #100;
        
        $stop;
    end

endmodule