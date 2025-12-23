module conv_accelerator(
    input clk,
    input rst_n,
    input valid_in,           // 来自 Nios 的 valid 信号
    input [7:0] pixel_in,     // 来自 Nios 的像素数据
    
    // 这里为了演示，我们只计算 1 个通道 (Channel 0)
    // 实际项目中你需要实例化 6 个 PE
    output signed [31:0] result_ch0, 
    output result_valid
);

    // 1. 连接 Window Generator
    wire [7:0] w00, w01, w02, w03, w04;
    wire [7:0] w10, w11, w12, w13, w14;
    wire [7:0] w20, w21, w22, w23, w24;
    wire [7:0] w30, w31, w32, w33, w34;
    wire [7:0] w40, w41, w42, w43, w44;
    wire win_valid;

    layer1_window_gen u_window_gen (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .din(pixel_in),
        // 连接所有 w00...w44 信号
        .w00(w00), .w01(w01), .w02(w02), .w03(w03), .w04(w04),
        .w10(w10), .w11(w11), .w12(w12), .w13(w13), .w14(w14),
        .w20(w20), .w21(w21), .w22(w22), .w23(w23), .w24(w24),
        .w30(w30), .w31(w31), .w32(w32), .w33(w33), .w34(w34),
        .w40(w40), .w41(w41), .w42(w42), .w43(w43), .w44(w44),
        .window_valid(win_valid)
    );

    // 2. 连接计算核心 (Processing Element)
    // 假设我们有一组固定的权重 (这里先写死用于测试，后续通过 parameter 或 ROM 读取)
    // 这是一个 Sobel 算子之类的简单权重，或者你可以把 Python 导出的权重填在这里
    wire signed [7:0] weight_00 = 8'd10; // 示例权重
    // ... 实际需要定义全部 25 个权重 ...
    
    // 生成 Start/End of Packet 信号 (简化版：假设 win_valid 持续有效)
    // 注意：这里的 sop/eop 逻辑在流式处理中需要更精细的设计
    // 现阶段我们先让它一直算
    
    vector_dot_product u_pe_ch0 (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(win_valid), // 只有当窗口准备好时才计算
        .sop(0), // 暂时置0，需要根据计数器生成
        .eop(0), 
        .data_a(w00), // 这里只连接了第一个点作为演示，实际需要修改 PE 支持并行 25点输入
        // ⚠️ 注意：你之前的 vector_dot_product 是串行的(1个乘法器)。
        // 要实现高性能卷积，你需要修改 vector_dot_product 让它拥有 25 个乘法器并行计算！
        // 或者，我们依然用串行，但那样控制逻辑会很复杂。
        
        // 为了让你现在能跑通，我们假设你已经将其修改为并行输入，或者我们只算这一个点验证数据流。
        .data_b(weight_00),
        .result(result_ch0),
        .result_valid(result_valid)
    );
    
    // 重要提示：
    // 你之前的 vector_dot_product 是“向量点乘流式版”，适合 Nios 慢慢喂数据算全连接。
    // 对于卷积，我们通常写一个“并行乘加树”。
    // 这一步如果不理解，我们可以后续专门讲“并行卷积核写法”。

endmodule