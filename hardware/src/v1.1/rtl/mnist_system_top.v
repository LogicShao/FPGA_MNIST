module mnist_system_top(
    input  sys_clk,
    input  sys_rst_n,
    
    // UART 接口
    input  uart_rx,      // PC -> FPGA (接收图片)
    output uart_tx,      // FPGA -> PC (发送结果)
    
    // 数码管接口 (595 驱动)
    output ds,
    output oe,
    output shcp,
    output stcp
);

    // ==========================================
    // 1. 内部信号定义
    // ==========================================
    // UART RX 信号
    wire [7:0] rx_byte;
    wire       rx_valid;     // 接收到一个字节的脉冲

    // 加速器信号
    wire [31:0] conv_result;
    wire        conv_valid;  // 计算完成的脉冲

    // 结果锁存寄存器 (用于数码管持续显示)
    reg [3:0]   display_num;

    // ==========================================
    // 2. 结果锁存逻辑 (关键！)
    // ==========================================
    // 加速器的结果是瞬时的，数码管需要持续显示，
    // 所以我们需要由一个寄存器来“记住”最后一次识别的结果。
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            display_num <= 4'd0; // 复位默认显示 0
        end else if (conv_valid) begin
            // 当计算完成脉冲到来时，更新寄存器里的值
            // 我们假设结果在 0-9 之间，取低 4 位即可
            display_num <= conv_result[3:0]; 
        end
    end

    // ==========================================
    // 3. 实例化 UART RX (接收图片)
    // ==========================================
    uart_rx #(
        .UART_BPS(115200),
        .CLK_FREQ(50_000_000)
    ) u_rx (
        .sys_clk   (sys_clk),
        .sys_rst_n (sys_rst_n),
        .rx        (uart_rx),
        .po_data   (rx_byte),
        .po_flag   (rx_valid)  // 直接连到加速器的 valid_in
    );

    // ==========================================
    // 4. 实例化 卷积加速器 (计算核心)
    // ==========================================
    conv_accelerator u_acc (
        .clk        (sys_clk),
        .rst_n      (sys_rst_n),
        .valid_in   (rx_valid),
        .pixel_in   (rx_byte),
        .result_ch0 (conv_result),
        .result_valid (conv_valid)
    );

    // ==========================================
    // 5. 实例化 UART TX (发送结果回PC)
    // ==========================================
    uart_tx #(
        .UART_BPS(115200),
        .CLK_FREQ(50_000_000)
    ) u_tx (
        .sys_clk   (sys_clk),
        .sys_rst_n (sys_rst_n),
        .pi_data   (conv_result[7:0]), // 发送低8位
        .pi_flag   (conv_valid),       // 算完立刻发
        .tx        (uart_tx)
    );

    // ==========================================
    // 6. 实例化 数码管驱动 (本地显示)
    // ==========================================
    // 注意：这里假设你的 seg_595_dynamic 模块 data 接口宽度为 20位 (5位 x 4bits/digit)
    // 我们把结果显示在最右边的个位上。
    
    seg_595_dynamic u_seg_595 (
        .sys_clk   (sys_clk),
        .sys_rst_n (sys_rst_n),
        
        // 数据映射：假设高位补0，最低4位接我们的识别结果
        .data      ({16'd0, display_num}), 
        
        // 小数点：全不亮
        .point     (6'b000000),         
        
        // 使能：常开
        .seg_en    (1'b1),              
        
        // 符号位：正数
        .sign      (1'b0),              
        
        // 物理引脚输出
        .ds        (ds),
        .oe        (oe),
        .shcp      (shcp),
        .stcp      (stcp)
    );

endmodule
