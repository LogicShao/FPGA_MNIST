module mnist_system_top(
    input sys_clk,
    input sys_rst_n,
    
    // 串口
    input uart_rx,  
    output uart_tx, 
    
    // 数码管 (沿用 V1)
    output ds,
    output oe,
    output shcp,
    output stcp
);

    wire nios2clk;
    wire locked;
    
    // PIO 信号连接
    wire [31:0] pio_seg_ctrl;     // 原来的数码管控制
    wire [31:0] pio_img_data;     // [新增] Nios 发送给 FPGA 的数据
    wire [31:0] pio_result;       // [新增] FPGA 返回给 Nios 的结果

    // PLL
    clk_pll niospll_inst(
        .inclk0(sys_clk),
        .c0(nios2clk),
        .locked(locked)
    );
    
    // 系统复位：物理复位键 + PLL 锁定信号
    wire system_rst_n = sys_rst_n & locked;

    // ========================================================
    // 1. 实例化 Nios II 系统 (Qsys)
    // ========================================================
    qsys_system qsys_inst(
        .clk_clk(nios2clk),
        .reset_reset_n(system_rst_n),
        
        .uart_0_external_connection_rxd(uart_rx), 
        .uart_0_external_connection_txd(uart_tx),
        
        // 旧的 PIO：控制数码管
        .out_pio_external_connection_export(pio_seg_ctrl),
        
        // 新增的 PIO (需要在 Qsys 里添加)
        // pio_img_data: bit[7:0]=pixel, bit[8]=valid, bit[9]=acc_rst
        .pio_img_data_external_connection_export(pio_img_data),
        
        // 新增的 PIO: 读取结果
        .pio_result_external_connection_export(pio_result)
    );

    // ========================================================
    // 2. 信号解析
    // ========================================================
    // 从 pio_img_data 拆分信号
    wire [7:0] pixel_from_cpu = pio_img_data[7:0];
    wire       valid_from_cpu = pio_img_data[8];
    wire       acc_rst_n      = system_rst_n & (~pio_img_data[9]); // 软复位

    wire [31:0] conv_result;
    wire        conv_valid;

    // ========================================================
    // 3. 实例化你的硬件加速器
    // ========================================================
    conv_accelerator u_accelerator (
        .clk(nios2clk),
        .rst_n(acc_rst_n),
        .valid_in(valid_from_cpu),
        .pixel_in(pixel_from_cpu),
        .result_ch0(conv_result),
        .result_valid(conv_valid)
    );

    // 把结果送回给 Nios (PIO Input)
    // 也可以接一个 FIFO，或者简单的寄存器锁存
    reg [31:0] result_latch;
    always @(posedge nios2clk) begin
        if (!acc_rst_n) result_latch <= 0;
        else if (conv_valid) result_latch <= conv_result;
    end
    assign pio_result = result_latch;

    // ========================================================
    // 4. 数码管显示模块 (保持不变)
    // ========================================================
    seg_595_dynamic seg_inst(
        .sys_clk(nios2clk),
        .sys_rst_n(system_rst_n),
        .data(pio_seg_ctrl[19:0]),
        .point(pio_seg_ctrl[25:20]),
        .seg_en(pio_seg_ctrl[26]),
        .sign(pio_seg_ctrl[27]),
        .ds(ds),
        .oe(oe),
        .shcp(shcp),
        .stcp(stcp)     
    );

endmodule