module mnist_nios(
	input sys_clk,
	input sys_rst_n,
	
	// --- 新增: 串口管脚 ---
	input uart_rx,  // 接收数据 (来自 PC/USB-TTL 的 TX)
	output uart_tx, // 发送数据 (去往 PC/USB-TTL 的 RX)
	// --------------------
	
	output ds,
	output oe,
	output shcp,
	output stcp
);

	wire nios2clk;
	wire [31:0] pio;

	clk_pll niospll_inst(
		.inclk0(sys_clk),
		.c0(nios2clk)
	);
	
	// Qsys 系统实例化
	// 注意：这里的端口名必须与 Qsys 生成的 .v 文件中的名字完全一致
	// 如果编译报错 "port not found"，请打开 qsys_system.v 确认实际名字
	qsys_system qsys_system_inst(
		.clk_clk(nios2clk),
		.reset_reset_n(sys_rst_n),
		.out_pio_external_connection_export(pio),
		
		// --- 连接串口 ---
		// 通常 Qsys 导出的名字格式为：<导出名>_rxd / <导出名>_txd
		.uart_0_external_connection_rxd(uart_rx), 
		.uart_0_external_connection_txd(uart_tx)
	);
	
	seg_595_dynamic seg_595_dynamic_inst(
		.sys_clk(nios2clk),
		.sys_rst_n(sys_rst_n),
		.data(pio[19:0]),
		.point(pio[25:20]),
		.seg_en(pio[26]),
		.sign(pio[27]),
		.ds(ds),
		.oe(oe),
		.shcp(shcp),
		.stcp(stcp)		
	);
	
endmodule