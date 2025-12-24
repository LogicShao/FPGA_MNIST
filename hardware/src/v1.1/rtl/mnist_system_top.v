module mnist_system_top(
    input  sys_clk,
    input  sys_rst_n,

    // UART
    input  uart_rx,
    output uart_tx,

    // 7-seg (595)
    output ds,
    output oe,
    output shcp,
    output stcp
);

    // ==========================================
    // 1. Internal signals
    // ==========================================
    wire [7:0] rx_byte;
    wire       rx_valid;

    wire [31:0] net_result;
    wire        net_valid;

    reg [3:0] display_num;

    // ==========================================
    // 2. Result latch for display
    // ==========================================
    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            display_num <= 4'd0;
        end else if (net_valid) begin
            display_num <= net_result[3:0];
        end
    end

    // ==========================================
    // 3. UART RX
    // ==========================================
    uart_rx #(
        .UART_BPS(115200),
        .CLK_FREQ(50_000_000)
    ) u_rx (
        .sys_clk   (sys_clk),
        .sys_rst_n (sys_rst_n),
        .rx        (uart_rx),
        .po_data   (rx_byte),
        .po_flag   (rx_valid)
    );

    // ==========================================
    // 4. MNIST network core
    // ==========================================
    mnist_network_core u_core (
        .clk          (sys_clk),
        .rst_n        (sys_rst_n),
        .valid_in     (rx_valid),
        .pixel_in     (rx_byte),
        .result       (net_result),
        .result_valid (net_valid)
    );

    // ==========================================
    // 5. UART TX
    // ==========================================
    uart_tx #(
        .UART_BPS(115200),
        .CLK_FREQ(50_000_000)
    ) u_tx (
        .sys_clk   (sys_clk),
        .sys_rst_n (sys_rst_n),
        .pi_data   (net_result[7:0]),
        .pi_flag   (net_valid),
        .tx        (uart_tx)
    );

    // ==========================================
    // 6. 7-seg driver
    // ==========================================
    seg_595_dynamic u_seg_595 (
        .sys_clk   (sys_clk),
        .sys_rst_n (sys_rst_n),
        .data      ({16'd0, display_num}),
        .point     (6'b000000),
        .seg_en    (1'b1),
        .sign      (1'b0),
        .ds        (ds),
        .oe        (oe),
        .shcp      (shcp),
        .stcp      (stcp)
    );

endmodule
