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

    localparam integer UART_BPS = 115200;
    localparam integer CLK_FREQ = 50_000_000;

    // ==========================================
    // 1. Internal signals
    // ==========================================
    wire [7:0] rx_byte;
    wire       rx_valid;

    wire [31:0] net_result;
    wire        net_valid;
    wire signed [31:0] net_result_s;

    reg [3:0] display_num;
    reg [3:0] out_idx;
    reg signed [31:0] max_val;
    reg [3:0] max_idx;
    wire signed [31:0] next_max_val;
    wire [3:0] next_max_idx;

    wire [7:0] tx_data;
    wire       tx_flag;

    // ==========================================
    // 2. Argmax over 10 outputs for display
    // ==========================================
    assign net_result_s = net_result;
    assign next_max_val = (out_idx == 4'd0 || net_result_s > max_val) ? net_result_s : max_val;
    assign next_max_idx = (out_idx == 4'd0 || net_result_s > max_val) ? out_idx : max_idx;

    always @(posedge sys_clk or negedge sys_rst_n) begin
        if (!sys_rst_n) begin
            display_num <= 4'd0;
            out_idx <= 4'd0;
            max_val <= 32'sh8000_0000;
            max_idx <= 4'd0;
        end else if (net_valid) begin
            max_val <= next_max_val;
            max_idx <= next_max_idx;
            if (out_idx == 4'd9) begin
                display_num <= next_max_idx;
                out_idx <= 4'd0;
            end else begin
                out_idx <= out_idx + 1'b1;
            end
        end
    end


    // ==========================================
    // 3. UART RX
    // ==========================================
    uart_rx #(
        .UART_BPS(UART_BPS),
        .CLK_FREQ(CLK_FREQ)
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
    // 5. Inference timer + UART payload formatter
    // ==========================================
    inference_tx_timer #(
        .UART_BPS (UART_BPS),
        .CLK_FREQ (CLK_FREQ)
    ) u_timer_tx (
        .clk        (sys_clk),
        .rst_n      (sys_rst_n),
        .rx_valid   (rx_valid),
        .net_result (net_result),
        .net_valid  (net_valid),
        .tx_data    (tx_data),
        .tx_flag    (tx_flag)
    );

    // ==========================================
    // 6. UART TX
    // ==========================================

    uart_tx #(
        .UART_BPS(UART_BPS),
        .CLK_FREQ(CLK_FREQ)
    ) u_tx (
        .sys_clk   (sys_clk),
        .sys_rst_n (sys_rst_n),
        .pi_data   (tx_data),
        .pi_flag   (tx_flag),
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
