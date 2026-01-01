module inference_tx_timer #(
    parameter integer IMG_PIXELS = 784,
    parameter integer OUT_PIXELS = 10,
    parameter integer UART_BPS = 115200,
    parameter integer CLK_FREQ = 50_000_000
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        rx_valid,
    input  wire [31:0] net_result,
    input  wire        net_valid,
    output reg  [7:0]  tx_data,
    output reg         tx_flag
);

    localparam integer TX_FRAME_CYCLES = (CLK_FREQ / UART_BPS) * 10;

    reg [9:0] rx_count;
    reg [3:0] out_count;
    reg       inf_running;
    reg [31:0] inf_cycles;
    reg [31:0] inf_cycles_latched;
    reg       inf_pure_running;
    reg [31:0] inf_cycles_pure;
    reg [31:0] inf_cycles_pure_latched;
    reg        inf_done;
    reg        tx_clear_inf_done;

    reg [7:0] result_buf [0:OUT_PIXELS-1];

    reg [1:0] tx_state;
    reg [3:0] tx_idx;
    reg [2:0] time_idx;
    reg [31:0] tx_cooldown;

    wire [31:0] time_word;
    wire [1:0] time_byte_idx;

    localparam [1:0] TX_IDLE = 2'd0;
    localparam [1:0] TX_RESULTS = 2'd1;
    localparam [1:0] TX_TIME = 2'd2;

    assign time_word = time_idx[2] ? inf_cycles_pure_latched : inf_cycles_latched;
    assign time_byte_idx = time_idx[1:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_count <= 10'd0;
            out_count <= 4'd0;
            inf_running <= 1'b0;
            inf_cycles <= 32'd0;
            inf_cycles_latched <= 32'd0;
            inf_pure_running <= 1'b0;
            inf_cycles_pure <= 32'd0;
            inf_cycles_pure_latched <= 32'd0;
            inf_done <= 1'b0;
        end else begin
            if (rx_valid && !inf_running && !inf_done) begin
                inf_running <= 1'b1;
                inf_cycles <= 32'd0;
                inf_pure_running <= 1'b0;
                inf_cycles_pure <= 32'd0;
                rx_count <= 10'd1;
                out_count <= 4'd0;
            end else if (rx_valid && inf_running) begin
                if (rx_count < IMG_PIXELS[9:0])
                    rx_count <= rx_count + 1'b1;
                if (rx_count == (IMG_PIXELS[9:0] - 1'b1))
                    inf_pure_running <= 1'b1;
            end

            if (inf_running)
                inf_cycles <= inf_cycles + 1'b1;

            if (inf_pure_running)
                inf_cycles_pure <= inf_cycles_pure + 1'b1;

            if (net_valid && inf_running) begin
                result_buf[out_count] <= net_result[7:0];
                if (out_count == OUT_PIXELS - 1) begin
                    inf_cycles_latched <= inf_cycles;
                    inf_cycles_pure_latched <= inf_cycles_pure;
                    inf_done <= 1'b1;
                    inf_running <= 1'b0;
                    inf_pure_running <= 1'b0;
                end else begin
                    out_count <= out_count + 1'b1;
                end
            end

            if (tx_clear_inf_done)
                inf_done <= 1'b0;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state <= TX_IDLE;
            tx_idx <= 4'd0;
            time_idx <= 3'd0;
            tx_cooldown <= 32'd0;
            tx_data <= 8'd0;
            tx_flag <= 1'b0;
            tx_clear_inf_done <= 1'b0;
        end else begin
            tx_flag <= 1'b0;
            tx_clear_inf_done <= 1'b0;
            if (tx_cooldown != 0)
                tx_cooldown <= tx_cooldown - 1'b1;

            case (tx_state)
                TX_IDLE: begin
                    if (inf_done) begin
                        tx_state <= TX_RESULTS;
                        tx_idx <= 4'd0;
                        time_idx <= 3'd0;
                    end
                end
                TX_RESULTS: begin
                    if (tx_cooldown == 0) begin
                        tx_data <= result_buf[tx_idx];
                        tx_flag <= 1'b1;
                        tx_cooldown <= TX_FRAME_CYCLES[31:0];
                        if (tx_idx == OUT_PIXELS - 1) begin
                            tx_state <= TX_TIME;
                            tx_idx <= 4'd0;
                        end else begin
                            tx_idx <= tx_idx + 1'b1;
                        end
                    end
                end
                TX_TIME: begin
                    if (tx_cooldown == 0) begin
                        tx_data <= time_word[8*time_byte_idx +: 8];
                        tx_flag <= 1'b1;
                        tx_cooldown <= TX_FRAME_CYCLES[31:0];
                        if (time_idx == 3'd7) begin
                            tx_state <= TX_IDLE;
                            time_idx <= 3'd0;
                            tx_clear_inf_done <= 1'b1;
                        end else begin
                            time_idx <= time_idx + 1'b1;
                        end
                    end
                end
                default: tx_state <= TX_IDLE;
            endcase
        end
    end

endmodule
