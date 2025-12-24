module uart_tx
#(
    parameter UART_BPS = 'd115200,        // 串口波特率
    parameter CLK_FREQ = 'd50_000_000   // 时钟频率
)
(
    input wire sys_clk,         // 系统时钟 50MHz
    input wire sys_rst_n,       // 全局复位
    input wire [7:0] pi_data,   // 模块输入的 8bit 数据
    input wire pi_flag,         // 并行数据有效标志信号

    output reg tx               // 串行输出的 1bit 数据
);

//********************************************************************//
//****************** Parameter and Internal Signal *******************//
//********************************************************************//

// localparam define
localparam BAUD_CNT_MAX = CLK_FREQ / UART_BPS;

// reg define
reg [12:0] baud_cnt;
reg bit_flag;
reg [3:0] bit_cnt;
reg work_en;

//********************************************************************//
//***************************** Main Code ****************************//
//********************************************************************//

// work_en: 发送工作使能信号
always @(posedge sys_clk or negedge sys_rst_n)
    if (sys_rst_n == 1'b0)
        work_en <= 1'b0;
    else if (pi_flag == 1'b1)
        work_en <= 1'b1;
    else if ((bit_flag == 1'b1) && (bit_cnt == 4'd9))
        work_en <= 1'b0;

// baud_cnt: 波特率计数器，从 0 计数到 BAUD_CNT_MAX - 1
always @(posedge sys_clk or negedge sys_rst_n)
    if (sys_rst_n == 1'b0)
        baud_cnt <= 13'b0;
    else if ((baud_cnt == BAUD_CNT_MAX - 1) || (work_en == 1'b0))
        baud_cnt <= 13'b0;
    else if (work_en == 1'b1)
        baud_cnt <= baud_cnt + 1'b1;

// bit_flag: 当 baud_cnt 计数到 1 时拉高一个周期，用于精确位发送时机
always @(posedge sys_clk or negedge sys_rst_n)
    if (sys_rst_n == 1'b0)
        bit_flag <= 1'b0;
    else if (baud_cnt == 13'd1)
        bit_flag <= 1'b1;
    else
        bit_flag <= 1'b0;

// bit_cnt: 发送位计数器，共 10 位（1 起始位 + 8 数据位 + 1 停止位）
always @(posedge sys_clk or negedge sys_rst_n)
    if (sys_rst_n == 1'b0)
        bit_cnt <= 4'b0;
    else if ((bit_flag == 1'b1) && (bit_cnt == 4'd9))
        bit_cnt <= 4'b0;
    else if ((bit_flag == 1'b1) && (work_en == 1'b1))
        bit_cnt <= bit_cnt + 1'b1;

// tx: 按照 UART 协议逐位输出（起始位 0，数据位 LSB 先发，停止位 1）
always @(posedge sys_clk or negedge sys_rst_n)
    if (sys_rst_n == 1'b0)
        tx <= 1'b1;  // 空闲状态为高电平
    else if (bit_flag == 1'b1)
        case (bit_cnt)
            0: tx <= 1'b0;       // 起始位
            1: tx <= pi_data[0];
            2: tx <= pi_data[1];
            3: tx <= pi_data[2];
            4: tx <= pi_data[3];
            5: tx <= pi_data[4];
            6: tx <= pi_data[5];
            7: tx <= pi_data[6];
            8: tx <= pi_data[7];
            9: tx <= 1'b1;       // 停止位
            default: tx <= 1'b1;
        endcase

endmodule
