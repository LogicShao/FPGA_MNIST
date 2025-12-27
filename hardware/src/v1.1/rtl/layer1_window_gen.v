`timescale 1ns/1ps

module layer1_window_gen #(
    parameter IMG_WIDTH = 28,  // 鍥剧墖瀹藉害
    parameter DATA_WIDTH = 8   // 鏁版嵁浣嶅
)(
    input wire clk,
    input wire rst_n,
    input wire valid_in,             // 杈撳叆鏁版嵁鏈夋晥
    input wire signed [DATA_WIDTH-1:0] din, // 瀹炴椂杈撳叆鐨勪竴涓儚绱?(Newest Pixel)

    // ============================================
    // 杈撳嚭锛?x5 鐭╅樀绐楀彛 (鍏?25 涓儚绱?
    // 鍛藉悕绾﹀畾锛歸_琛宊鍒?(w_0_0 鏄獥鍙ｅ乏涓婅/鏈€鏃э紝w_4_4 鏄獥鍙ｅ彸涓嬭/鏈€鏂?
    // ============================================
    output wire signed [DATA_WIDTH-1:0] w00, w01, w02, w03, w04, // Row 0 (Top)
    output wire signed [DATA_WIDTH-1:0] w10, w11, w12, w13, w14, // Row 1
    output wire signed [DATA_WIDTH-1:0] w20, w21, w22, w23, w24, // Row 2
    output wire signed [DATA_WIDTH-1:0] w30, w31, w32, w33, w34, // Row 3
    output wire signed [DATA_WIDTH-1:0] w40, w41, w42, w43, w44, // Row 4 (Bottom - current)
    
    output reg window_valid // 鍙湁褰撶獥鍙ｅ畬鍏ㄥ～婊℃湁鏁堟暟鎹椂缃?1
);

    // =========================================================
    // Part 1: Line Buffers (琛岀紦瀛? - 璐熻矗绾靛悜寤惰繜
    // =========================================================
    // 鎴戜滑闇€瑕?4 鏉￠暱寤惰繜绾匡紝姣忔潯寤惰繜 IMG_WIDTH 涓懆鏈?
    // 鏁版嵁娴佸悜锛歞in -> LB3 -> LB2 -> LB1 -> LB0 -> (涓㈠純)
    // 杩欐牱锛宒in 鏄 4 琛岋紝LB3鍑虹殑鏄 3 琛?.. LB0鍑虹殑鏄 0 琛?
    
    reg signed [DATA_WIDTH-1:0] lb0 [0:IMG_WIDTH-1];
    reg signed [DATA_WIDTH-1:0] lb1 [0:IMG_WIDTH-1];
    reg signed [DATA_WIDTH-1:0] lb2 [0:IMG_WIDTH-1];
    reg signed [DATA_WIDTH-1:0] lb3 [0:IMG_WIDTH-1];
    
    // 瀹氫箟 Line Buffer 鐨勮緭鍑虹 (Taps)
    wire signed [DATA_WIDTH-1:0] row0_out = lb0[IMG_WIDTH-1];
    wire signed [DATA_WIDTH-1:0] row1_out = lb1[IMG_WIDTH-1];
    wire signed [DATA_WIDTH-1:0] row2_out = lb2[IMG_WIDTH-1];
    wire signed [DATA_WIDTH-1:0] row3_out = lb3[IMG_WIDTH-1];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 澶嶄綅娓呴浂 (鍦‵PGA涓繖閮ㄥ垎閫昏緫鍏跺疄鍙互鐪佺暐锛屼緷璧栧垵濮嬪€?
            for(i=0; i<IMG_WIDTH; i=i+1) begin
                lb0[i] <= 0; lb1[i] <= 0; lb2[i] <= 0; lb3[i] <= 0;
            end
        end else if (valid_in) begin
            // 绉讳綅閫昏緫锛氬皢鏁版嵁鎺ㄥ叆绉讳綅瀵勫瓨鍣?
            // 杩欐槸涓€涓畝鍗曠殑 Shift Register 閾?
            
            // 1. 鍐呴儴绉讳綅 (闄や簡绗竴涓厓绱?
            for(i=IMG_WIDTH-1; i>0; i=i-1) begin
                lb3[i] <= lb3[i-1];
                lb2[i] <= lb2[i-1];
                lb1[i] <= lb1[i-1];
                lb0[i] <= lb0[i-1];
            end
            
            // 2. 绾ц仈杈撳叆 (Chain connection)
            lb3[0] <= din;        // 褰撳墠鏁版嵁杩?LB3
            lb2[0] <= row3_out;   // LB3 鐨勫熬宸磋繘 LB2
            lb1[0] <= row2_out;   // LB2 鐨勫熬宸磋繘 LB1
            lb0[0] <= row1_out;   // LB1 鐨勫熬宸磋繘 LB0
        end
    end

    // =========================================================
    // Part 2: Window Registers (绐楀彛瀵勫瓨鍣? - 璐熻矗妯悜寤惰繜
    // =========================================================
    // 鎴戜滑闇€瑕?5 涓皬鐨勭Щ浣嶅瘎瀛樺櫒锛堥暱搴︿负 5锛夛紝鍒嗗埆瀵瑰簲绐楀彛鐨?5 琛?
    
    reg signed [DATA_WIDTH-1:0] win_row0 [0:4];
    reg signed [DATA_WIDTH-1:0] win_row1 [0:4];
    reg signed [DATA_WIDTH-1:0] win_row2 [0:4];
    reg signed [DATA_WIDTH-1:0] win_row3 [0:4];
    reg signed [DATA_WIDTH-1:0] win_row4 [0:4];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // 娓呴浂閫昏緫鐪佺暐
        end else if (valid_in) begin
            // 姣忎釜绐楀彛琛岄兘鍦ㄥ仛绉讳綅: [0]<=[1], [1]<=[2]... 
            // 娉ㄦ剰锛氳繖閲屾垜浠畾涔?index 4 涓烘渶鏂帮紝0 涓烘渶鑰?
            
            // Row 4 (Bottom) - 鏉ヨ嚜瀹炴椂杈撳叆
            win_row4[4] <= din;
            win_row4[3] <= win_row4[4]; win_row4[2] <= win_row4[3]; win_row4[1] <= win_row4[2]; win_row4[0] <= win_row4[1];

            // Row 3 - 鏉ヨ嚜 LB3 杈撳嚭 (娉ㄦ剰瑕佺敤 row3_out 涔嬪墠鐨勬暟鎹紝鍗?lb3[IMG_WIDTH-1] 鏄垰鍚愬嚭鏉ョ殑)
            // 淇锛氭垜浠鎺ョ殑鏄?LineBuffer 鍒氬垰鍚愬嚭鏉ョ殑閭ｄ釜鏁?
            win_row3[4] <= row3_out; 
            win_row3[3] <= win_row3[4]; win_row3[2] <= win_row3[3]; win_row3[1] <= win_row3[2]; win_row3[0] <= win_row3[1];

            // Row 2
            win_row2[4] <= row2_out;
            win_row2[3] <= win_row2[4]; win_row2[2] <= win_row2[3]; win_row2[1] <= win_row2[2]; win_row2[0] <= win_row2[1];

            // Row 1
            win_row1[4] <= row1_out;
            win_row1[3] <= win_row1[4]; win_row1[2] <= win_row1[3]; win_row1[1] <= win_row1[2]; win_row1[0] <= win_row1[1];

            // Row 0 (Top)
            win_row0[4] <= row0_out;
            win_row0[3] <= win_row0[4]; win_row0[2] <= win_row0[3]; win_row0[1] <= win_row0[2]; win_row0[0] <= win_row0[1];
        end
    end

    // =========================================================
    // Part 3: 璁℃暟鍣ㄤ笌 Valid 鎺у埗 logic
    // =========================================================
    reg [9:0] x_cnt; // 鍒楄鏁?
    reg [9:0] y_cnt; // 琛岃鏁?

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_cnt <= 0;
            y_cnt <= 0;
            window_valid <= 0;
        end else if (valid_in) begin
            // 鍧愭爣鏇存柊閫昏緫
            if (x_cnt == IMG_WIDTH - 1) begin
                x_cnt <= 0;
                y_cnt <= y_cnt + 1;
            end else begin
                x_cnt <= x_cnt + 1;
            end

            // Valid 鍒ゆ柇閫昏緫锛氭棤 Padding 鍗风Н (Valid Convolution)
            // 鍙湁褰撴垜浠湪绗?4 琛?(y_cnt >= 4) 涓旂 4 鍒?(x_cnt >= 4) 涔嬪悗锛岀獥鍙ｆ墠鍏呮弧浜嗘潵鑷浘鍍忓唴閮ㄧ殑鏁版嵁
            if (y_cnt >= 4 && x_cnt >= 4) begin
                window_valid <= 1;
            end else begin
                window_valid <= 0;
            end
        end else begin
            window_valid <= 0; // 濡傛灉杈撳叆鏆傚仠锛岃緭鍑烘湁鏁堟€т篃鏆傚仠
        end
    end

    // =========================================================
    // Part 4: 杈撳嚭杩炵嚎
    // =========================================================
    // 灏嗗瘎瀛樺櫒鏄犲皠鍒拌緭鍑虹鍙?
    
    assign w00 = win_row0[0]; assign w01 = win_row0[1]; assign w02 = win_row0[2]; assign w03 = win_row0[3]; assign w04 = win_row0[4];
    assign w10 = win_row1[0]; assign w11 = win_row1[1]; assign w12 = win_row1[2]; assign w13 = win_row1[3]; assign w14 = win_row1[4];
    assign w20 = win_row2[0]; assign w21 = win_row2[1]; assign w22 = win_row2[2]; assign w23 = win_row2[3]; assign w24 = win_row2[4];
    assign w30 = win_row3[0]; assign w31 = win_row3[1]; assign w32 = win_row3[2]; assign w33 = win_row3[3]; assign w34 = win_row3[4];
    assign w40 = win_row4[0]; assign w41 = win_row4[1]; assign w42 = win_row4[2]; assign w43 = win_row4[3]; assign w44 = win_row4[4];

endmodule
