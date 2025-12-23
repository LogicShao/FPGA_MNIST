module vector_dot_product (
    input wire clk,
    input wire rst_n,

    // 控制信号
    input wire valid_in,     // 当输入数据有效时置 1
    input wire sop,          // Start of Packet: 标志向量的开始（用于清空累加器）
    input wire eop,          // End of Packet:   标志向量的结束（用于输出结果）

    // 数据输入 (8-bit 有符号数)
    input wire signed [7:0] data_a, // 比如像素值
    input wire signed [7:0] data_b, // 比如权重值

    // 结果输出
    output reg signed [31:0] result, // 最终的点乘和
    output reg result_valid          // 结果有效标志
  );

  // ==========================================
  // 内部信号
  // ==========================================
  reg signed [15:0] product;      // 乘法结果 (8位*8位=16位)
  reg signed [31:0] accumulator;  // 累加寄存器

  // 延迟一拍的控制信号（为了匹配流水线）
  reg eop_d;

  // ==========================================
  // 第一级流水线：乘法 (Multiplier)
  // ==========================================
  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
    begin
      product <= 0;
      eop_d <= 0;
    end
    else if (valid_in)
    begin
      // 硬件乘法器 DSP Block 会在这里生成
      product <= data_a * data_b;

      // 传递控制信号
      eop_d <= eop;
    end
    else
    begin
      // 如果输入无效，保持流水线干净（可选）
      eop_d <= 0;
    end
  end

  // ==========================================
  // 第二级流水线：累加 (Accumulator)
  // ==========================================
  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
    begin
      accumulator <= 0;
      result <= 0;
      result_valid <= 0;
    end
    else
    begin
      // 这里我们简化处理：假设sop到来时，不仅输入第一个数据，
      // 同时也意味着我们要把之前的累加器清零（或者重新赋值为新的乘积）

      // 注意：因为乘法有一拍延迟，所以累加逻辑要稍微小心
      // 真实的 sop 信号其实也需要延迟一拍来匹配 product，
      // 但为了让你容易理解，我们这里写一个简化版的逻辑：

      // 如果上一拍有数据进来...
      // (在实际工程中，valid_in 最好也延迟一拍叫 valid_d)

      // 这里的逻辑是一个简化的 MAC 行为：
      // 只要乘法器有输出，我们就累加

      // 修正：我们需要根据 sop 来决定是“清零后加”还是“继续加”
      // 考虑到流水线延迟，我们通常会把 sop 延迟一拍使用
    end
  end

  // ----------------------------------------------------
  // 重新整理：标准流水线写法
  // ----------------------------------------------------
  reg sop_d, valid_d;

  always @(posedge clk or negedge rst_n)
  begin
    if (!rst_n)
    begin
      product <= 0;
      accumulator <= 0;
      result <= 0;
      result_valid <= 0;
      sop_d <= 0;
      valid_d <= 0;
      eop_d <= 0;
    end
    else
    begin
      // --- Stage 1: 乘法 & 控制信号延迟 ---
      if (valid_in)
      begin
        product <= data_a * data_b;
      end
      sop_d <= sop;     // 延迟一拍，因为 product 也是延迟一拍出来的
      eop_d <= eop;
      valid_d <= valid_in;

      // --- Stage 2: 累加 ---
      if (valid_d)
      begin
        if (sop_d)
        begin
          // 如果是这组向量的第一个数，直接覆盖累加器
          accumulator <= {{16{product[15]}}, product}; // 符号扩展
        end
        else
        begin
          // 否则，累加
          accumulator <= accumulator + {{16{product[15]}}, product};
        end

        // 如果是这组向量的最后一个数，输出结果
        if (eop_d)
        begin
          // 注意：如果是 sop 和 eop 同时为 1 (向量长度为1)，逻辑也没问题
          // 但这里的 result 实际上要等下一拍才出来，或者是组合逻辑输出
          // 我们这里用寄存器输出，所以会有总共 2 拍的 latency

          // 特殊处理：如果是 EOP，我们需要把当前的加进去之后再输出
          // 上面的 accumulator 更新是非阻塞赋值，
          // 所以我们要输出 "旧 accumulator + 当前 product"
          // 为了代码整洁，我们在下一拍输出 accumulator 的新值
          result_valid <= 1;
        end
        else
        begin
          result_valid <= 0;
        end
      end
      else
      begin
        result_valid <= 0;
      end

      // 在 result_valid 有效的时候，result 更新为当前的 accumulator
      if (eop_d && valid_d)
      begin
        // 这里的 accumulator 还没更新完成（非阻塞），
        // 所以真正的结果是 accumulator + product
        result <= (sop_d ? 0 : accumulator) + {{16{product[15]}}, product};
      end
    end
  end

endmodule
