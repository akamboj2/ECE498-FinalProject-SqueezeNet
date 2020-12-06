module testbench();
timeunit 10ns;
timeprecision 1ns;

logic reset, Clk;


always begin : CLOCK_GENERATION
 
 #1 Clk = ~Clk;

end
 
initial begin : CLOCK_INITIALIZATION
	Clk = 0;
end

fire f(.*);

parameter [3:0] data [4] = {0.19778589 * 16, 0.55186864 * 16, 0.73500254 * 16, -0.69549231 * 16};
//{4, 8, 12, -0.69549231 * 16};

//x_hat should be = -0.69549231
real result;
assign result = x_hat/16/16;

initial begin : TEST_VECTORS

#2 reset = 0;
#2 reset = 1;
#2 reset = 0;

end

endmodule