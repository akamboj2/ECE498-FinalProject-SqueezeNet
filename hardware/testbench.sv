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


initial begin : TEST_VECTORS

#2 reset = 0;
#2 reset = 1;
#2 reset = 0;

end

endmodule