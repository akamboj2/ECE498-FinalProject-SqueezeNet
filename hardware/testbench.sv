module testbench();
timeunit 10ns;
timeprecision 1ns;

logic reset, Clk, ld_MAC, ld_output;


always begin : CLOCK_GENERATION
 
 #1 Clk = ~Clk;

end
 
initial begin : CLOCK_INITIALIZATION
	Clk = 0;
end

fire f(.*);


initial begin : TEST_VECTORS

#2 reset = 0; ld_MAC = 0; ld_output = 0;
#2 reset = 1;
#2 reset = 0;

#2 ld_MAC = 1;
#2 ld_MAC = 0;

#2 ld_output = 1;
#2 ld_output = 0;

//testing accumulate
#2 ld_MAC = 1;
#2 ld_MAC = 0;

#2 ld_output = 1;
#2 ld_output = 0;

#2 ld_MAC = 1;
#2 ld_MAC = 0;

#2 ld_output = 1;
#2 ld_output = 0;
end

endmodule