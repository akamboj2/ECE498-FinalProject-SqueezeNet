module input_RAM(
	input [9:0] addr,
	output [7:0] data);

	
	parameter ADDR_WIDTH = 10;
  	parameter DATA_WIDTH =  8;
	logic [ADDR_WIDTH-1:0] addr_reg;
				
	// RAM definition				
	parameter [0:2**ADDR_WIDTH-1][DATA_WIDTH-1:0] RAM;
	
	
	initial begin : RAM_INIT
		for (integer i=0; i<2**ADDR_WIDTH; i++)
      			RAM[i] <=  {DATA_WIDTH{1'b0}};
		RAM[0:3]='{8'd1, 8'd2, 8'd3, 8'd4};
	end
	assign data = RAM[addr];

endmodule  