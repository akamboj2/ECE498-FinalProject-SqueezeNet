module input_RAM(
	input [9:0] addr,
	output [7:0] data);

	
	parameter ADDR_WIDTH = 10;
  	parameter DATA_WIDTH =  8;
	logic [ADDR_WIDTH-1:0] addr_reg;
				
	// RAM definition				
	logic [2**ADDR_WIDTH-1:0] RAM[0:DATA_WIDTH-1];

	initial
	begin
		 $readmemh("C:/Users/Abhi Kamboj/ECE498-Project-SqueezeNet/hardware/RAM.txt", RAM);
	end

	assign data = RAM[addr];

endmodule  

