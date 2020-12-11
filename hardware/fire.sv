parameter nbits = 15; //note this is actually the precision in bits + 1 (includes zero)
//^note this is only in this file. make sure to change ram file and testbench if you change this value!

module fire(
	input logic reset, Clk, 
	input logic ld_MAC, //these are control signals, currently i'm sending from testbench
	input logic signed [7:0] in_input[9],
	input logic signed [7:0] in_weight[9], //only used for 3x3 conv where all PEs make 1 pixel
	input logic signed [7:0] bias3x3,

	output logic signed [7:0] PE_added,
	output logic signed [7:0] out_PE[9]
	//if this module was a part of a large nn there could be signals here indicating where it is, when to run, etc.
);

logic signed [nbits:0] out_PE_[9];
logic signed [nbits:0] PE_added_;

always_comb begin
	for(int i = 0; i < 9; i++)begin
		out_PE[i] = {out_PE_[i][23],out_PE_[i][14:8]};
	end 
end
//here we should be doing 3*4 = 12
//PE PE1(.in_input(in_ram_out), .in_weight(w_ram_out), .out_activation(out_activation), .ld_MAC(ld_MAC),.*);
PE PE_array[9](
	.in_input(in_input), //9 different inputs for each PE
	.in_weight(in_weight), //single weight broadcasted to all PEs
	.out_activation(out_PE_), //9 different outputs for each PE
	.ld_MAC(ld_MAC), //single ld signal broadcasted
	.*
);
logic signed out_bias_3x3;

multiply m_bias_3x3 (bias3x3, 8'd16 , out_bias_3x3);

assign PE_added_ = out_PE_[0] + out_PE_[1] + out_PE_[2] + out_PE_[3] + out_PE_[4] + out_PE_[5] 
	+ out_PE_[6] + out_PE_[7] + out_PE_[8] + out_bias_3x3;
	
assign PE_added = PE_added_[15:8];

endmodule



module PE(
	input logic signed [7:0] in_input, in_weight,
	input logic Clk, reset, ld_MAC,
	output logic signed [nbits:0] out_activation);
//logic signed [7:0] in_input_, in_weight_;
logic signed [nbits:0] out_MAC, mult_out, add_out;

//register_ input_reg(.data_in(in_input), .data_out(in_input_), .ld(1'b1), .*);
//register_ weight_reg(.data_in(in_weight), .data_out(in_weight_), .ld(1'b1), .*);
multiply m(in_weight, in_input, mult_out);
add a(mult_out, out_MAC, add_out);
register hold_MAC(.data_in(add_out), .data_out(out_MAC), .ld(ld_MAC), .*); //to initialize to zero hit reset
RELU activation(out_MAC, out_activation);
endmodule

module RELU(
	input [nbits:0] in,
	output [nbits:0] out );
assign out = in[nbits] ? 0 : in; //if in is negative = 0, else it's just in
endmodule

module add(
	input logic signed [nbits:0] a, b,
	output logic signed [nbits:0] result);
assign result = a + b;//(real'(a) + real'(b));
endmodule

module multiply(
	input logic signed [7:0] a,b,
	output logic signed [nbits:0] result);
assign result = a*b;//(real'(a) * real'(b));
endmodule

module register(
	input logic signed [nbits:0] data_in,
	input logic ld, reset, Clk,
	output logic signed [nbits:0] data_out);
always_ff @(posedge Clk)
begin
	if (reset)
		data_out <=0;
	else if(ld==1'b1)
		data_out <= data_in;
	else
		data_out <= data_out;
end
endmodule 

module register_(
	input logic signed [7:0] data_in,
	input logic ld, reset, Clk,
	output logic signed [7:0] data_out);
always_ff @(posedge Clk)
begin
	if (reset)
		data_out <=0;
	else if(ld==1'b1)
		data_out <= data_in;
	else
		data_out <= data_out;
end
endmodule 