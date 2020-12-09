import arrays::*;

module tb;
    always #20 clk = ~clk; //generate clock

    int counter;
    real index;
    logic [7:0] val; 

    logic [7:0] PE_inputs [8:0];
    logic [7:0] PE_weights [8:0];
    logic [7:0] PE_outputs [8:0];
    logic [7:0] 3x3_out;
    logic [7:0] bias3x3;

    logic [7:0] output_s [];
    logic [7:0] output_ [];

    logic_da lookup_input[real];
    logic_da lookup_weight[real];
    logic_da lookup_output[real];

    fire dut(.reset(rst),
            .Clk(clk),
            .in_input(PE_inputs),
            .ld_MAC(~rst),
            .in_weight(PE_weights),
            .out_PE(PE_outputs),
            .PE_added(3x3_out),
            .bias3x3(bias3x3));

    initial begin
        clk <= 0;

        counter <= 0;

        rst = 1'b1;
        #40;
        rst = 0;
        #40;
        bias3x3 <= 0;

        val = -8'd128;
        index = -8;
        for(int i = 0; i < 256; i++)begin
            lookup_input[index] = new[1];
            lookup_input[index] = val;
            index = index + 0.015625;
            val = val + 8'd1;
        end

        val = -8'd128;
        index = -8;
        for(int i = 0; i < 256; i++)begin
            lookup_output[index] = new[1];
            lookup_output[index] = val;
            index = index + 0.0625;
            val = val + 8'd1;
        end

        val = 8'd0;
        index = 0;
        for(int i = 0; i < 256; i++)begin
            lookup_weight[index] = new[1];
            lookup_weight[index] = val;
            index = index + 0.00390625;
            val = val + 8'd1;
        end




        for(int i =0; i < 32; i++)begin 
            for(int j = 0; j < 256; j++)begin//for loop for first convolution
                for(int k = 0; k < 9; k++)begin
                    PE_inputs[k] = lookup_input[arrays::inputs[j*9+k]];
                    PE_weights[k] = lookup_weight[arrays::weight_s[i*256+j]];
                end
                #40;
                counter <= counter + 1;
            end
            for(int w = 0; w < 9; w++)begin
                PE_weights[w] = lookup_weight[arrays::bias_s[i]];
                PE_inputs[w] = lookup_input[1];
            end
            #40;
            counter <= counter + 1;
            for(int k = 0; k < 9; k++)begin
                output_s[i*9 + k] = new[1]
                output_s[i*9 + k] = PE_outputs[k];
            end
            rst <= 1'b1;
            #40;
            rst <= 1'b0;
        end

        //have to modify the first layer outputs to feed into other layers since clipping is different

        for(int i = 0; i < 128; i++)begin
            for(int j = 0; j < 32; j++)begin//for loop for the 1x1 expand
                for(int k = 0; k < 9; k++)begin
                    PE_inputs[k] = output_s[j*9+k];
                    PE_weights[k] = lookup_weight[arrays::weight_1x1[i*32+j]];
                end
                #40;
                counter <= counter + 1;    
            end
            for(int w = 0; w < 9; w++)begin
                PE_weights[w] = lookup_weight[arrays::bias_1x1[i]];
                PE_inputs[w] = lookup_input[1];
            end
            #40;
            counter <= counter + 1;
            for(int k = 0; k < 9; k++)begin
                output_[i*9 + k] = new[1]
                output_[i*9 + k] = PE_outputs[k];
            end
            rst <= 1'b1;
            #40;
            rst <= 1'b0;
        end

        for(int i = 0; i < 128; i++)begin//for loop for the 3x3 expand
            for(int j = 0; j < 9; j++)begin
                for(int k = 0; w < 32; k++)begin
                    for(int w = 0; w < 9; w++)begin
                        PE_weights[w] = lookup_weight[arrays::weight3x3[i*32*9+k*9+w]];
                    end
                    if(j==0)begin
                        PE_inputs[0] = lookup_input[0];
                        PE_inputs[1] = lookup_input[0];
                        PE_inputs[2] = lookup_input[0];
                        PE_inputs[3] = lookup_input[0];
                        PE_inputs[4] = output_s[k*9];
                        PE_inputs[5] = output_s[k*9+1];
                        PE_inputs[6] = lookup_input[0];
                        PE_inputs[7] = output_s[k*9+3];
                        PE_inputs[8] = output_s[k*9+4];
                    end
                    else if(j==1)begin
                        PE_inputs[0] = lookup_input[0];
                        PE_inputs[1] = lookup_input[0];
                        PE_inputs[2] = lookup_input[0];
                        PE_inputs[3] = output_s[k*9];
                        PE_inputs[4] = output_s[k*9+1];
                        PE_inputs[5] = output_s[k*9+2];
                        PE_inputs[6] = output_s[k*9+3];
                        PE_inputs[7] = output_s[k*9+4];
                        PE_inputs[8] = output_s[k*9+5];
                    end
                    else if(j==2)begin
                        PE_inputs[0] = lookup_input[0];
                        PE_inputs[1] = lookup_input[0];
                        PE_inputs[2] = lookup_input[0];
                        PE_inputs[3] = output_s[k*9+1];
                        PE_inputs[4] = output_s[k*9+2];
                        PE_inputs[5] = lookup_input[0];
                        PE_inputs[6] = output_s[k*9+4];
                        PE_inputs[7] = output_s[k*9+5];
                        PE_inputs[8] = lookup_input[0];
                    end
                    else if(j==3)begin
                        PE_inputs[0] = lookup_input[0];
                        PE_inputs[1] = output_s[k*9];
                        PE_inputs[2] = output_s[k*9+1];
                        PE_inputs[3] = lookup_input[0];
                        PE_inputs[4] = output_s[k*9+3];
                        PE_inputs[5] = output_s[k*9+4];
                        PE_inputs[6] = lookup_input[0];
                        PE_inputs[7] = output_s[k*9+6];
                        PE_inputs[8] = output_s[k*9+7];
                    end
                    else if(j==4)begin
                        PE_inputs[0] = output_s[k*9];
                        PE_inputs[1] = output_s[k*9+1];
                        PE_inputs[2] = output_s[k*9+2];
                        PE_inputs[3] = output_s[k*9+3];
                        PE_inputs[4] = output_s[k*9+4];
                        PE_inputs[5] = output_s[k*9+5];
                        PE_inputs[6] = output_s[k*9+6];
                        PE_inputs[7] = output_s[k*9+7];
                        PE_inputs[8] = output_s[k*9+8];
                    end
                    else if(j==5)begin
                        PE_inputs[0] = output_s[k*9+1];
                        PE_inputs[1] = output_s[k*9+2];
                        PE_inputs[2] = lookup_input[0];
                        PE_inputs[3] = output_s[k*9+4];
                        PE_inputs[4] = output_s[k*9+5];
                        PE_inputs[5] = lookup_input[0];
                        PE_inputs[6] = output_s[k*9+7];
                        PE_inputs[7] = output_s[k*9+8];
                        PE_inputs[8] = lookup_input[0];
                    end
                    else if(j==6)begin
                        PE_inputs[0] = lookup_input[0];
                        PE_inputs[1] = output_s[k*9+3];
                        PE_inputs[2] = output_s[k*9+4];
                        PE_inputs[3] = lookup_input[0];
                        PE_inputs[4] = output_s[k*9+6];
                        PE_inputs[5] = output_s[k*9+7];
                        PE_inputs[6] = lookup_input[0];
                        PE_inputs[7] = lookup_input[0];
                        PE_inputs[8] = lookup_input[0];
                    end
                    else if(j==7)begin
                        PE_inputs[0] = output_s[k*9+3];
                        PE_inputs[1] = output_s[k*9+4];
                        PE_inputs[2] = output_s[k*9+5];
                        PE_inputs[3] = output_s[k*9+6];
                        PE_inputs[4] = output_s[k*9+7];
                        PE_inputs[5] = output_s[k*9+8];
                        PE_inputs[6] = lookup_input[0];
                        PE_inputs[7] = lookup_input[0];
                        PE_inputs[8] = lookup_input[0];
                    end
                    else if(j==8)begin
                        PE_inputs[0] = output_s[k*9+4];
                        PE_inputs[1] = output_s[k*9+5];
                        PE_inputs[2] = lookup_input[0];
                        PE_inputs[3] = output_s[k*9+7];
                        PE_inputs[4] = output_s[k*9+8];
                        PE_inputs[5] = lookup_input[0];
                        PE_inputs[6] = lookup_input[0];
                        PE_inputs[7] = lookup_input[0];
                        PE_inputs[8] = lookup_input[0];
                    end
                    #40;
                    counter <= counter + 1;
                end
                bias3x3 = lookup_weight[arrays::bias_3x3[i]];
                #40;
                bias3x3 = 8'b0;
                counter <= counter + 1;
                output_3x3[i*9+j] = new[1];
                output_3x3[i*9+j] = 3x3_out; 
                rst <= 1'b1;
                #40;
                rst <= 1'b0;
            end
        end

        //compare the ideal outputs 
        // for(int i = 0; i < 11385; i++)begin
        //     if(output_[i] != lookup_output[arrays::outputs[i]])begin
        //            $display(); 
        //     end
        // end
    end